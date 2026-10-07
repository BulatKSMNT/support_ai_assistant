import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from app.config import Settings
from app.documents import extract_file
from app.errors import ServiceError
from app.metrics import Metrics
from app.middleware import BodyLimitMiddleware
from app.retrieval import VectorIndex
from app.schemas import (
    DocumentInfo,
    DocumentInput,
    DraftRequest,
    DraftResponse,
    IngestResult,
    SearchRequest,
    Source,
)
from app.service import Assistant
from app.storage import Repository
from app.upstream import Ollama

logger = logging.getLogger("assistant")
bearer = HTTPBearer(auto_error=False)
STATIC = Path(__file__).parent / "static"


async def authorize(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
):
    expected = request.app.state.settings.api_key.get_secret_value()
    if not credentials or not hmac.compare_digest(
        credentials.credentials.encode(), expected.encode()
    ):
        raise ServiceError("unauthorized", "Нужен ключ доступа к сервису.", 401)


def create_app(settings: Settings | None = None, ollama_transport=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        config = settings or Settings()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        repository = Repository(config.data_dir / "knowledge.sqlite3", config.index_signature)
        metrics = Metrics()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.upstream_timeout, connect=5),
            transport=ollama_transport,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            vectors = VectorIndex(config) if config.uses_vectors else None
            ollama = Ollama(config, client, on_retry=metrics.retries.inc)
            application.state.settings = config
            application.state.assistant = Assistant(config, repository, ollama, vectors, metrics)
            application.state.metrics = metrics
            try:
                yield
            finally:
                if vectors:
                    await vectors.close()

    application = FastAPI(
        title="Support AI Assistant",
        version="0.1.0",
        lifespan=lifespan,
        description="Подготовка черновиков с источниками. Независимый сервис, без подключения к CRM.",
    )

    @application.middleware("http")
    async def request_metadata(request: Request, call_next):
        started = perf_counter()
        try:
            request_id = str(UUID(request.headers.get("X-Request-ID", "")))
        except ValueError:
            request_id = str(uuid4())
        request.state.request_id = request_id
        # Multipart parsing also happens before the route's byte limit.
        length = request.headers.get("content-length", "")
        max_bytes = request.app.state.settings.max_upload_bytes + 64_000
        if length and (not length.isdigit() or int(length) > max_bytes):
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {"code": "request_too_large", "message": "Запрос слишком большой."},
                    "request_id": request_id,
                },
            )
        else:
            response = await call_next(request)
        elapsed = perf_counter() - started
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics = request.app.state.metrics
        metrics.requests.labels(route, str(response.status_code)).inc()
        metrics.latency.labels(route).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            )
            if not request.url.path.startswith(("/docs", "/redoc"))
            else "frame-ancestors 'none'"
        )
        if request.url.path.startswith(("/v1", "/metrics", "/health/ready")):
            response.headers["Cache-Control"] = "no-store"
        logger.info(
            json.dumps(
                {
                    "event": "http",
                    "request_id": request_id,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                }
            )
        )
        return response

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else {}
        if exc.status == 429:
            headers["Retry-After"] = "3"
        return JSONResponse(
            status_code=exc.status,
            headers=headers,
            content={
                "error": {"code": exc.code, "message": exc.message},
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo uploaded text, personal data, or a raw upstream body in errors.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Проверьте поля запроса.",
                    "fields": [{"path": list(e["loc"]), "message": e["msg"]} for e in exc.errors()],
                },
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC), name="static")

    @application.get("/health/live", tags=["Health"])
    async def live():
        return {"status": "ok"}

    @application.get("/health/ready", dependencies=[Depends(authorize)], tags=["Health"])
    async def ready(request: Request):
        assistant = request.app.state.assistant
        model_status = await assistant.ollama.ready()
        vectors_ready = await assistant.vectors.ready() if assistant.vectors else True
        good = model_status["ready"] and vectors_ready
        return JSONResponse(
            status_code=200 if good else 503,
            content={
                "status": "ready" if good else "not_ready",
                "search_mode": assistant.settings.search_mode,
                "generation_mode": assistant.settings.generation_mode,
                "documents": len(await asyncio.to_thread(assistant.repository.documents)),
                "ollama": model_status,
                "vector_index_ready": vectors_ready,
            },
        )

    @application.get("/metrics", dependencies=[Depends(authorize)], tags=["Health"])
    async def metrics(request: Request):
        return Response(
            content=generate_latest(request.app.state.metrics.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])

    @router.get("/documents", response_model=list[DocumentInfo], tags=["Knowledge"])
    async def list_documents(request: Request):
        return await asyncio.to_thread(request.app.state.assistant.repository.documents)

    @router.post("/documents", response_model=IngestResult, tags=["Knowledge"])
    async def ingest_document(data: DocumentInput, request: Request):
        return await request.app.state.assistant.ingest(data)

    @router.post("/documents/upload", response_model=IngestResult, tags=["Knowledge"])
    async def upload_document(
        request: Request,
        file: Annotated[UploadFile, File()],
        source_key: Annotated[str, Form()],
        title: Annotated[str, Form()],
    ):
        config = request.app.state.settings
        try:
            content = await file.read(config.max_upload_bytes + 1)
            if len(content) > config.max_upload_bytes:
                raise ServiceError("file_too_large", "Файл превышает допустимый размер.", 413)
            passages = await asyncio.to_thread(
                extract_file, file.filename or "", content, config.max_document_chars
            )
            data = DocumentInput(
                source_key=source_key, title=title, text="\n\n".join(p.text for p in passages)
            )
            return await request.app.state.assistant.ingest(data, passages)
        except ValidationError as exc:
            raise ServiceError(
                "invalid_document", "Проверьте название и код документа.", 422
            ) from exc
        finally:
            await file.close()

    @router.delete("/documents/{document_id}", status_code=204, tags=["Knowledge"])
    async def delete_document(document_id: UUID, request: Request):
        await request.app.state.assistant.delete(str(document_id))
        return Response(status_code=204)

    @router.post("/search", response_model=list[Source], tags=["Search"])
    async def search(data: SearchRequest, request: Request):
        return await request.app.state.assistant.search(
            data.query, mode=data.mode, limit=data.limit
        )

    @router.post("/drafts", response_model=DraftResponse, tags=["Assistant"])
    async def draft(data: DraftRequest, request: Request):
        return await request.app.state.assistant.draft(data, request.state.request_id)

    application.include_router(router)
    application.add_middleware(BodyLimitMiddleware)
    return application


app = create_app()
