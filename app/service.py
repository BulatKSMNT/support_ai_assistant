import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.config import Settings
from app.documents import Passage, chunk_passages, normalize_text
from app.errors import ServiceError
from app.grounding import live_data_requirement, unsupported_numbers
from app.metrics import Metrics
from app.retrieval import VectorIndex, fuse, lexical_search
from app.schemas import DocumentInput, DraftRequest, DraftResponse, IngestResult, Source
from app.storage import Repository
from app.upstream import Ollama

logger = logging.getLogger("assistant")


class Assistant:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        ollama: Ollama,
        vectors: VectorIndex | None,
        metrics: Metrics,
    ):
        self.settings = settings
        self.repository = repository
        self.ollama = ollama
        self.vectors = vectors
        self.metrics = metrics
        self.write_lock = asyncio.Lock()
        self.generation_slots = asyncio.Semaphore(settings.max_parallel_generations)

    async def ingest(
        self, data: DocumentInput, passages: list[Passage] | None = None
    ) -> IngestResult:
        passages = passages or [Passage(normalize_text(data.text))]
        if sum(len(p.text) for p in passages) > self.settings.max_document_chars:
            raise ServiceError("document_too_large", "Документ превышает лимит текста.", 413)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "title": data.title,
                    "url": str(data.source_url) if data.source_url else None,
                    "passages": [(p.text, p.page) for p in passages],
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        async with self.write_lock:
            previous = await asyncio.to_thread(self.repository.document, data.source_key)
            if previous and previous["content_hash"] == fingerprint:
                return IngestResult(document=previous, changed=False)
            identifier = str(uuid5(NAMESPACE_URL, "support-assistant:" + data.source_key))
            pieces = chunk_passages(passages, self.settings.chunk_size, self.settings.chunk_overlap)
            chunks = [
                {
                    "id": str(uuid4()),
                    "document_id": identifier,
                    "ordinal": i,
                    "text": p.text,
                    "page": p.page,
                }
                for i, p in enumerate(pieces)
            ]
            if not chunks or not any(len(c["text"]) >= 20 for c in chunks):
                raise ServiceError("empty_document", "Недостаточно содержательного текста.", 422)
            existing = await asyncio.to_thread(self.repository.chunks)
            old_ids = [c["id"] for c in existing if c["document_id"] == identifier]
            if len(existing) - len(old_ids) + len(chunks) > self.settings.max_total_chunks:
                raise ServiceError("knowledge_limit", "Достигнут лимит базы знаний.", 409)
            # New points are staged. Only the atomic SQLite commit activates them.
            if self.vectors:
                for offset in range(0, len(chunks), 12):
                    batch = chunks[offset : offset + 12]
                    embeddings = await self.ollama.embed(
                        [data.title + "\n" + c["text"] for c in batch]
                    )
                    await self.vectors.upsert(batch, embeddings)
            now = datetime.now(UTC).isoformat()
            document = {
                "id": identifier,
                "source_key": data.source_key,
                "title": data.title,
                "source_url": str(data.source_url) if data.source_url else None,
                "version": previous["version"] + 1 if previous else 1,
                "content_hash": fingerprint,
                "chunk_count": len(chunks),
                "created_at": previous["created_at"] if previous else now,
                "updated_at": now,
            }
            await asyncio.to_thread(self.repository.replace, document, chunks)
            await self._cleanup(old_ids)
            return IngestResult(document=document, changed=True)

    async def _cleanup(self, identifiers: list[str]):
        if self.vectors and identifiers:
            try:
                await self.vectors.delete(identifiers)
            except Exception:
                # Inactive IDs are excluded from every query, even if cleanup fails.
                logger.warning("vector_cleanup_deferred count=%d", len(identifiers))

    async def delete(self, identifier: str):
        async with self.write_lock:
            chunks = await asyncio.to_thread(self.repository.chunks)
            identifiers = [c["id"] for c in chunks if c["document_id"] == identifier]
            deleted = await asyncio.to_thread(self.repository.delete, identifier)
            if not deleted:
                raise ServiceError("document_not_found", "Документ не найден.", 404)
            await self._cleanup(identifiers)

    async def search(self, query: str, *, mode: str | None = None, limit: int | None = None):
        mode = mode or self.settings.search_mode
        limit = limit or self.settings.top_k
        if mode != "lexical" and not self.vectors:
            raise ServiceError("vectors_disabled", "В этом режиме векторный индекс отключён.", 409)
        chunks = await asyncio.to_thread(self.repository.chunks)
        if not chunks:
            return []
        lexical, dense = [], []
        if mode != "vector":
            lexical = await asyncio.to_thread(
                lexical_search, query, chunks, self.settings.candidate_k
            )
        if mode != "lexical":
            # Bound the retrieval query for the embedding model's context window.
            vector = (await self.ollama.embed([query[:1600]], query=True))[0]
            dense = await self.vectors.search(
                vector, [c["id"] for c in chunks], self.settings.candidate_k
            )
        ranking = (
            fuse(lexical, dense) if mode == "hybrid" else lexical if mode == "lexical" else dense
        )
        lookup = {c["id"]: c for c in chunks}
        return [
            Source(
                id=f"S{i}",
                chunk_id=identifier,
                document_id=lookup[identifier]["document_id"],
                title=lookup[identifier]["title"],
                source_key=lookup[identifier]["source_key"],
                source_url=lookup[identifier]["source_url"],
                version=lookup[identifier]["version"],
                page=lookup[identifier]["page"],
                text=lookup[identifier]["text"],
                score=score,
            )
            for i, (identifier, score) in enumerate(ranking[:limit], 1)
        ]

    async def draft(self, request: DraftRequest, request_id: str) -> DraftResponse:
        started = perf_counter()
        try:
            await asyncio.wait_for(self.generation_slots.acquire(), timeout=2)
        except TimeoutError as exc:
            raise ServiceError(
                "busy", "Все места обработки заняты. Попробуйте немного позже.", 429
            ) from exc
        try:
            query = request.question
            if request.lead:
                query += "\n" + request.lead.message + "\n" + " ".join(request.lead.products)
            requirement = live_data_requirement(query)
            if requirement:
                self.metrics.drafts.labels(
                    "insufficient_evidence", self.settings.generation_mode
                ).inc()
                return DraftResponse(
                    request_id=request_id,
                    status="insufficient_evidence",
                    generation_mode=self.settings.generation_mode,
                    draft="",
                    clarifying_questions=[],
                    missing_information=[requirement],
                    sources=[],
                    timings_ms={
                        "search": 0,
                        "generation": 0,
                        "total": round((perf_counter() - started) * 1000, 1),
                    },
                )
            sources = await self.search(query)
            if sources and self.settings.search_mode == "hybrid":
                # top_k is an upper bound, not a requirement to pad the context.
                # With a strong lexical+dense consensus, drop weak single-channel matches.
                cutoff = sources[0].score * self.settings.context_score_ratio
                sources = [source for source in sources if source.score >= cutoff]
            searched = perf_counter()
            self.metrics.stage.labels("search").observe(searched - started)
            status, text, questions = "insufficient_evidence", "", []
            missing = ["В базе знаний не найдено достаточно подходящей информации."]
            if sources and self.settings.generation_mode == "extractive":
                text = "\n\n".join(f"{s.text} [{s.id}]" for s in sources)
                status, missing = (
                    "ready",
                    ["Показаны выдержки. Черновик языковой моделью не создавался."],
                )
            elif sources:
                result = await self.ollama.generate(request, sources)
                questions, missing = result.clarifying_questions, result.missing_information
                if result.answerable:
                    available = {s.id for s in sources}
                    source_text = {s.id: s.text for s in sources}
                    if not result.blocks:
                        raise ServiceError(
                            "invalid_citations", "Модель не указала основания ответа.", 502
                        )
                    rendered, used = [], set()
                    for block in result.blocks:
                        refs = set(block.source_ids)
                        inline = set(re.findall(r"\[(S\d+)\]", block.text))
                        if not refs <= available or not inline <= refs:
                            raise ServiceError(
                                "invalid_citations",
                                "Модель сослалась на неизвестный источник.",
                                502,
                            )
                        evidence = "\n".join(source_text[key] for key in refs)
                        if request.lead:
                            evidence += (
                                "\n" + request.lead.message + "\n" + " ".join(request.lead.products)
                            )
                        if unsupported_numbers(block.text, evidence):
                            raise ServiceError(
                                "unsupported_numeric_claim",
                                "Ответ содержит числовые значения без подтверждения в источниках. "
                                "Подготовьте ответ вручную или уточните запрос.",
                                502,
                            )
                        used |= refs
                        clean = re.sub(r"\[S\d+\]", "", block.text).strip()
                        rendered.append(clean + " " + " ".join(f"[{key}]" for key in sorted(refs)))
                    sources = [source for source in sources if source.id in used]
                    text, status = "\n\n".join(rendered), "ready"
                else:
                    missing = missing or ["Найденные документы не позволяют ответить на вопрос."]
            ended = perf_counter()
            self.metrics.stage.labels("generation").observe(ended - searched)
            self.metrics.drafts.labels(status, self.settings.generation_mode).inc()
            return DraftResponse(
                request_id=request_id,
                status=status,
                generation_mode=self.settings.generation_mode,
                draft=text,
                clarifying_questions=questions,
                missing_information=missing,
                sources=sources,
                timings_ms={
                    "search": round((searched - started) * 1000, 1),
                    "generation": round((ended - searched) * 1000, 1),
                    "total": round((ended - started) * 1000, 1),
                },
            )
        finally:
            self.generation_slots.release()
