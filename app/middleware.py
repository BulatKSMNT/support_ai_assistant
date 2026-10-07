from uuid import uuid4

from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    """Bound the raw stream before JSON/multipart parsing, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        limit = scope["app"].state.settings.max_upload_bytes + 64_000
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                request_id = str(uuid4())
                response = JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "request_too_large",
                            "message": "Запрос слишком большой.",
                        },
                        "request_id": request_id,
                    },
                    headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)
