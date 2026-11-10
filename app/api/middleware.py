"""Small ASGI safety middleware for bounded request bodies."""

from collections.abc import Awaitable, Callable
from typing import Any

from app.api.errors import error_response

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]


class RequestBodyLimitMiddleware:
    """Reject HTTP request bodies before buffering more than the limit."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body = bytearray()
        more_body = True
        messages: list[AsgiMessage] = []
        while more_body:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                await self.app(scope, receive, send)
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                response = error_response(
                    413,
                    "request_too_large",
                    "Request body exceeds the configured size limit.",
                    details={"max_body_bytes": self.max_body_bytes},
                )
                await response(scope, receive, send)
                return
            body.extend(chunk)
            more_body = message.get("more_body", False)

        delivered = False

        async def replay() -> AsgiMessage:
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
