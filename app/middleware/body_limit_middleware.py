"""Enforces a global request body size limit and returns 413 on excess."""

from __future__ import annotations

import json
from typing import Callable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.middleware.request_id_middleware import new_request_id, request_id_from_scope

_10MB = 10 * 1024 * 1024  # bytes


def _resolve_request_id(scope: Scope) -> str:
    rid = request_id_from_scope(scope)
    return rid or new_request_id()


def _413_payload(request_id: str) -> bytes:
    body = {
        "error": {
            "code": "payload_too_large",
            "message": "Request body exceeds the 10 MB limit.",
            "requestId": request_id,
        }
    }
    return json.dumps(body).encode()


class BodyLimitMiddleware:
    """Rejects requests whose Content-Length or streamed body exceeds *max_bytes*."""

    def __init__(self, app: ASGIApp, max_bytes: int = _10MB) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        request_id = _resolve_request_id(scope)

        # Fast-path: reject immediately when Content-Length alone exceeds limit.
        if content_length and int(content_length) > self.max_bytes:
            await self._send_413(send, request_id)
            return

        received_bytes = 0
        limit_exceeded = False

        async def limited_receive() -> dict:
            nonlocal received_bytes, limit_exceeded
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    limit_exceeded = True
            return message

        async def guarded_send(message: dict) -> None:
            if limit_exceeded and message["type"] == "http.response.start":
                await self._send_413(send, request_id)
                raise _BodyLimitExceeded()
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyLimitExceeded:
            pass

    @staticmethod
    async def _send_413(send: Send, request_id: str) -> None:
        body = _413_payload(request_id)
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-request-id", request_id.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyLimitExceeded(Exception):
    """Internal sentinel — never escapes the middleware."""
