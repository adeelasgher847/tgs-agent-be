"""Dynamic CORS for the public Web SDK endpoint.

The global ``CORSMiddleware`` only allows origins listed in the static
``ALLOWED_ORIGINS`` env var. But /api/v1/sdk/public-call-token must be
embeddable from whatever domain a tenant adds to their ``allowed_domains``
table — a value that only exists at the application/DB layer, not at
CORS-middleware startup time. Without this, Starlette's CORSMiddleware
rejects the browser's preflight OPTIONS for any origin outside the static
list before the request ever reaches our handler, so a whitelisted tenant
domain could never actually call the endpoint from a browser.

This middleware reflects the request's Origin for that one path only — it
grants no security by itself. The real authorization boundary is the
allowed_domains check inside app/routers/sdk.py (403 domain_not_allowed).
Must be registered OUTERMOST (added after CORSMiddleware in main.py) so it
intercepts preflight before the static-origin CORSMiddleware can 400 it.
"""
from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

_PUBLIC_SDK_PATHS = frozenset({"/api/v1/sdk/public-call-token"})


class PublicSdkCorsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in _PUBLIC_SDK_PATHS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin")

        if scope.get("method") == "OPTIONS":
            response_headers = [(b"access-control-allow-methods", b"POST, OPTIONS")]
            if origin:
                response_headers.append((b"access-control-allow-origin", origin))
            requested_headers = headers.get(b"access-control-request-headers")
            if requested_headers:
                response_headers.append((b"access-control-allow-headers", requested_headers))
            await send({"type": "http.response.start", "status": 200, "headers": response_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        async def _send(message):
            if message["type"] == "http.response.start" and origin:
                headers_list = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != b"access-control-allow-origin"
                ]
                headers_list.append((b"access-control-allow-origin", origin))
                headers_list.append((b"vary", b"Origin"))
                message = {**message, "headers": headers_list}
            await send(message)

        await self.app(scope, receive, _send)
