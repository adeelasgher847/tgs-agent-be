"""Dynamic CORS for the public Web SDK endpoints.

The global ``CORSMiddleware`` only allows origins listed in the static
``ALLOWED_ORIGINS`` env var. But /api/v1/sdk/public-call-token must be
embeddable from whatever domain a tenant adds to their ``allowed_domains``
table — a value that only exists at the application/DB layer, not at
CORS-middleware startup time. Without this, Starlette's CORSMiddleware
rejects the browser's preflight OPTIONS for any origin outside the static
list before the request ever reaches our handler, so a whitelisted tenant
domain could never actually call the endpoint from a browser.

/api/v1/sdk/demo/{token}/call-token has an even looser requirement: it is
origin-unrestricted by design (any site holding the token may use it, see
app/models/call_flow_demo_link.py), so it needs the same Origin reflection
with no allowlist check at all — that check lives entirely inside the
handler (token/expiry/budget validation), not here.

This middleware reflects the request's Origin for these paths only — it
grants no security by itself. The real authorization boundary is the
allowed_domains check (public-call-token) or the demo-link validation
(demo/{token}/call-token) inside app/routers/sdk.py.
Must be registered OUTERMOST (added after CORSMiddleware in main.py) so it
intercepts preflight before the static-origin CORSMiddleware can 400 it.
"""
from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

_PUBLIC_SDK_PATHS = frozenset({"/api/v1/sdk/public-call-token"})


def _is_public_sdk_path(path: str) -> bool:
    if path in _PUBLIC_SDK_PATHS:
        return True
    # /api/v1/sdk/demo/{token}/call-token — token is opaque, so match by shape
    # rather than adding every issued token to a static set.
    if path.startswith("/api/v1/sdk/demo/") and path.endswith("/call-token"):
        return True
    return False


class PublicSdkCorsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_public_sdk_path(scope.get("path", "")):
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
