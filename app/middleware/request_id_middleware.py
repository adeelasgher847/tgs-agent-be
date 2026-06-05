"""Attaches a nanoid request ID to every request via request.state and X-Request-ID response header."""

from __future__ import annotations

from nanoid import generate
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Scope

_NANOID_SIZE = 21  # default nanoid length — URL-safe, shorter than UUID


def new_request_id() -> str:
    return generate(size=_NANOID_SIZE)


def _request_id_from_headers(scope: Scope) -> str | None:
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.lower() == b"x-request-id":
            return raw_value.decode("latin-1")
    return None


def request_id_from_scope(scope: Scope) -> str:
    """
    Read the request ID from ASGI scope state (set by :class:`RequestIdMiddleware`).

    Safe for pure-ASGI middleware (e.g. body limit) that runs inside the stack
    after the request ID has been assigned.
    """
    if scope.get("type") != "http":
        return ""
    state = scope.get("state")
    if state is not None:
        rid = getattr(state, "request_id", None)
        if rid:
            return str(rid)
    return _request_id_from_headers(scope) or ""


def get_request_id(request: Request) -> str:
    """
    Return the canonical request ID for this request.

    Prefer ``request.state.request_id`` (nanoid from :class:`RequestIdMiddleware`).
    Fall back to the incoming header, then generate a new nanoid.
    """
    rid = getattr(request.state, "request_id", None)
    if rid:
        return str(rid)
    header_rid = request.headers.get("x-request-id")
    if header_rid:
        return header_rid
    return new_request_id()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generates a nanoid request ID, stores it on request.state.request_id, and echoes it in X-Request-ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Honour an upstream-supplied ID (e.g. from a load balancer); otherwise generate.
        request_id = request.headers.get("x-request-id") or new_request_id()
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
