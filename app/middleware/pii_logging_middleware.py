"""
Request logging middleware with mandatory PII redaction.

Never logs raw ``request.body`` or ``request.headers`` — all request metadata
passes through :func:`~app.core.pii_redactor.prepare_request_log_context` first.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import logger
from app.core.pii_redactor import prepare_request_log_context


class PiiLoggingMiddleware(BaseHTTPMiddleware):
    """Logs redacted request metadata at DEBUG; no raw body or headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if logger.isEnabledFor(logging.DEBUG):
            content_length = request.headers.get("content-length")
            body_length: int | None = None
            if content_length and content_length.isdigit():
                body_length = int(content_length)

            ctx = prepare_request_log_context(
                request.method,
                request.url.path,
                request.headers,
                query_params=request.query_params,
                body_length=body_length,
            )
            logger.debug("Incoming request %s", ctx)

        return await call_next(request)
