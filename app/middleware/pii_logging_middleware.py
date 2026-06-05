"""
Request logging middleware with mandatory PII redaction.

Logs one INFO line per request: method, path, status, duration_ms, requestId.
Never logs raw ``request.body`` or ``request.headers`` — all request metadata
passes through :func:`~app.core.pii_redactor.prepare_request_log_context` first.
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import logger
from app.core.pii_redactor import prepare_request_log_context


class PiiLoggingMiddleware(BaseHTTPMiddleware):
    """Logs redacted request metadata at INFO; no raw body or header values."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()

        # Optional verbose debug context (headers, query params, body length).
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

        response: Response = await call_next(request)

        duration_ms = round((time.monotonic() - start) * 1000)
        request_id = getattr(request.state, "request_id", "")

        logger.info(
            "%s %s %s %dms requestId=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        return response
