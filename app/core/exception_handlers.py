"""Global FastAPI exception handlers with PII-safe API responses."""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.error_responses import build_api_error_payload
from app.core.logger import logger
from app.core.pii_redactor import redact_pii


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = _get_request_id(request)
    if (
        exc.status_code == 429
        and isinstance(exc.detail, dict)
        and exc.detail.get("code") == "rate_limit_exceeded"
    ):
        error = dict(exc.detail)
        error.setdefault("requestId", request_id)
        return JSONResponse(
            status_code=429,
            content={"error": error},
            headers={
                **(getattr(exc, "headers", None) or {}),
                "X-Request-ID": request_id,
            },
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=build_api_error_payload(exc.status_code, exc.detail, request_id=request_id),
        headers={
            **(getattr(exc, "headers", None) or {}),
            "X-Request-ID": request_id,
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = _get_request_id(request)
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        redact_pii(exc.errors()),
    )
    return JSONResponse(
        status_code=422,
        content=build_api_error_payload(
            422,
            "Request validation failed",
            error_code="validation_error",
            request_id=request_id,
        ),
        headers={"X-Request-ID": request_id},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _get_request_id(request)
    logger.error(
        "Unhandled exception on %s %s (requestId=%s)",
        request.method,
        request.url.path,
        request_id,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content=build_api_error_payload(
            500,
            "An internal error occurred. Please try again later.",
            error_code="internal_error",
            request_id=request_id,
        ),
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
