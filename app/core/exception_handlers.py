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
    return JSONResponse(
        status_code=exc.status_code,
        content=build_api_error_payload(exc.status_code, exc.detail, request_id=request_id),
        headers={
            **(getattr(exc, "headers", None) or {}),
            "X-Request-ID": request_id,
        },
    )


async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _get_request_id(request)
    return JSONResponse(
        status_code=404,
        content=build_api_error_payload(
            404,
            "Route not found",
            error_code="not_found",
            request_id=request_id,
        ),
        headers={"X-Request-ID": request_id},
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
    fields: list[dict[str, str]] = []
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", ()) if part not in ("body",)]
        path = ".".join(loc) if loc else "(root)"
        fields.append({"path": path, "message": err.get("msg", "Invalid value")})
    return JSONResponse(
        status_code=400,
        content=build_api_error_payload(
            400,
            "Request validation failed",
            error_code="validation_error",
            request_id=request_id,
            extras={"fields": fields} if fields else None,
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
    # 404 for unknown routes — must come after the generic HTTPException handler.
    # Starlette raises a plain HTTPException(404) for missing routes, so
    # http_exception_handler already covers it; not_found_handler is the
    # explicit hook for the 404 status specifically.
    app.add_exception_handler(404, not_found_handler)
