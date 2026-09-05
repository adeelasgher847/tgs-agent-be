"""
Logs every API request into sys_request_log for the SysAdmin Portal.
Captures tenant_id + user_id from request.state (set by ApiKeyMiddleware).
Runs AFTER the response so duration_ms is accurate.
"""
from __future__ import annotations

import time
import traceback
import uuid as _uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Paths that should never be logged (health checks, static assets)
_SKIP_PREFIXES = ("/health", "/docs", "/redoc", "/openapi", "/favicon")


class SysAdminRequestLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        start = time.monotonic()
        response: Response | None = None
        error_message: str | None = None
        stack_trace: str | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            error_message = str(exc)
            stack_trace = traceback.format_exc()
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            _write_log(
                request=request,
                path=path,
                method=request.method,
                status_code=status_code,
                duration_ms=duration_ms,
                error_message=error_message,
                stack_trace=stack_trace,
            )

        return response


def _write_log(
    *,
    request: Request,
    path: str,
    method: str,
    status_code: int,
    duration_ms: int,
    error_message: str | None,
    stack_trace: str | None,
) -> None:
    try:
        from app.db.session import SessionLocal
        from app.models.sysadmin_user import SysRequestLog

        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = getattr(request.state, "user_id", None)
        request_id = getattr(request.state, "request_id", None)
        ip = request.client.host if request.client else None

        entry = SysRequestLog(
            tenant_id=_to_uuid(tenant_id),
            user_id=_to_uuid(user_id),
            path=path[:500],
            method=method[:10],
            status_code=status_code,
            duration_ms=duration_ms,
            source="backend",
            error_message=error_message,
            stack_trace=stack_trace,
            request_id=str(request_id)[:64] if request_id else None,
            ip_address=ip,
        )

        db = SessionLocal()
        try:
            db.add(entry)
            db.commit()
        finally:
            db.close()
    except Exception:
        # Never let logging failures affect the API response
        pass


def _to_uuid(value: object) -> _uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, _uuid.UUID):
        return value
    try:
        return _uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None
