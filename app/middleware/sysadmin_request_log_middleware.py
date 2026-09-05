"""
Logs SysAdmin Portal requests into sysrequestlog.
Scoped to /sysadmin/* only — never runs on voice, tenant, or health paths.
"""
from __future__ import annotations

import time
import uuid as _uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

_LOG_PREFIX = "/sysadmin"
_SKIP_SUFFIXES = ("/health", "/docs", "/redoc", "/openapi", "/favicon")


class SysAdminRequestLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Only log sysadmin traffic; skip noisy health/doc paths
        if not path.startswith(_LOG_PREFIX) or any(path.startswith(s) for s in _SKIP_SUFFIXES):
            return await call_next(request)

        start = time.monotonic()
        status_code = 500
        error_message: str | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            # Redact full exception text — store only the type name to avoid PII leakage
            error_message = type(exc).__name__
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            # Fire-and-forget: don't block the response on the DB write
            _write_log(
                request=request,
                path=path,
                method=request.method,
                status_code=status_code,
                duration_ms=duration_ms,
                error_message=error_message,
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
) -> None:
    try:
        from app.db.session import SessionLocal
        from app.models.sysadmin_log import SysRequestLog

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
