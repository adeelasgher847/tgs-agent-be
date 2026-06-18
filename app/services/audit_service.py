"""
Audit log helper used by all mutation endpoints.

Call log_audit_event() after a successful DB commit to record the event.
The function is intentionally non-raising — a logging failure must never
abort a business operation.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.request_auth import AUTH_METHOD_API_KEY, get_auth_method
from app.models.audit_log import AuditLog


def _extract_ip(request: Request) -> Optional[str]:
    """Return the real client IP, respecting X-Forwarded-For from load balancers."""
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    client = request.client
    return client.host if client else None


def log_audit_event(
    db: Session,
    *,
    request: Request,
    tenant_id: uuid.UUID,
    action: str,
    resource_type: str,
    resource_id: Optional[uuid.UUID] = None,
    old_value: Optional[dict[str, Any]] = None,
    new_value: Optional[dict[str, Any]] = None,
    actor_user_id: Optional[uuid.UUID] = None,
) -> None:
    """
    Append one row to the auditlog table.

    Never raises — errors are logged as warnings so a logging failure
    cannot roll back or block the caller's response.

    Never log audit events for the auditlog table itself (infinite recursion guard):
    this function trusts callers to never pass resource_type='auditlog'.
    """
    try:
        ip = _extract_ip(request)
        user_agent = request.headers.get("user-agent", "")[:512] or None

        api_key_prefix: Optional[str] = None
        if get_auth_method(request) == AUTH_METHOD_API_KEY:
            api_key_prefix = getattr(request.state, "api_key_prefix", None)

        entry = AuditLog(
            tenant_id=tenant_id,
            user_id=actor_user_id,
            actor_api_key_prefix=api_key_prefix,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_value,
            new_value=new_value,
            ip_address=ip,
            user_agent=user_agent,
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        logger.warning(
            "audit_log write failed (action=%s resource=%s): %s",
            action,
            resource_type,
            exc,
        )
        db.rollback()
