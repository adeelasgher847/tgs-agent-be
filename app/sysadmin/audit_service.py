"""Records every sysadmin action to sys_audit_log."""
from __future__ import annotations

from sqlalchemy.orm import Session


def record_audit(
    *,
    db: Session,
    admin: object,
    action: str,
    source_page: str | None = None,
    tenant_schema: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    from app.models.sysadmin_user import SysAuditLog

    entry = SysAuditLog(
        admin_id=admin.id,
        admin_email=admin.email,
        action=action,
        source_page=source_page,
        tenant_schema=tenant_schema,
        details=details,
        ip_address=ip_address,
    )
    db.add(entry)
    db.commit()
