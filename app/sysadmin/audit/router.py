"""Audit log endpoint."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.sysadmin.deps import get_current_sysadmin, get_sysadmin_db

router = APIRouter(prefix="/audit-log", tags=["SysAdmin Audit"])


@router.get("")
async def get_audit_log(
    from_dt: datetime = Query(default=None, alias="from"),
    to_dt: datetime = Query(default=None, alias="to"),
    tenant_schema: str = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    from app.models.sysadmin_user import SysAuditLog

    stmt = select(SysAuditLog)

    if from_dt:
        stmt = stmt.where(SysAuditLog.created_at >= from_dt)
    if to_dt:
        stmt = stmt.where(SysAuditLog.created_at <= to_dt)
    if tenant_schema:
        stmt = stmt.where(SysAuditLog.tenant_schema == tenant_schema)

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
    rows = (
        db.execute(stmt.order_by(SysAuditLog.created_at.desc()).offset((page - 1) * limit).limit(limit))
        .scalars()
        .all()
    )

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [
            {
                "id": str(r.id),
                "created_at": r.created_at.isoformat(),
                "admin_email": r.admin_email,
                "action": r.action,
                "source_page": r.source_page,
                "tenant_schema": r.tenant_schema,
                "details": r.details,
                "ip_address": r.ip_address,
            }
            for r in rows
        ],
    }
