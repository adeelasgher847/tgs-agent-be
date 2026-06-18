"""
v2 Audit Events router.

Endpoints (admin RBAC required for all):
  GET  /api/v2/audit-events              — paginated list with filters
  GET  /api/v2/audit-events/{id}         — single event detail
  POST /api/v2/audit-events/export       — streaming CSV export
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.models.audit_log import AuditLog
from app.models.user import User

router = APIRouter(prefix="/audit-events", tags=["audit-events"])

_PAGE_SIZE = 25


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AuditEventOut(BaseModel):
    id: uuid.UUID
    timestamp: datetime
    tenant_id: uuid.UUID
    user_id: Optional[uuid.UUID]
    actor_api_key_prefix: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[uuid.UUID]
    old_value: Optional[Any]
    new_value: Optional[Any]
    ip_address: Optional[str]
    user_agent: Optional[str]

    model_config = {"from_attributes": True}


class PaginatedAuditEvents(BaseModel):
    items: list[AuditEventOut]
    total: int
    page: int
    page_size: int
    has_next: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_filters(
    tenant_id: uuid.UUID,
    action: Optional[str],
    resource_type: Optional[str],
    actor_api_key_prefix: Optional[str],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> list:
    clauses = [AuditLog.tenant_id == tenant_id]
    if action:
        clauses.append(AuditLog.action == action)
    if resource_type:
        clauses.append(AuditLog.resource_type == resource_type)
    if actor_api_key_prefix:
        clauses.append(AuditLog.actor_api_key_prefix == actor_api_key_prefix)
    if date_from:
        clauses.append(AuditLog.timestamp >= date_from)
    if date_to:
        clauses.append(AuditLog.timestamp <= date_to)
    return clauses


# ── GET /audit-events ─────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedAuditEvents)
def list_audit_events(
    page: int = Query(1, ge=1),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    actor_api_key_prefix: Optional[str] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> PaginatedAuditEvents:
    """Return paginated audit events for the authenticated workspace. Admin role required."""
    tenant_id = user.current_tenant_id
    clauses = _build_filters(tenant_id, action, resource_type, actor_api_key_prefix, date_from, date_to)
    where = and_(*clauses)

    offset = (page - 1) * _PAGE_SIZE

    total = db.execute(
        select(AuditLog).where(where)
    ).scalars().all()
    total_count = len(total)

    rows = db.execute(
        select(AuditLog)
        .where(where)
        .order_by(AuditLog.timestamp.desc())
        .offset(offset)
        .limit(_PAGE_SIZE)
    ).scalars().all()

    return PaginatedAuditEvents(
        items=[AuditEventOut.model_validate(r) for r in rows],
        total=total_count,
        page=page,
        page_size=_PAGE_SIZE,
        has_next=(offset + _PAGE_SIZE) < total_count,
    )


# ── GET /audit-events/{id} ────────────────────────────────────────────────────

@router.get("/{event_id}", response_model=AuditEventOut)
def get_audit_event(
    event_id: uuid.UUID,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AuditEventOut:
    """Return a single audit event including old_value and new_value. Admin role required."""
    row = db.execute(
        select(AuditLog).where(
            AuditLog.id == event_id,
            AuditLog.tenant_id == user.current_tenant_id,
        )
    ).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audit event not found")

    return AuditEventOut.model_validate(row)


# ── POST /audit-events/export ─────────────────────────────────────────────────

@router.post("/export")
def export_audit_events(
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    actor_api_key_prefix: Optional[str] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Stream all matching audit events as CSV.

    Columns: id, event_type, resource_type, resource_id, actor,
             ip_address, created_at, old_value, new_value.
    Admin role required.
    """
    tenant_id = user.current_tenant_id
    clauses = _build_filters(tenant_id, action, resource_type, actor_api_key_prefix, date_from, date_to)

    rows = db.execute(
        select(AuditLog)
        .where(and_(*clauses))
        .order_by(AuditLog.timestamp.asc())
    ).scalars().all()

    def _csv_stream():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "event_type", "resource_type", "resource_id",
            "actor", "ip_address", "created_at", "old_value", "new_value",
        ])
        yield buf.getvalue()

        for row in rows:
            buf = io.StringIO()
            writer = csv.writer(buf)
            actor = row.actor_api_key_prefix or str(row.user_id or "")
            writer.writerow([
                str(row.id),
                row.action,
                row.resource_type or "",
                str(row.resource_id) if row.resource_id else "",
                actor,
                str(row.ip_address) if row.ip_address else "",
                row.timestamp.isoformat() if row.timestamp else "",
                json.dumps(row.old_value) if row.old_value is not None else "",
                json.dumps(row.new_value) if row.new_value is not None else "",
            ])
            yield buf.getvalue()

    return StreamingResponse(
        _csv_stream(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_events.csv"},
    )
