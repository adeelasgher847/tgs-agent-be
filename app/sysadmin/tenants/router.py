"""Tenant list + per-tenant drill-down endpoints."""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from app.sysadmin.deps import get_current_sysadmin, get_sysadmin_db, validate_tenant_schema

router = APIRouter(prefix="/tenants", tags=["SysAdmin Tenants"])


def _30d_bounds():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    return start, end


@router.get("")
async def list_tenants(
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    from app.models.sysadmin_user import SysRequestLog
    from app.models.tenant import Tenant

    start, end = _30d_bounds()

    tenants = db.execute(select(Tenant)).scalars().all()

    # Aggregate request stats per tenant in one query
    agg_rows = db.execute(
        select(
            SysRequestLog.tenant_id,
            func.count().label("total"),
            func.avg(SysRequestLog.duration_ms).label("avg_ms"),
            func.sum(
                case((SysRequestLog.status_code >= 500, 1), else_=0)
            ).label("errors_5xx"),
            func.max(SysRequestLog.created_at).label("last_request"),
        )
        .where(SysRequestLog.created_at >= start, SysRequestLog.created_at <= end)
        .group_by(SysRequestLog.tenant_id)
    ).mappings().all()

    stats_by_tenant = {str(r["tenant_id"]): r for r in agg_rows}

    result = []
    for t in tenants:
        s = stats_by_tenant.get(str(t.id), {})
        result.append({
            "id": str(t.id),
            "name": t.name,
            "schema": f"tenant_{t.id.hex}",
            "status": getattr(t, "subscription_status", "active"),
            "requests_30d": s.get("total", 0),
            "avg_response_ms": round(s["avg_ms"]) if s.get("avg_ms") else 0,
            "errors_5xx": int(s.get("errors_5xx") or 0),
            "last_request_at": s["last_request"].isoformat() if s.get("last_request") else None,
        })

    return {"tenants": result}


@router.get("/{schema}/stats")
async def tenant_stats(
    schema: str,
    month: str = Query(default=None),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    schema = validate_tenant_schema(schema)
    from app.sysadmin.stats.service import _month_bounds
    from app.models.sysadmin_user import SysRequestLog, SysRequestStats
    import uuid as _uuid

    m = month or datetime.now(timezone.utc).strftime("%Y-%m")

    # Resolve tenant_id from schema
    tenant_id = _tenant_id_from_schema(schema, db)

    stmt = (
        select(
            SysRequestStats.path,
            SysRequestStats.method,
            SysRequestStats.total_requests,
            SysRequestStats.avg_duration_ms,
            SysRequestStats.p95_duration_ms,
            SysRequestStats.error_count,
        )
        .where(SysRequestStats.month == m, SysRequestStats.tenant_id == tenant_id)
        .order_by(SysRequestStats.total_requests.desc())
    )
    rows = db.execute(stmt).mappings().all()
    return {"month": m, "schema": schema, "endpoints": [dict(r) for r in rows]}


@router.get("/{schema}/errors")
async def tenant_errors(
    schema: str,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    schema = validate_tenant_schema(schema)
    from app.sysadmin.stats.service import get_errors

    tenant_id = str(_tenant_id_from_schema(schema, db))
    return get_errors(db, month=None, tenant_id=tenant_id, source=None, path=None, page=page, limit=limit)


@router.get("/{schema}/users")
async def tenant_users(
    schema: str,
    month: str = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    schema = validate_tenant_schema(schema)
    from app.models.sysadmin_user import SysRequestLog
    from app.models.user import User
    from app.sysadmin.stats.service import _month_bounds
    import uuid as _uuid

    m = month or datetime.now(timezone.utc).strftime("%Y-%m")
    start, end = _month_bounds(m)
    tenant_id = _tenant_id_from_schema(schema, db)

    rows = db.execute(
        select(
            SysRequestLog.user_id,
            func.count().label("total_requests"),
            func.sum(
                case((SysRequestLog.status_code.between(200, 299), 1), else_=0)
            ).label("success_count"),
            func.sum(
                case((SysRequestLog.status_code >= 500, 1), else_=0)
            ).label("errors_5xx"),
            func.max(SysRequestLog.created_at).label("last_request"),
        )
        .where(
            SysRequestLog.tenant_id == tenant_id,
            SysRequestLog.created_at >= start,
            SysRequestLog.created_at <= end,
            SysRequestLog.user_id.is_not(None),
        )
        .group_by(SysRequestLog.user_id)
        .order_by(func.count().desc())
        .limit(limit)
    ).mappings().all()

    # Enrich with emails
    user_ids = [r["user_id"] for r in rows if r["user_id"]]
    users = {str(u.id): u.email for u in db.execute(select(User).where(User.id.in_(user_ids))).scalars().all()}

    return {
        "month": m,
        "users": [
            {
                "user_id": str(r["user_id"]),
                "email": users.get(str(r["user_id"]), "unknown"),
                "total_requests": r["total_requests"],
                "success_count": int(r["success_count"] or 0),
                "errors_5xx": int(r["errors_5xx"] or 0),
                "success_rate": round(int(r["success_count"] or 0) / r["total_requests"] * 100, 1) if r["total_requests"] else 0,
                "last_request_at": r["last_request"].isoformat() if r["last_request"] else None,
            }
            for r in rows
        ],
    }


def _tenant_id_from_schema(schema: str, db: Session):
    """Extract UUID from schema name tenant_<hex32> and verify tenant exists."""
    import uuid as _uuid
    from sqlalchemy import select
    from app.models.tenant import Tenant

    if schema == "public":
        raise HTTPException(status_code=400, detail="Cannot drill down on public schema")

    hex_part = schema.replace("tenant_", "")
    try:
        tenant_id = _uuid.UUID(hex_part)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid schema format")

    tenant = db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return tenant_id
