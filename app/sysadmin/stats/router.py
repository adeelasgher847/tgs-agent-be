"""SysAdmin stats endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.sysadmin.deps import get_current_sysadmin, get_sysadmin_db, require_super_admin
from app.sysadmin.stats.service import (
    get_errors,
    get_errors_csv,
    get_monthly_trend,
    get_overview,
    get_slow_endpoints,
    get_top_endpoints,
    recompute_stats,
)

router = APIRouter(prefix="/stats", tags=["SysAdmin Stats"])

_current_month = lambda: datetime.now(timezone.utc).strftime("%Y-%m")


@router.get("/overview")
async def overview(
    month: str = Query(default=None),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    m = month or _current_month()
    current = get_overview(db, m)

    # Prior month delta
    year, mon = int(m[:4]), int(m[5:7])
    prior_mon = mon - 1 if mon > 1 else 12
    prior_year = year if mon > 1 else year - 1
    prior = get_overview(db, f"{prior_year:04d}-{prior_mon:02d}")

    def delta(key: str) -> float | None:
        c, p = current.get(key, 0), prior.get(key, 0)
        if not p:
            return None
        return round((c - p) / p * 100, 1)

    return {
        "month": m,
        "metrics": current,
        "deltas": {
            "total_requests": delta("total_requests"),
            "success_rate": round((current["success_rate"] - prior["success_rate"]), 2),
            "failure_rate": round((current["failure_rate"] - prior["failure_rate"]), 2),
            "avg_response_ms": delta("avg_response_ms"),
            "p95_response_ms": delta("p95_response_ms"),
            "active_tenants": current["active_tenants"] - prior["active_tenants"],
        },
    }


@router.get("/monthly-trend")
async def monthly_trend(
    months: int = Query(default=6, ge=1, le=24),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    return {"trend": get_monthly_trend(db, months)}


@router.post("/recompute")
async def recompute(
    month: str = Query(default=None),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(require_super_admin),
):
    m = month or _current_month()
    recompute_stats(db, m)
    return {"status": "ok", "month": m}


@router.get("/errors")
async def errors(
    month: str = Query(default=None),
    tenant_id: str = Query(default=None),
    source: str = Query(default=None),
    path: str = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    return get_errors(db, month, tenant_id, source, path, page, limit)


@router.get("/errors/export")
async def errors_export(
    month: str = Query(default=None),
    tenant_id: str = Query(default=None),
    source: str = Query(default=None),
    path: str = Query(default=None),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    csv_data = get_errors_csv(db, month, tenant_id, source, path)
    filename = f"errors_{month or 'all'}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/top-endpoints")
async def top_endpoints(
    month: str = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    m = month or _current_month()
    return {"month": m, "endpoints": get_top_endpoints(db, m, limit)}


@router.get("/slow-endpoints")
async def slow_endpoints(
    month: str = Query(default=None),
    threshold_ms: int = Query(default=1000, ge=100),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(get_current_sysadmin),
):
    m = month or _current_month()
    return {"month": m, "threshold_ms": threshold_ms, "endpoints": get_slow_endpoints(db, m, threshold_ms)}
