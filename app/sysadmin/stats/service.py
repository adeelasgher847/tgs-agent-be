"""Stats aggregation service — reads sys_request_log, writes sys_request_stats."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    """Return (start_inclusive, end_exclusive) for a 'YYYY-MM' string."""
    import calendar

    year, mon = int(month[:4]), int(month[5:7])
    last_day = calendar.monthrange(year, mon)[1]
    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    end = datetime(year, mon, last_day, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return start, end


def get_overview(db: Session, month: str) -> dict:
    from app.models.sysadmin_log import SysRequestLog, SysRequestStats

    start, end = _month_bounds(month)

    # Read from pre-aggregated stats when available
    stats_rows = (
        db.execute(
            select(SysRequestStats).where(
                SysRequestStats.month == month,
                SysRequestStats.tenant_id.is_(None),
            )
        )
        .scalars()
        .all()
    )

    if stats_rows:
        total = sum(r.total_requests for r in stats_rows)
        success = sum(r.success_count for r in stats_rows)
        errors = sum(r.error_count for r in stats_rows)
        total_dur = sum(r.total_duration_ms for r in stats_rows)
        p95 = max((r.p95_duration_ms or 0 for r in stats_rows), default=0)
        avg = int(total_dur / total) if total else 0
        active_tenants_count = _active_tenants(db, start, end)
        return _build_overview(total, success, errors, avg, p95, active_tenants_count)

    # Fallback: scan raw logs (slow — only for current month before nightly recompute)
    return _compute_overview_raw(db, start, end)


def _compute_overview_raw(db: Session, start: datetime, end: datetime) -> dict:
    from sqlalchemy import Integer, cast

    from app.models.sysadmin_log import SysRequestLog

    base = select(SysRequestLog).where(
        SysRequestLog.created_at >= start,
        SysRequestLog.created_at <= end,
    )
    rows = db.execute(base).scalars().all()
    if not rows:
        return _build_overview(0, 0, 0, 0, 0, 0)

    total = len(rows)
    success = sum(1 for r in rows if 200 <= r.status_code < 300)
    errors = sum(1 for r in rows if r.status_code >= 500)
    durations = sorted(r.duration_ms for r in rows)
    avg = int(sum(durations) / total) if total else 0
    p95_idx = int(0.95 * total)
    p95 = durations[min(p95_idx, total - 1)]
    active_tenants = len({r.tenant_id for r in rows if r.tenant_id})
    return _build_overview(total, success, errors, avg, p95, active_tenants)


def _active_tenants(db: Session, start: datetime, end: datetime) -> int:
    from app.models.sysadmin_log import SysRequestLog

    result = db.execute(
        select(func.count(func.distinct(SysRequestLog.tenant_id))).where(
            SysRequestLog.created_at >= start,
            SysRequestLog.created_at <= end,
            SysRequestLog.tenant_id.is_not(None),
        )
    ).scalar()
    return result or 0


def _build_overview(total, success, errors, avg, p95, active_tenants) -> dict:
    success_rate = round(success / total * 100, 2) if total else 0
    failure_rate = round(errors / total * 100, 2) if total else 0
    return {
        "total_requests": total,
        "success_count": success,
        "error_count": errors,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "avg_response_ms": avg,
        "p95_response_ms": p95,
        "active_tenants": active_tenants,
    }


def get_monthly_trend(db: Session, months: int = 6) -> list[dict]:
    from app.models.sysadmin_log import SysRequestLog

    rows = db.execute(
        text("""
            SELECT
                to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM') AS month,
                COUNT(*) AS total_requests,
                SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS error_count
            FROM sysrequestlog
            WHERE created_at >= NOW() - (INTERVAL '1 month' * :months)
            GROUP BY 1
            ORDER BY 1 ASC
        """),
        {"months": months},
    ).mappings().all()
    return [dict(r) for r in rows]


def recompute_stats(db: Session, month: str) -> None:
    """Full recompute of sys_request_stats for a month — uses PERCENTILE_CONT."""
    from app.models.sysadmin_log import SysRequestStats

    start, end = _month_bounds(month)

    rows = db.execute(
        text("""
            SELECT
                path,
                method,
                tenant_id,
                COUNT(*) AS total_requests,
                SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS error_count,
                SUM(duration_ms) AS total_duration_ms,
                AVG(duration_ms)::INTEGER AS avg_duration_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::INTEGER AS p95_duration_ms
            FROM sysrequestlog
            WHERE created_at >= :start AND created_at <= :end

            GROUP BY path, method, tenant_id
        """),
        {"start": start, "end": end},
    ).mappings().all()

    # Delete old stats for this month
    db.execute(
        text("DELETE FROM sysrequeststats WHERE month = :month"),  # noqa: auto-derived tablename
        {"month": month},
    )

    for r in rows:
        stat = SysRequestStats(
            month=month,
            path=r["path"],
            method=r["method"],
            tenant_id=r["tenant_id"],
            total_requests=r["total_requests"],
            success_count=r["success_count"],
            error_count=r["error_count"],
            total_duration_ms=r["total_duration_ms"],
            avg_duration_ms=r["avg_duration_ms"],
            p95_duration_ms=r["p95_duration_ms"],
            computed_at=datetime.now(timezone.utc),
        )
        db.add(stat)
    db.commit()


def get_errors(
    db: Session,
    month: str | None,
    tenant_id: str | None,
    source: str | None,
    path: str | None,
    page: int,
    limit: int,
) -> dict:
    from app.models.sysadmin_log import SysRequestLog

    stmt = select(SysRequestLog).where(SysRequestLog.status_code >= 500)

    if month:
        start, end = _month_bounds(month)
        stmt = stmt.where(SysRequestLog.created_at >= start, SysRequestLog.created_at <= end)
    if tenant_id:
        import uuid as _uuid
        stmt = stmt.where(SysRequestLog.tenant_id == _uuid.UUID(tenant_id))
    if source:
        stmt = stmt.where(SysRequestLog.source == source)
    if path:
        stmt = stmt.where(SysRequestLog.path.ilike(f"{path}%"))

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
    rows = db.execute(stmt.order_by(SysRequestLog.created_at.desc()).offset((page - 1) * limit).limit(limit)).scalars().all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [_serialize_log(r) for r in rows],
    }


def get_errors_csv(
    db: Session,
    month: str | None,
    tenant_id: str | None,
    source: str | None,
    path: str | None,
) -> str:
    result = get_errors(db, month, tenant_id, source, path, page=1, limit=10000)
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["id", "created_at", "path", "method", "status_code", "tenant_id", "source", "error_message"],
    )
    writer.writeheader()
    for item in result["items"]:
        writer.writerow({k: _csv_safe(item.get(k, "")) for k in writer.fieldnames})
    return buf.getvalue()


_FORMULA_CHARS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: object) -> str:
    """Prefix formula-injection characters with a tab to neutralise spreadsheet attacks."""
    s = str(value) if value is not None else ""
    if s and s[0] in _FORMULA_CHARS:
        return "\t" + s
    return s


def get_top_endpoints(db: Session, month: str, limit: int) -> list[dict]:
    from app.models.sysadmin_log import SysRequestStats

    stmt = (
        select(
            SysRequestStats.path,
            SysRequestStats.method,
            func.sum(SysRequestStats.total_requests).label("total_requests"),
        )
        .where(SysRequestStats.month == month, SysRequestStats.tenant_id.is_(None))
        .group_by(SysRequestStats.path, SysRequestStats.method)
        .order_by(func.sum(SysRequestStats.total_requests).desc())
        .limit(limit)
    )
    rows = db.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def get_slow_endpoints(db: Session, month: str, threshold_ms: int) -> list[dict]:
    from app.models.sysadmin_log import SysRequestStats

    stmt = (
        select(
            SysRequestStats.path,
            SysRequestStats.method,
            func.sum(SysRequestStats.total_requests).label("total_requests"),
            (func.sum(SysRequestStats.total_duration_ms) / func.sum(SysRequestStats.total_requests)).label("avg_duration_ms"),
            func.max(SysRequestStats.p95_duration_ms).label("p95_duration_ms"),
        )
        .where(SysRequestStats.month == month, SysRequestStats.tenant_id.is_(None))
        .group_by(SysRequestStats.path, SysRequestStats.method)
        .having(
            func.sum(SysRequestStats.total_duration_ms) / func.sum(SysRequestStats.total_requests) >= threshold_ms
        )
        .order_by((func.sum(SysRequestStats.total_duration_ms) / func.sum(SysRequestStats.total_requests)).desc())
    )
    rows = db.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def _serialize_log(r: Any) -> dict:
    curl = f"curl -X {r.method} 'https://api.tgs.ai{r.path}' -H 'Authorization: Bearer <token>'"
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "path": r.path,
        "method": r.method,
        "status_code": r.status_code,
        "duration_ms": r.duration_ms,
        "tenant_id": str(r.tenant_id) if r.tenant_id else None,
        "source": r.source,
        "error_message": r.error_message,
        "stack_trace": r.stack_trace,
        "request_id": r.request_id,
        "ip_address": r.ip_address,
        "country": r.country,
        "curl_repro": curl,
    }
