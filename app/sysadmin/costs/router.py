"""AWS Cost Explorer + tenant billing profitability endpoints."""
from __future__ import annotations

import functools
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.sysadmin.deps import get_sysadmin_db, require_super_admin

router = APIRouter(prefix="/costs", tags=["SysAdmin Costs"])

# 6-hour in-memory cache for AWS Cost Explorer results ($0.01/call)
_aws_cache: dict[str, tuple[float, dict]] = {}
_AWS_CACHE_TTL = 6 * 3600


def _get_cached_aws(month: str) -> dict | None:
    entry = _aws_cache.get(month)
    if entry and (time.time() - entry[0]) < _AWS_CACHE_TTL:
        return entry[1]
    return None


def _set_cached_aws(month: str, data: dict) -> None:
    _aws_cache[month] = (time.time(), data)


@router.get("/aws")
async def aws_costs(
    month: str = Query(default=None),
    _admin=Depends(require_super_admin),
):
    m = month or datetime.now(timezone.utc).strftime("%Y-%m")

    enabled = getattr(settings, "AWS_COST_EXPLORER_ENABLED", False)
    if not enabled:
        return {
            "configured": False,
            "setup_guide": {
                "steps": [
                    "Enable AWS Cost Explorer in the AWS Billing console (account level)",
                    "Add ce:GetCostAndUsage and ce:GetDimensionValues to the backend IAM role",
                    "Set AWS_COST_EXPLORER_ENABLED=true in backend environment",
                ],
                "docs_url": "https://docs.aws.amazon.com/cost-management/latest/userguide/ce-enable.html",
            },
        }

    cached = _get_cached_aws(m)
    if cached:
        return {**cached, "cached": True}

    try:
        import boto3

        client = boto3.client("ce", region_name=getattr(settings, "AWS_REGION_NAME", "us-east-1"))
        year, mon = int(m[:4]), int(m[5:7])
        import calendar

        last_day = calendar.monthrange(year, mon)[1]
        start_date = f"{year:04d}-{mon:02d}-01"
        end_date = f"{year:04d}-{mon:02d}-{last_day:02d}"

        response = client.get_cost_and_usage(
            TimePeriod={"Start": start_date, "End": end_date},
            Granularity="MONTHLY",
            Metrics=["BlendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )

        results = response["ResultsByTime"]
        services = []
        total = 0.0
        for result in results:
            for group in result.get("Groups", []):
                service = group["Keys"][0]
                amount = float(group["Metrics"]["BlendedCost"]["Amount"])
                total += amount
                services.append({"service": service, "cost_usd": round(amount, 4)})

        services.sort(key=lambda x: x["cost_usd"], reverse=True)

        data = {
            "configured": True,
            "month": m,
            "total_cost_usd": round(total, 4),
            "services_count": len(services),
            "top_services": services[:10],
            "all_services": services,
            "cached": False,
        }
        _set_cached_aws(m, data)
        return data

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AWS Cost Explorer error: {exc}")


@router.get("/tenants")
async def tenant_billing(
    month: str = Query(default=None),
    db: Session = Depends(get_sysadmin_db),
    _admin=Depends(require_super_admin),
):
    m = month or datetime.now(timezone.utc).strftime("%Y-%m")

    from app.models.sysadmin_user import SysRequestLog
    from app.models.tenant import Tenant

    # Total requests across all tenants this month
    from app.sysadmin.stats.service import _month_bounds

    start, end = _month_bounds(m)

    total_requests_result = db.execute(
        select(func.count()).where(
            SysRequestLog.created_at >= start,
            SysRequestLog.created_at <= end,
        )
    ).scalar() or 1  # avoid division by zero

    # Per-tenant request counts
    per_tenant = db.execute(
        select(
            SysRequestLog.tenant_id,
            func.count().label("requests"),
        )
        .where(
            SysRequestLog.created_at >= start,
            SysRequestLog.created_at <= end,
            SysRequestLog.tenant_id.is_not(None),
        )
        .group_by(SysRequestLog.tenant_id)
    ).mappings().all()

    # AWS total spend
    aws_data = _get_cached_aws(m)
    aws_total = aws_data["total_cost_usd"] if aws_data else 0.0

    # Stripe revenue per tenant
    revenue = _get_stripe_revenue(db, m)

    tenants = {str(t.id): t for t in db.execute(select(Tenant)).scalars().all()}

    rows = []
    for r in per_tenant:
        tid = str(r["tenant_id"])
        share = r["requests"] / total_requests_result
        est_cost = round(aws_total * share, 4)
        charged = revenue.get(tid, 0.0)
        net = round(charged - est_cost, 4)

        rows.append({
            "tenant_id": tid,
            "name": tenants[tid].name if tid in tenants else "Unknown",
            "requests_30d": r["requests"],
            "share_pct": round(share * 100, 2),
            "est_aws_cost_usd": est_cost,
            "charged_usd": charged,
            "net_usd": net,
            "profitable": net >= 0,
        })

    rows.sort(key=lambda x: x["est_aws_cost_usd"], reverse=True)
    return {"month": m, "aws_total_usd": aws_total, "tenants": rows}


def _get_stripe_revenue(db: Session, month: str) -> dict[str, float]:
    """Sum successful billing_transactions per tenant for the month."""
    from sqlalchemy import text as _text

    from app.sysadmin.stats.service import _month_bounds

    start, end = _month_bounds(month)
    try:
        rows = db.execute(
            _text("""
                SELECT tenant_id::text, SUM(amount) AS total
                FROM billing_transactions
                WHERE status = 'success'
                  AND created_at >= :start
                  AND created_at <= :end
                GROUP BY tenant_id
            """),
            {"start": start, "end": end},
        ).mappings().all()
        return {r["tenant_id"]: float(r["total"]) for r in rows}
    except Exception:
        # billing_transactions may not exist in all environments
        return {}
