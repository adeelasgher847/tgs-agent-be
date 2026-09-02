from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin, require_billing, get_current_workspace, require_workspace_owner
from app.models.branding_configs import BrandingConfig
from app.models.pricing_configs import PricingConfig
import secrets
import hashlib
from app.models.api_key import Apikey
from app.models.tenant import Tenant
from app.models.usage_record import UsageRecord
from app.schemas.workspace import (
    BrandingConfigUpsert,
    BrandingConfigOut,
    PricingConfigUpsert,
    PricingConfigOut,
    SurchargeInfoOut,
    WorkspaceUsageOut,
    MemberRoleUpdate,
    MemberRoleOut,
    SubAccountCreate,
    SubAccountUpdate,
    SubAccountOut,
    SubAccountCreateOut,
    SubAccountListOut,
    WorkspaceBreakdownRowOut,
    WorkspaceUsageBreakdownOut,
    RecentActivityItemOut,
    RecentActivityOut,
    MonthlyMinutesUsageOut,
    MinutesByMonthOut,
    WalletSharingUpdate,
    AutoLinkNewWorkspacesUpdate,
    LinkedWorkspaceOut,
    LinkedWorkspacesOut,
)
from app.schemas.base import SuccessResponse
from app.services.credit_service import credit_service

import uuid
from datetime import datetime, timezone

from fastapi import Request, Response, status
from pydantic import BaseModel

from app.core.logger import logger
from app.models.role import Role
from app.models.user import User, user_tenant_association
from app.services import rbac_cache_service, role_service
from app.services.account_deletion_service import delete_workspace_account
from app.services.audit_service import log_audit_event
from app.services.data_export_service import create_export_job, get_export_job
from app.utils.response import create_success_response

router = APIRouter(prefix="/workspace", tags=["workspace-gdpr"])

v2_router = APIRouter()


def _surcharge_catalog_out() -> list[SurchargeInfoOut]:
    """Advertise the full surcharge catalog (see
    app.services.credit_service.SURCHARGE_CATALOG) via the pricing/usage
    APIs so a tenant can discover what can stack on top of the base
    per-minute rate — these endpoints are workspace-scoped, not agent-scoped,
    so this lists everything the platform *can* charge rather than what's
    active for a specific agent/call right now."""
    return [
        SurchargeInfoOut(
            key=s.key,
            label=s.label,
            rate_per_minute=s.rate_per_minute,
            applies_when=s.applies_when,
        )
        for s in credit_service.get_surcharge_catalog()
    ]

@v2_router.get("/branding", response_model=BrandingConfigOut)
def get_branding_config(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get the current branding configuration for the workspace."""
    config = db.query(BrandingConfig).filter(BrandingConfig.workspace_id == user.current_tenant_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Branding configuration not found")
    return config

@v2_router.put("/branding", response_model=BrandingConfigOut)
def upsert_branding_config(
    payload: BrandingConfigUpsert,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upsert the branding configuration for the workspace."""
    from sqlalchemy.dialects.postgresql import insert
    stmt = insert(BrandingConfig).values(
        workspace_id=user.current_tenant_id,
        logo_url=payload.logo_url,
        primary_colour=payload.primary_colour,
        display_name=payload.display_name,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=['workspace_id'],
        set_={
            'logo_url': stmt.excluded.logo_url,
            'primary_colour': stmt.excluded.primary_colour,
            'display_name': stmt.excluded.display_name,
        }
    )
    db.execute(stmt)
    db.commit()
    return db.query(BrandingConfig).filter(BrandingConfig.workspace_id == user.current_tenant_id).first()

@v2_router.get("/pricing", response_model=PricingConfigOut)
def get_pricing_config(
    user=Depends(require_billing),
    db: Session = Depends(get_db),
):
    """Get the current pricing configuration for the workspace."""
    from decimal import Decimal
    config = db.query(PricingConfig).filter(PricingConfig.workspace_id == user.current_tenant_id).first()
    
    if not config:
        per_minute_rate = Decimal("0.12")
        markup_percent = Decimal("0.00")
    else:
        per_minute_rate = config.per_minute_rate
        markup_percent = config.markup_percent
        
    effective_client_rate = Decimal(str(per_minute_rate)) * (Decimal("1") + Decimal(str(markup_percent)) / Decimal("100"))
    
    return PricingConfigOut(
        per_minute_rate=per_minute_rate,
        markup_percent=markup_percent,
        effective_client_rate=effective_client_rate,
        available_surcharges=_surcharge_catalog_out(),
    )

@v2_router.put("/pricing", response_model=PricingConfigOut)
def upsert_pricing_config(
    payload: PricingConfigUpsert,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upsert the pricing configuration for the workspace."""
    from decimal import Decimal
    from sqlalchemy.dialects.postgresql import insert
    stmt = insert(PricingConfig).values(
        workspace_id=user.current_tenant_id,
        per_minute_rate=payload.per_minute_rate,
        markup_percent=payload.markup_percent,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=['workspace_id'],
        set_={
            'per_minute_rate': stmt.excluded.per_minute_rate,
            'markup_percent': stmt.excluded.markup_percent,
        }
    )
    db.execute(stmt)
    db.commit()
    
    config = db.query(PricingConfig).filter(PricingConfig.workspace_id == user.current_tenant_id).first()
    effective_client_rate = Decimal(str(config.per_minute_rate)) * (Decimal("1") + Decimal(str(config.markup_percent)) / Decimal("100"))
    
    return PricingConfigOut(
        per_minute_rate=config.per_minute_rate,
        markup_percent=config.markup_percent,
        effective_client_rate=effective_client_rate,
        available_surcharges=_surcharge_catalog_out(),
    )

@v2_router.get("/usage", response_model=WorkspaceUsageOut)
def get_workspace_usage(
    user=Depends(require_billing),
    db: Session = Depends(get_db),
):
    """Get the usage statistics for the current billing cycle."""
    from decimal import Decimal

    from app.services.billing_service import BillingService

    minutes_included, minutes_used_this_cycle, remaining = BillingService.get_included_minutes_status(
        db, user.current_tenant_id
    )

    overage_minutes = max(Decimal("0"), minutes_used_this_cycle - minutes_included)

    config = db.query(PricingConfig).filter(PricingConfig.workspace_id == user.current_tenant_id).first()
    if config:
        effective_rate = Decimal(str(config.per_minute_rate)) * (Decimal("1") + Decimal(str(config.markup_percent)) / Decimal("100"))
    else:
        effective_rate = Decimal("0.12")
        
    overage_cost = overage_minutes * effective_rate
    
    return WorkspaceUsageOut(
        minutes_used_this_cycle=minutes_used_this_cycle,
        minutes_included=minutes_included,
        overage_minutes=overage_minutes,
        overage_cost=overage_cost,
        available_surcharges=_surcharge_catalog_out(),
    )


# ── Agency billing-dashboard reporting (owner-only) ──────────────────────────
#
# GET /workspace/usage/breakdown, /recent-activity, /minutes-by-month, and the
# .csv variants of the first and third. Scoped to {caller's current tenant} ∪
# {its direct sub_accounts} — never to all Tenant/UsageRecord rows. Gated by
# require_workspace_owner (is_creator=True on the caller's membership row for
# the *current* tenant), not require_admin: per product, a non-creator admin
# on the parent, and any user (including that sub-account's own creator)
# viewing from a sub-account's tenant context, must get 403 — "owner" isn't
# one of the canonical ranked roles, so this can't be expressed with
# require_admin/has_rank.


def _month_bounds(dt):
    """(start, end_exclusive) of the UTC calendar month containing dt."""
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def _shift_months(dt, n: int):
    """dt shifted by n calendar months (n may be negative), day pinned to 1."""
    month_index = dt.year * 12 + (dt.month - 1) + n
    year, month = divmod(month_index, 12)
    return dt.replace(year=year, month=month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


_SUPPORTED_PERIODS = ("this_month",)


def _validate_period(period: str) -> None:
    """Reject an unrecognized `period` instead of silently serving
    this_month data for it — only 'this_month' is implemented today."""
    if period not in _SUPPORTED_PERIODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported period '{period}'. Supported: {', '.join(_SUPPORTED_PERIODS)}.",
        )


def _workspace_family(db: Session, tenant_id: uuid.UUID) -> tuple[Tenant, list[Tenant]]:
    """(parent_tenant, [parent, *direct sub_accounts]) — the exact scope every
    aggregation query below must stay within.

    Family-level billing reporting is only meaningful from the *root*
    workspace's own tenant context. A sub-account has no sub_accounts of its
    own, so calling this from a sub-account's context would silently degrade
    to a harmless single-row response rather than leaking the parent
    family's data — but product explicitly wants a hard 403 here (even for
    that sub-account's own creator), so it's rejected outright rather than
    left to degrade quietly.
    """
    parent = db.query(Tenant).filter(Tenant.id == tenant_id, Tenant.deleted_at.is_(None)).first()
    if parent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    if parent.parent_workspace_id is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Billing breakdown reporting is only available from the master workspace, not a sub-account.",
        )
    subs = (
        db.query(Tenant)
        .filter(Tenant.parent_workspace_id == parent.id, Tenant.deleted_at.is_(None))
        .order_by(Tenant.created_at.asc())
        .all()
    )
    return parent, [parent, *subs]


def _sums_by_workspace(
    db: Session, tenant_ids: list, start=None, end=None, batch_size: int = 500
) -> dict:
    """{workspace_id: Decimal(sum(credits_charged))} for the given family scope.

    Batches tenant_ids to avoid unbounded IN clauses on large workspace families.
    """
    from decimal import Decimal
    from sqlalchemy import func

    if not tenant_ids:
        return {}

    result = {}
    for i in range(0, len(tenant_ids), batch_size):
        chunk = tenant_ids[i : i + batch_size]
        q = db.query(
            UsageRecord.workspace_id,
            func.coalesce(func.sum(UsageRecord.credits_charged), 0),
        ).filter(UsageRecord.workspace_id.in_(chunk))
        if start is not None:
            q = q.filter(UsageRecord.recorded_at >= start)
        if end is not None:
            q = q.filter(UsageRecord.recorded_at < end)
        for wid, total in q.group_by(UsageRecord.workspace_id).all():
            result[wid] = Decimal(str(total))
    return result


def _build_breakdown(db: Session, parent: Tenant, family: list) -> WorkspaceUsageBreakdownOut:
    from datetime import datetime, timezone
    from decimal import Decimal

    now = datetime.now(timezone.utc)
    this_month_start, this_month_end = _month_bounds(now)
    last_month_start, last_month_end = _month_bounds(_shift_months(now, -1))

    tenant_ids = [t.id for t in family]
    this_month_sums = _sums_by_workspace(db, tenant_ids, this_month_start, this_month_end)
    last_month_sums = _sums_by_workspace(db, tenant_ids, last_month_start, last_month_end)
    all_time_sums = _sums_by_workspace(db, tenant_ids)

    rows: list[WorkspaceBreakdownRowOut] = []
    total_this_month = Decimal("0")
    total_all_time = Decimal("0")
    total_last_month = Decimal("0")
    total_avg_monthly = Decimal("0")
    active_this_month = 0

    for t in family:
        this_month = this_month_sums.get(t.id, Decimal("0"))
        last_month = last_month_sums.get(t.id, Decimal("0"))
        all_time = all_time_sums.get(t.id, Decimal("0"))

        created_at = t.created_at or now
        months_since_created = max(
            1, (now.year - created_at.year) * 12 + (now.month - created_at.month) + 1
        )
        avg_monthly = all_time / months_since_created

        growth_percent = None
        if last_month != 0:
            growth_percent = float((this_month - last_month) / last_month * 100)

        if this_month > 0:
            active_this_month += 1

        total_this_month += this_month
        total_all_time += all_time
        total_last_month += last_month
        total_avg_monthly += avg_monthly

        rows.append(
            WorkspaceBreakdownRowOut(
                workspace_id=t.id,
                name=t.name,
                is_master=(t.id == parent.id),
                this_month=this_month,
                all_time=all_time,
                avg_monthly=avg_monthly,
                growth_percent=growth_percent,
            )
        )

    workspace_count = len(family)
    avg_per_workspace_this_month = (total_this_month / workspace_count) if workspace_count else Decimal("0")

    this_month_growth_percent = None
    if total_last_month != 0:
        this_month_growth_percent = float((total_this_month - total_last_month) / total_last_month * 100)

    totals = WorkspaceBreakdownRowOut(
        workspace_id=None,
        name="Total",
        is_master=False,
        this_month=total_this_month,
        all_time=total_all_time,
        avg_monthly=total_avg_monthly,
        growth_percent=this_month_growth_percent,
    )

    return WorkspaceUsageBreakdownOut(
        this_month_total=total_this_month,
        this_month_growth_percent=this_month_growth_percent,
        all_time_total=total_all_time,
        avg_per_workspace_this_month=avg_per_workspace_this_month,
        workspace_count=workspace_count,
        active_workspace_count_this_month=active_this_month,
        rows=rows,
        totals=totals,
        period="this_month",
    )


def _build_minutes_by_month(
    db: Session, family: list, batch_size: int = 500
) -> MinutesByMonthOut:
    from datetime import datetime, timezone
    from decimal import Decimal
    from sqlalchemy import func

    now = datetime.now(timezone.utc)
    tenant_ids = [t.id for t in family]

    months: list[MonthlyMinutesUsageOut] = []
    for offset in (0, -1, -2):
        ref = _shift_months(now, offset)
        start, end = _month_bounds(ref)

        total_minutes = Decimal("0")
        call_count = 0
        total_cost = Decimal("0")

        for i in range(0, len(tenant_ids), batch_size):
            chunk = tenant_ids[i : i + batch_size]
            row = (
                db.query(
                    func.coalesce(func.sum(UsageRecord.billable_minutes), 0),
                    func.count(UsageRecord.call_id),
                    func.coalesce(func.sum(UsageRecord.credits_charged), 0),
                )
                .filter(
                    UsageRecord.workspace_id.in_(chunk),
                    UsageRecord.recorded_at >= start,
                    UsageRecord.recorded_at < end,
                )
                .first()
            )
            if row:
                total_minutes += Decimal(str(row[0] or 0))
                call_count += int(row[1] or 0)
                total_cost += Decimal(str(row[2] or 0))

        months.append(
            MonthlyMinutesUsageOut(
                month=start.strftime("%Y-%m"),
                label=start.strftime("%B %Y"),
                total_minutes=total_minutes,
                call_count=call_count,
                total_cost=total_cost,
            )
        )

    # Oldest first, matching how "last 3 months" reads left-to-right in the mockup.
    months.reverse()
    return MinutesByMonthOut(months=months)


def _build_recent_activity(
    db: Session, family: list, limit: int = 10, offset: int = 0
) -> RecentActivityOut:
    tenant_ids = [t.id for t in family]

    records = (
        db.query(UsageRecord)
        .filter(UsageRecord.workspace_id.in_(tenant_ids))
        .order_by(UsageRecord.recorded_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        RecentActivityItemOut(
            date=r.recorded_at,
            type="call_usage",
            description=f"Call ID - {r.call_id}" if r.call_id else f"Call ID - {r.id}",
            amount=-r.credits_charged,
        )
        for r in records
    ]
    return RecentActivityOut(items=items, limit=limit, offset=offset)


def _csv_response(rows: list[list], header: list[str], filename: str) -> Response:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@v2_router.get("/usage/breakdown", response_model=WorkspaceUsageBreakdownOut)
def get_workspace_usage_breakdown(
    period: str = "this_month",
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    """Agency billing-dashboard breakdown: spend per workspace (parent +
    direct sub-accounts) for the current billing period. Owner-only —
    `period` is a placeholder for a future selector; only 'this_month' is
    supported today."""
    _validate_period(period)
    parent, family = _workspace_family(db, user.current_tenant_id)
    return _build_breakdown(db, parent, family)


@v2_router.get("/usage/breakdown.csv")
def get_workspace_usage_breakdown_csv(
    period: str = "this_month",
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    _validate_period(period)
    parent, family = _workspace_family(db, user.current_tenant_id)
    breakdown = _build_breakdown(db, parent, family)

    rows = [
        [
            r.name,
            "Master" if r.is_master else "Sub-Account",
            str(r.this_month),
            str(r.all_time),
            str(r.avg_monthly),
            "" if r.growth_percent is None else f"{r.growth_percent:.2f}",
        ]
        for r in breakdown.rows
    ]
    rows.append(
        [
            breakdown.totals.name,
            "",
            str(breakdown.totals.this_month),
            str(breakdown.totals.all_time),
            str(breakdown.totals.avg_monthly),
            "" if breakdown.totals.growth_percent is None else f"{breakdown.totals.growth_percent:.2f}",
        ]
    )
    return _csv_response(
        rows,
        header=["Workspace", "Type", "This Month", "All Time", "Avg Monthly", "Growth %"],
        filename="workspace-usage-breakdown.csv",
    )


@v2_router.get("/usage/recent-activity", response_model=RecentActivityOut)
def get_workspace_recent_activity(
    limit: int = Query(10, ge=1, le=100, description="Number of recent activity records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    """Spend transactions across the parent tenant + its direct
    sub-accounts with pagination support. Owner-only."""
    parent, family = _workspace_family(db, user.current_tenant_id)
    return _build_recent_activity(db, family, limit=limit, offset=offset)


@v2_router.get("/usage/minutes-by-month", response_model=MinutesByMonthOut)
def get_workspace_minutes_by_month(
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    """Minutes/calls/cost for the last 3 calendar months (current + 2 prior),
    aggregated across the parent tenant + its direct sub-accounts. Owner-only."""
    parent, family = _workspace_family(db, user.current_tenant_id)
    return _build_minutes_by_month(db, family)


@v2_router.get("/usage/minutes-by-month.csv")
def get_workspace_minutes_by_month_csv(
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    parent, family = _workspace_family(db, user.current_tenant_id)
    minutes = _build_minutes_by_month(db, family)

    rows = [
        [m.label, str(m.total_minutes), str(m.call_count), str(m.total_cost)]
        for m in minutes.months
    ]
    return _csv_response(
        rows,
        header=["Month", "Total Minutes", "Call Count", "Total Cost"],
        filename="workspace-minutes-by-month.csv",
    )


# ── Wallet sharing (agency "master wallet") ──────────────────────────────────
#
# PUT /workspace/sub-accounts/{sub_account_id}/wallet-sharing,
# PUT /workspace/auto-link-new-workspaces, GET /workspace/linked-workspaces.
# Owner-only (require_workspace_owner — see its docstring in deps/rbac.py),
# same rationale as the usage/breakdown endpoints above: this is
# cross-sub-account billing configuration that must stay invisible to a
# non-creator admin and to anyone operating from a sub-account's own tenant
# context.


@v2_router.put("/sub-accounts/{sub_account_id}/wallet-sharing", response_model=LinkedWorkspaceOut)
def update_sub_account_wallet_sharing(
    sub_account_id: uuid.UUID,
    payload: WalletSharingUpdate,
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    """Toggle whether `sub_account_id` draws call credits from the caller's
    (parent) wallet ("Using master wallet") or its own ("Using own wallet").
    `sub_account_id` must be a direct sub-account of the caller's current
    tenant — never trusted without that check."""
    parent, family = _workspace_family(db, user.current_tenant_id)
    sub = db.query(Tenant).filter(
        Tenant.id == sub_account_id,
        Tenant.parent_workspace_id == parent.id,
        Tenant.deleted_at.is_(None),
    ).first()
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sub-account not found or does not belong to this workspace.",
        )

    sub.uses_master_wallet = payload.using_master_wallet
    db.commit()
    db.refresh(sub)

    is_branded = db.query(BrandingConfig).filter(BrandingConfig.workspace_id == sub.id).first() is not None
    return LinkedWorkspaceOut(
        id=sub.id,
        name=sub.name,
        is_master=False,
        is_branded=is_branded,
        using_master_wallet=bool(sub.uses_master_wallet),
    )


@v2_router.put("/auto-link-new-workspaces", response_model=AutoLinkNewWorkspacesUpdate)
def update_auto_link_new_workspaces(
    payload: AutoLinkNewWorkspacesUpdate,
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    """Toggle the caller's own (parent/agency) `auto_link_new_workspaces` —
    when on, newly created sub-accounts default to wallet-sharing-on."""
    tenant = db.query(Tenant).filter(
        Tenant.id == user.current_tenant_id, Tenant.deleted_at.is_(None)
    ).first()
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    tenant.auto_link_new_workspaces = payload.auto_link_new_workspaces
    db.commit()
    db.refresh(tenant)

    return AutoLinkNewWorkspacesUpdate(auto_link_new_workspaces=bool(tenant.auto_link_new_workspaces))


@v2_router.get("/linked-workspaces", response_model=LinkedWorkspacesOut)
def get_linked_workspaces(
    user: User = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
):
    """Parent workspace + its direct sub-accounts, with branding/wallet-sharing
    status for the "Linked Workspaces" modal."""
    parent, family = _workspace_family(db, user.current_tenant_id)

    tenant_ids = [t.id for t in family]
    branded_ids = {
        wid
        for (wid,) in db.query(BrandingConfig.workspace_id).filter(
            BrandingConfig.workspace_id.in_(tenant_ids)
        ).all()
    }

    workspaces = [
        LinkedWorkspaceOut(
            id=t.id,
            name=t.name,
            is_master=(t.id == parent.id),
            is_branded=(t.id in branded_ids),
            using_master_wallet=bool(t.uses_master_wallet),
        )
        for t in family
    ]

    return LinkedWorkspacesOut(
        auto_link_new_workspaces=bool(parent.auto_link_new_workspaces),
        workspaces=workspaces,
    )


@v2_router.put("/members/{user_id}/role", response_model=MemberRoleOut)
def update_member_role(
    user_id: uuid.UUID,
    payload: MemberRoleUpdate,
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Assign a member's role within the current workspace. Admin only.

    A caller cannot lower their own role below their current rank (self
    targeting `user_id` == the caller's id with a role that outranks
    nothing they currently hold returns 400) — this only ever fires against
    one's own row; an admin may freely set any role on *other* members,
    including the workspace creator (whose `is_creator` override means
    they remain admin-equivalent regardless of what's stored here).
    """
    tenant_id = user.current_tenant_id

    if payload.role not in role_service.CANONICAL_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid role '{payload.role}'. Must be one of: "
                f"{', '.join(role_service.CANONICAL_ROLES)}"
            ),
        )

    if user_id == user.id:
        # Re-read fresh (uncached) — this is a security-critical precondition
        # check and must not act on a possibly-stale cached role.
        actor_role = role_service.get_membership_role_name(db, user.id, tenant_id)
        actor_rank = role_service.ROLE_RANK.get(actor_role, 0)
        new_rank = role_service.ROLE_RANK.get(payload.role, 0)
        if new_rank < actor_rank:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Cannot self-demote from '{actor_role}' to "
                    f"'{payload.role}' — assign a role of equal or higher rank."
                ),
            )

    membership = db.execute(
        user_tenant_association.select().where(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
            user_tenant_association.c.removed_at.is_(None),
        )
    ).first()
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this workspace",
        )

    role = db.query(Role).filter(Role.name == payload.role).first()
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Canonical role '{payload.role}' missing from role table",
        )

    old_role_name = membership.role_id
    db.execute(
        user_tenant_association.update()
        .where(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
        )
        .values(role_id=role.id)
    )
    db.commit()

    # Invalidate immediately so the next request re-resolves from the DB
    # instead of serving the stale role for up to the 60s TTL.
    rbac_cache_service.invalidate(user_id, tenant_id)

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="workspace.member_role_updated",
        resource_type="user_tenant_association",
        resource_id=user_id,
        old_value={"role_id": str(old_role_name) if old_role_name else None},
        new_value={"role": payload.role},
        actor_user_id=user.id,
    )

    return MemberRoleOut(
        user_id=user_id,
        workspace_id=tenant_id,
        role="owner" if membership.is_creator else payload.role
    )


@v2_router.delete("/members/{user_id}", response_model=SuccessResponse[dict])
def remove_member(
    user_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Remove a member from the current workspace. Admin only."""
    tenant_id = user.current_tenant_id
    membership = db.execute(
        user_tenant_association.select().where(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
            user_tenant_association.c.removed_at.is_(None),
        )
    ).first()
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this workspace",
        )

    db.execute(
        user_tenant_association.update().where(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
        ).values(removed_at=datetime.now(timezone.utc))
    )
    db.commit()

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="workspace.member_removed",
        resource_type="user_tenant_association",
        resource_id=user_id,
        actor_user_id=user.id,
    )

    return create_success_response(
        {"user_id": user_id, "workspace_id": tenant_id},
        "Member removed successfully",
    )



"""
v2 GDPR data subject rights router.

Endpoints (admin role required for all):
  POST   /api/v2/workspace/data-export             — trigger async export, returns 202 {job_id}
  GET    /api/v2/workspace/data-export/{job_id}     — export job status + signed download URL
  POST   /api/v2/workspace/account/delete           — irreversible hard delete + PII wipe

Account deletion is a POST action endpoint rather than DELETE-with-body:
proxies/load balancers (nginx, AWS ALB) are not guaranteed to forward a
request body on DELETE, which would silently turn the confirmation-phrase
check into a 400 (body missing) or, with a looser body-parsing path, a
bypass. POST has no such ambiguity.
"""


_DELETE_CONFIRMATION_PHRASE = "DELETE MY ACCOUNT"


# ── Schemas ───────────────────────────────────────────────────────────────────


class DataExportTriggerOut(BaseModel):
    job_id: uuid.UUID


class DataExportStatusOut(BaseModel):
    status: str
    download_url: str | None = None


class AccountDeletionRequest(BaseModel):
    confirmation: str


# ── ARQ enqueue helper ────────────────────────────────────────────────────────


async def _enqueue_data_export_job(export_job_id: str) -> None:
    """
    Push a run_data_export_job task into ARQ. Falls back to a temporary
    per-call pool if the shared pool was not initialised, same as the
    batch-calls enqueue helper. Never raises — caller can still return 202
    and the job stays 'processing' until an operator notices and re-runs it.
    """
    try:
        from app.utils.arq_pool import get_arq_pool

        pool = get_arq_pool()
        _owns_pool = False

        if pool is None:
            import arq  # type: ignore
            from app.core.config import settings as cfg

            redis_settings = arq.connections.RedisSettings.from_dsn(cfg.REDIS_URL)
            pool = await arq.create_pool(redis_settings)
            _owns_pool = True

        try:
            await pool.enqueue_job("run_data_export_job", export_job_id)
            logger.info("DataExportJob %s enqueued in ARQ", export_job_id)
        finally:
            if _owns_pool:
                await pool.aclose()

    except Exception as exc:
        logger.warning(
            "ARQ enqueue failed for data export %s: %s",
            export_job_id,
            exc,
        )


# ── POST /workspace/data-export ─────────────────────────────────────────────


@router.post(
    "/data-export",
    response_model=DataExportTriggerOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_data_export(
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> DataExportTriggerOut:
    """Kick off an async export of all workspace data. Admin role required."""
    tenant_id = user.current_tenant_id
    job = create_export_job(db, tenant_id, user.id)

    await _enqueue_data_export_job(str(job.id))

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="workspace.data_export_requested",
        resource_type="data_export_job",
        resource_id=job.id,
        actor_user_id=user.id,
    )

    return DataExportTriggerOut(job_id=job.id)


# ── GET /workspace/data-export/{job_id} ──────────────────────────────────────


@router.get("/data-export/{job_id}", response_model=DataExportStatusOut)
def get_data_export_status(
    job_id: uuid.UUID,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> DataExportStatusOut:
    """Poll export job status. Returns a fresh 24h signed URL once ready."""
    tenant_id = user.current_tenant_id
    job = get_export_job(db, tenant_id, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found")

    download_url = None
    if job.status == "ready" and job.s3_path:
        from app.services import s3_data_export_service
        from app.services.s3_recording_service import generate_signed_url

        download_url = generate_signed_url(
            job.s3_path,
            expiry_seconds=s3_data_export_service.DATA_EXPORT_SIGNED_URL_EXPIRY_SECONDS,
        )

    return DataExportStatusOut(status=job.status, download_url=download_url)


# ── POST /workspace/account/delete ────────────────────────────────────────────


@router.post(
    "/account/delete",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_account(
    body: AccountDeletionRequest,
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Response:
    """
    Irreversibly erase the workspace: wipes PII, deletes KB embeddings and
    GCS recordings, anonymizes audit log actor fields, and soft-deletes the
    workspace. Requires an exact, case-sensitive confirmation phrase.
    """
    if body.confirmation != _DELETE_CONFIRMATION_PHRASE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"confirmation must exactly match '{_DELETE_CONFIRMATION_PHRASE}'",
        )

    tenant_id = user.current_tenant_id

    # Logged before the wipe so the action itself is captured in the audit
    # trail; the actor fields on this very row are anonymized along with
    # every other auditlog row for this workspace inside delete_workspace_account.
    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="workspace.account_deleted",
        resource_type="workspace",
        resource_id=tenant_id,
        actor_user_id=user.id,
    )

    delete_workspace_account(db, tenant_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ── Sub-Accounts CRUD ─────────────────────────────────────────────────────────


@v2_router.post("/sub-accounts", response_model=SubAccountCreateOut, status_code=201)
def create_sub_account(
    payload: SubAccountCreate,
    request: Request,
    user: User = Depends(require_admin),
    workspace=Depends(get_current_workspace),
    db: Session = Depends(get_db),
):
    if workspace.workspace_type != "agency":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only agency workspaces can create sub-accounts.")

    from app.services.billing_service import BillingService
    from app.models.plan import Plan

    subscription = BillingService.get_workspace_subscription(db, workspace.id)
    if subscription is not None:
        plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
        if plan is not None and plan.max_subaccounts is not None:
            existing_count = db.query(Tenant).filter(
                Tenant.parent_workspace_id == workspace.id,
                Tenant.deleted_at.is_(None),
            ).count()
            if existing_count >= plan.max_subaccounts:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Sub-account limit reached for your plan "
                        f"({plan.max_subaccounts} max). Upgrade your plan to add more."
                    ),
                )

    # Create Tenant. When the parent has auto_link_new_workspaces enabled,
    # the new sub-account defaults to wallet-sharing-on (uses_master_wallet);
    # otherwise it keeps the column default (False, own-wallet).
    new_tenant = Tenant(
        name=payload.name,
        schema_name=f"sub_{uuid.uuid4().hex[:8]}",
        parent_workspace_id=workspace.id,
        workspace_type="sub_account",
        contact_email=payload.contact_email,
        status="active",
        uses_master_wallet=bool(workspace.auto_link_new_workspaces),
    )
    db.add(new_tenant)
    db.flush()

    # Generate API key
    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    api_key_prefix = raw_key[:8]

    new_api_key = Apikey(
        tenant_id=new_tenant.id,
        name="Sub-Account Default Key",
        key_prefix=api_key_prefix,
        key_hash=key_hash,
        is_active=True
    )
    db.add(new_api_key)
    db.commit()
    db.refresh(new_tenant)

    # We return usage as 0 for new
    return SubAccountCreateOut(
        id=new_tenant.id,
        name=new_tenant.name,
        contact_email=new_tenant.contact_email,
        status=new_tenant.status,
        api_key_prefix=api_key_prefix,
        usage_this_cycle_minutes=0.0,
        api_key=raw_key
    )

@v2_router.get("/sub-accounts", response_model=SubAccountListOut)
def list_sub_accounts(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(require_admin),
    workspace=Depends(get_current_workspace),
    db: Session = Depends(get_db),
):
    if workspace.workspace_type != "agency":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only agency workspaces have sub-accounts.")

    query = db.query(Tenant).filter(Tenant.parent_workspace_id == workspace.id, Tenant.deleted_at.is_(None))
    total = query.count()
    sub_accounts = query.order_by(Tenant.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    # Fetch usage for all
    from sqlalchemy import func
    from datetime import datetime, timezone
    
    tenant_ids = [sa.id for sa in sub_accounts]
    usages = {}
    if tenant_ids:
        first_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        usage_res = db.query(
            UsageRecord.workspace_id, func.sum(UsageRecord.billable_minutes)
        ).filter(
            UsageRecord.workspace_id.in_(tenant_ids),
            UsageRecord.recorded_at >= first_of_month
        ).group_by(UsageRecord.workspace_id).all()
        usages = {wid: float(mins) for wid, mins in usage_res if mins is not None}

    # Fetch api key prefix for each
    key_res = db.query(Apikey.tenant_id, Apikey.key_prefix).filter(
        Apikey.tenant_id.in_(tenant_ids), Apikey.is_active.is_(True)
    ).all()
    prefixes = {t_id: prefix for t_id, prefix in key_res}

    data = []
    for sa in sub_accounts:
        data.append(SubAccountOut(
            id=sa.id,
            name=sa.name,
            contact_email=sa.contact_email,
            status=sa.status,
            api_key_prefix=prefixes.get(sa.id),
            usage_this_cycle_minutes=usages.get(sa.id, 0.0)
        ))

    return SubAccountListOut(
        data=data,
        total=total,
        page=page,
        page_size=page_size
    )

@v2_router.get("/sub-accounts/{sub_id}", response_model=SubAccountOut)
def get_sub_account(
    sub_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_admin),
    workspace=Depends(get_current_workspace),
    db: Session = Depends(get_db),
):
    return _fetch_sub_account_out(sub_id, workspace, db)


def _fetch_sub_account_out(sub_id: uuid.UUID, workspace, db: Session) -> SubAccountOut:
    """Shared logic used by get_sub_account and update_sub_account."""
    sa = db.query(Tenant).filter(Tenant.id == sub_id, Tenant.parent_workspace_id == workspace.id, Tenant.deleted_at.is_(None)).first()
    if not sa:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sub-account not found")
        
    from sqlalchemy import func
    from datetime import datetime, timezone
    first_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage_sum = db.query(func.sum(UsageRecord.billable_minutes)).filter(
        UsageRecord.workspace_id == sa.id,
        UsageRecord.recorded_at >= first_of_month
    ).scalar() or 0.0

    key = db.query(Apikey).filter(Apikey.tenant_id == sa.id, Apikey.is_active.is_(True)).first()
    
    return SubAccountOut(
        id=sa.id,
        name=sa.name,
        contact_email=sa.contact_email,
        status=sa.status,
        api_key_prefix=key.key_prefix if key else None,
        usage_this_cycle_minutes=float(usage_sum)
    )

@v2_router.put("/sub-accounts/{sub_id}", response_model=SubAccountOut)
def update_sub_account(
    sub_id: uuid.UUID,
    payload: SubAccountUpdate,
    request: Request,
    user: User = Depends(require_admin),
    workspace=Depends(get_current_workspace),
    db: Session = Depends(get_db),
):
    sa = db.query(Tenant).filter(Tenant.id == sub_id, Tenant.parent_workspace_id == workspace.id, Tenant.deleted_at.is_(None)).first()
    if not sa:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sub-account not found")
        
    if payload.name is not None:
        sa.name = payload.name
    if payload.contact_email is not None:
        sa.contact_email = payload.contact_email
        
    db.commit()
    db.refresh(sa)
    
    return _fetch_sub_account_out(sub_id, workspace, db)

@v2_router.delete("/sub-accounts/{sub_id}", status_code=204)
def delete_sub_account(
    sub_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_admin),
    workspace=Depends(get_current_workspace),
    db: Session = Depends(get_db),
):
    sa = db.query(Tenant).with_for_update().filter(Tenant.id == sub_id, Tenant.parent_workspace_id == workspace.id, Tenant.deleted_at.is_(None)).first()
    if not sa:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sub-account not found")
        
    from app.models.call_session import CallSession
    active_calls = db.query(CallSession).filter(CallSession.tenant_id == sa.id, CallSession.status == "in-progress").count()
    if active_calls > 0:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Cannot delete sub-account with active calls.")
        
    from sqlalchemy.sql import func
    sa.deleted_at = func.now()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@v2_router.post("/members/{user_id}/role", response_model=MemberRoleOut)
def create_member_role(
    user_id: uuid.UUID,
    payload: MemberRoleUpdate,
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return update_member_role(user_id, payload, request, user, db)
