from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin, require_billing, get_current_workspace
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
    WorkspaceUsageOut,
    MemberRoleUpdate,
    MemberRoleOut,
    SubAccountCreate,
    SubAccountUpdate,
    SubAccountOut,
    SubAccountCreateOut,
    SubAccountListOut,
)

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.logger import logger
from app.models.role import Role
from app.models.user import User, user_tenant_association
from app.services import rbac_cache_service, role_service
from app.services.account_deletion_service import delete_workspace_account
from app.services.audit_service import log_audit_event
from app.services.data_export_service import create_export_job, get_export_job

router = APIRouter(prefix="/workspace", tags=["workspace-gdpr"])

v2_router = APIRouter()

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
        effective_client_rate=effective_client_rate
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
        effective_client_rate=effective_client_rate
    )

@v2_router.get("/usage", response_model=WorkspaceUsageOut)
def get_workspace_usage(
    user=Depends(require_billing),
    db: Session = Depends(get_db),
):
    """Get the usage statistics for the current billing cycle."""
    from sqlalchemy import func
    from decimal import Decimal
    
    usage_sum = db.query(func.sum(UsageRecord.billable_minutes)).filter(
        UsageRecord.workspace_id == user.current_tenant_id,
        UsageRecord.recorded_at >= func.date_trunc('month', func.now())
    ).scalar() or Decimal("0")
    
    minutes_used_this_cycle = Decimal(str(usage_sum))
    minutes_included = None
    
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
        overage_cost=overage_cost
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
    download_url: Optional[str] = None


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
    if job.status == "ready" and job.gcs_path:
        from app.services import gcs_data_export_service
        from app.services.gcs_recording_service import generate_signed_url

        download_url = generate_signed_url(
            job.gcs_path,
            expiry_seconds=gcs_data_export_service.DATA_EXPORT_SIGNED_URL_EXPIRY_SECONDS,
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

    # Create Tenant
    new_tenant = Tenant(
        name=payload.name,
        schema_name=f"sub_{uuid.uuid4().hex[:8]}",
        parent_workspace_id=workspace.id,
        workspace_type="sub_account",
        contact_email=payload.contact_email,
        status="active"
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
    from datetime import datetime
    
    tenant_ids = [sa.id for sa in sub_accounts]
    usages = {}
    if tenant_ids:
        usage_res = db.query(
            UsageRecord.workspace_id, func.sum(UsageRecord.billable_minutes)
        ).filter(
            UsageRecord.workspace_id.in_(tenant_ids),
            UsageRecord.recorded_at >= func.date_trunc('month', func.now())
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
    sa = db.query(Tenant).filter(Tenant.id == sub_id, Tenant.parent_workspace_id == workspace.id, Tenant.deleted_at.is_(None)).first()
    if not sa:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sub-account not found")
        
    from sqlalchemy import func
    usage_sum = db.query(func.sum(UsageRecord.billable_minutes)).filter(
        UsageRecord.workspace_id == sa.id,
        UsageRecord.recorded_at >= func.date_trunc('month', func.now())
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
    
    return get_sub_account(sub_id, request, workspace, db)

@v2_router.delete("/sub-accounts/{sub_id}", status_code=204)
def delete_sub_account(
    sub_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_admin),
    workspace=Depends(get_current_workspace),
    db: Session = Depends(get_db),
):
    sa = db.query(Tenant).filter(Tenant.id == sub_id, Tenant.parent_workspace_id == workspace.id, Tenant.deleted_at.is_(None)).first()
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
