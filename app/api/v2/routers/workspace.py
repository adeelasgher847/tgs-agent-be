from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.models.branding_configs import BrandingConfig
from app.models.pricing_configs import PricingConfig
from app.models.usage_record import UsageRecord
from app.schemas.workspace import (
    BrandingConfigUpsert,
    BrandingConfigOut,
    PricingConfigUpsert,
    PricingConfigOut,
    WorkspaceUsageOut,
)
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.logger import logger
from app.models.user import User
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
    user=Depends(require_admin),
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
    user=Depends(require_admin),
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
