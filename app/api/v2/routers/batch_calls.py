"""
v2 Batch Calls router.

Auth: any authenticated tenant principal is accepted —
  - API key clients  (x-api-key + x-workspace-id  OR  header-resolved by middleware)
  - JWT dashboard users  with admin / member / owner / config roles

Write endpoints (POST, DELETE) additionally reject JWT users with readonly role.
Read endpoints (GET) allow all authenticated roles including readonly.

POST   /batch-calls                      — upload CSV + create job
GET    /batch-calls                      — paginated list
GET    /batch-calls/{batch_id}           — detail
GET    /batch-calls/{batch_id}/progress  — live counts
GET    /batch-calls/{batch_id}/calls     — paginated records
DELETE /batch-calls/{batch_id}          — cancel
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_workspace, require_tenant
from app.core.error_responses import build_api_error_payload
from app.core.logger import logger
from app.core.request_auth import ApiKeyPrincipal
from app.core.workspace import Workspace
from app.models.user import User
from app.services.audit_service import log_audit_event
from app.services.batch_call_service import AllNumbersFlaggedError
from app.schemas.batch_call import (
    BatchJobOut,
    BatchJobProgress,
    PaginatedBatchCallRecords,
    PaginatedBatchJobs,
)

router = APIRouter(prefix="/batch-calls", tags=["batch-calls"])


# ── Service factories ─────────────────────────────────────────────────────────

def _batch_service(
    workspace: Workspace = Depends(get_workspace),
    _principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Yield a BatchCallService for any authenticated tenant principal (read + write)."""
    from app.services.batch_call_service import BatchCallService

    yield BatchCallService(db)


def _batch_service_write(
    request: Request,
    workspace: Workspace = Depends(get_workspace),
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Like _batch_service but blocks JWT users with readonly role on write methods.

    API key principals always pass — they carry no per-user role concept.
    """
    from app.services.batch_call_service import BatchCallService
    from app.services.role_service import get_user_role_in_tenant

    if isinstance(principal, User) and principal.current_tenant_id:
        role = get_user_role_in_tenant(db, principal.id, principal.current_tenant_id)
        if role and role.name == "readonly":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Read-only access cannot modify resources",
            )

    yield BatchCallService(db)


# ── POST /batch-calls ─────────────────────────────────────────────────────────

@router.post("", response_model=BatchJobOut, status_code=status.HTTP_201_CREATED)
async def create_batch_job(
    request: Request,
    file: UploadFile = File(..., description="UTF-8 CSV file, max 20 MB"),
    agent_id: uuid.UUID = Form(...),
    scheduled_at: Optional[datetime] = Form(default=None),
    voicemail_action: str = Form(default="skip", description="skip | leave_message | continue"),
    voicemail_message: Optional[str] = Form(default=None, description="Max 500 chars"),
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    svc=Depends(_batch_service_write),
) -> BatchJobOut:
    """
    Upload a CSV and create a batch outbound-call job.

    Required CSV column: phone_number.
    Additional columns are available as {variable} substitutions in the agent prompt.
    Requires admin / member / owner / config role for JWT users; any API key client.
    """
    from app.schemas.batch_call import VOICEMAIL_ACTIONS

    if voicemail_action not in VOICEMAIL_ACTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"voicemail_action must be one of {VOICEMAIL_ACTIONS}",
        )
    if voicemail_message is not None and len(voicemail_message) > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="voicemail_message must be at most 500 characters",
        )

    if file.content_type and "csv" not in file.content_type and "text" not in file.content_type:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must be a CSV (text/csv or text/plain)",
        )

    from app.services.batch_call_service import MAX_CSV_BYTES

    if file.size is not None and file.size > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file exceeds maximum size of 20 MB",
        )

    raw = await file.read()

    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file exceeds maximum size of 20 MB",
        )

    job_out = svc.create_batch_job(
        workspace_id=workspace.id,
        agent_id=agent_id,
        csv_bytes=raw,
        scheduled_at=scheduled_at,
        voicemail_action=voicemail_action,
        voicemail_message=voicemail_message,
    )

    # Rotate the outbound caller ID before dispatch if the agent's bound number
    # is spam-flagged; 422s out if no clean same-country number is available.
    try:
        rotation = await svc.rotate_number_if_flagged(
            workspace_id=workspace.id,
            agent_id=agent_id,
            batch_job_id=job_out.id,
        )
    except AllNumbersFlaggedError:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=build_api_error_payload(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "All outbound numbers are spam-flagged. Please add a new number.",
                error_code="all_numbers_flagged",
                request_id=getattr(request.state, "request_id", ""),
            ),
        )

    if rotation is not None:
        old_number, new_number = rotation
        log_audit_event(
            db,
            request=request,
            tenant_id=workspace.id,
            action="batch.number_rotated",
            resource_type="batch_job",
            resource_id=job_out.id,
            old_value={"from_number": old_number},
            new_value={"from_number": new_number, "reason": "spam_flagged"},
        )

    await _enqueue_batch_job(str(job_out.id), scheduled_at)

    log_audit_event(
        db,
        request=request,
        tenant_id=workspace.id,
        action="batch_job.created",
        resource_type="batch_job",
        resource_id=job_out.id,
        new_value={"agent_id": str(agent_id), "scheduled_at": str(scheduled_at)},
    )

    return job_out


# ── GET /batch-calls ──────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedBatchJobs)
def list_batch_jobs(
    page: int = 1,
    page_size: int = 20,
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_batch_service),
) -> PaginatedBatchJobs:
    return svc.list_batch_jobs(workspace.id, page=page, page_size=page_size)


# ── GET /batch-calls/{batch_id} ───────────────────────────────────────────────

@router.get("/{batch_id}", response_model=BatchJobOut)
def get_batch_job(
    batch_id: uuid.UUID,
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_batch_service),
) -> BatchJobOut:
    job = svc.get_batch_job(workspace.id, batch_id)
    return BatchJobOut.model_validate(job)


# ── GET /batch-calls/{batch_id}/progress ─────────────────────────────────────

@router.get("/{batch_id}/progress", response_model=BatchJobProgress)
def get_batch_job_progress(
    batch_id: uuid.UUID,
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_batch_service),
) -> BatchJobProgress:
    return svc.get_batch_job_progress(workspace.id, batch_id)


# ── GET /batch-calls/{batch_id}/calls ─────────────────────────────────────────

@router.get("/{batch_id}/calls", response_model=PaginatedBatchCallRecords)
def list_batch_call_records(
    batch_id: uuid.UUID,
    page: int = 1,
    page_size: int = 50,
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_batch_service),
) -> PaginatedBatchCallRecords:
    return svc.list_batch_call_records(workspace.id, batch_id, page=page, page_size=page_size)


# ── DELETE /batch-calls/{batch_id} ───────────────────────────────────────────

@router.delete("/{batch_id}", response_model=BatchJobOut)
def cancel_batch_job(
    batch_id: uuid.UUID,
    request: Request,
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    svc=Depends(_batch_service_write),
) -> BatchJobOut:
    """
    Cancel a batch job.

    Requires admin / member / owner / config role for JWT users; any API key client.
    Already-connected calls complete naturally; waiting records are cancelled.
    """
    result = svc.cancel_batch_job(workspace.id, batch_id)
    log_audit_event(
        db,
        request=request,
        tenant_id=workspace.id,
        action="batch_job.cancelled",
        resource_type="batch_job",
        resource_id=batch_id,
    )
    return result


# ── ARQ enqueue helper ────────────────────────────────────────────────────────

async def _enqueue_batch_job(
    batch_job_id: str,
    scheduled_at: Optional[datetime],
) -> None:
    """
    Push a process_batch_job task into ARQ.

    Uses the application-level shared pool (app/utils/arq_pool.py) so no new
    connection is established per request.  Falls back to a temporary per-call
    pool if the shared pool was not initialised (e.g. Redis was down at startup).
    Falls back silently on any error — the 60-s poll cron self-heals.
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

        kwargs: dict = {}
        if scheduled_at is not None:
            from datetime import timezone as _tz

            now = datetime.now(_tz.utc)
            if scheduled_at.tzinfo is None:
                scheduled_at = scheduled_at.replace(tzinfo=_tz.utc)
            if scheduled_at > now:
                kwargs["_defer_until"] = scheduled_at

        try:
            await pool.enqueue_job("process_batch_job", batch_job_id, **kwargs)
            logger.info("BatchJob %s enqueued in ARQ", batch_job_id)
        finally:
            if _owns_pool:
                await pool.aclose()

    except Exception as exc:
        logger.warning(
            "ARQ enqueue failed for batch %s: %s — worker will self-heal on next poll",
            batch_job_id,
            exc,
        )
