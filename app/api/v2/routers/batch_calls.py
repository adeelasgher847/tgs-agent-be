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
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_workspace, require_tenant
from app.core.logger import logger
from app.core.request_auth import ApiKeyPrincipal
from app.core.workspace import Workspace
from app.models.user import User
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
    file: UploadFile = File(..., description="UTF-8 CSV file, max 20 MB"),
    agent_id: uuid.UUID = Form(...),
    scheduled_at: Optional[datetime] = Form(default=None),
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_batch_service_write),
) -> BatchJobOut:
    """
    Upload a CSV and create a batch outbound-call job.

    Required CSV column: phone_number.
    Additional columns are available as {variable} substitutions in the agent prompt.
    Requires admin / member / owner / config role for JWT users; any API key client.
    """
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
    )

    await _enqueue_batch_job(str(job_out.id), scheduled_at)

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
    workspace: Workspace = Depends(get_workspace),
    svc=Depends(_batch_service_write),
) -> BatchJobOut:
    """
    Cancel a batch job.

    Requires admin / member / owner / config role for JWT users; any API key client.
    Already-connected calls complete naturally; waiting records are cancelled.
    """
    return svc.cancel_batch_job(workspace.id, batch_id)


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
