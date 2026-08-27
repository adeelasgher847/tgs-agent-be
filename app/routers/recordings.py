"""
GET /api/v1/recordings/{call_id}

Returns a short-lived S3 signed URL for the call recording.

404 cases:
  - call_session not found or wrong tenant
  - recording_enabled was false for that call's number
  - no recording_s3_path and recording_error=true (upload failed)
  - no recording_s3_path yet (not yet uploaded)
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.config import settings
from app.core.logger import logger
from app.core.request_auth import ApiKeyPrincipal
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.recording import RecordingResponse
from app.services.recording_config_service import get_recording_enabled_for_call
from app.services import role_service
from app.utils.response import create_success_response

router = APIRouter()


def _enforce_hipaa_recording_access(
    principal: User | ApiKeyPrincipal,
    call_session: CallSession,
    db: Session,
) -> None:
    """Raise 403 if the caller is a low-privilege JWT user accessing a HIPAA-flow recording."""
    # API key callers (machine-to-machine) are not role-restricted
    if isinstance(principal, ApiKeyPrincipal):
        return

    # Resolve the flow's HIPAA flag
    if call_session.call_flow_id is None:
        return

    flow = db.execute(
        select(CallFlow).where(
            CallFlow.id == call_session.call_flow_id,
            CallFlow.tenant_id == call_session.tenant_id,
        )
    ).scalar_one_or_none()
    if flow is None or not flow.hipaa_compliance:
        return

    # HIPAA flow — check caller's role
    user: User = principal
    if user.current_tenant_id is None:
        return

    role_name = role_service.get_membership_role_name(db, user.id, user.current_tenant_id)
    if not role_service.has_rank(role_name, role_service.MANAGER):
        raise HTTPException(
            status_code=403,
            detail="Access to HIPAA-protected recordings requires admin or manager role",
        )


@router.get("/{call_id}", response_model=SuccessResponse[RecordingResponse])
async def get_recording(
    call_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> SuccessResponse[RecordingResponse]:
    """
    Return a signed S3 URL for the call recording.

    URL expires in {GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS} seconds (default 3600).
    For HIPAA-flagged flows, readonly and config roles receive 403.
    """
    tenant_id = principal.current_tenant_id

    # Tenant-scoped lookup
    session = db.execute(
        select(CallSession).where(
            CallSession.id == call_id,
            CallSession.tenant_id == tenant_id,
        )
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    # HIPAA RBAC gate — must run before returning any data
    _enforce_hipaa_recording_access(principal, session, db)

    # Check if a recorded file exists on S3 for this session
    if not session.recording_s3_path:
        if session.recording_error:
            raise HTTPException(
                status_code=404, detail="Recording upload failed for this call"
            )
        if not get_recording_enabled_for_call(db, session):
            raise HTTPException(
                status_code=404, detail="Recording not enabled for this call"
            )
        raise HTTPException(
            status_code=404, detail="Recording not available yet"
        )

    # Generate signed URL
    try:
        from app.services import s3_recording_service

        signed_url = s3_recording_service.generate_signed_url(
            s3_path=session.recording_s3_path,
            expiry_seconds=settings.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
        )
    except Exception as exc:
        logger.error(
            "Failed to generate signed URL for session %s path %s: %s",
            call_id,
            session.recording_s3_path,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Could not generate recording URL")

    # Optionally fetch file size from S3 (best-effort)
    size: int | None = None
    try:
        size = s3_recording_service.get_object_size(session.recording_s3_path)
    except Exception as exc:
        logger.debug("Failed to fetch S3 object size for %s: %s", session.recording_s3_path, exc)

    return create_success_response(
        RecordingResponse(
            url=signed_url,
            duration=session.duration,
            size=size,
        ),
        "Recording URL generated",
    )


@router.get("/public/{call_id}", response_model=SuccessResponse[RecordingResponse])
@router.get("/{call_id}/public", response_model=SuccessResponse[RecordingResponse])
async def get_public_recording(
    call_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> SuccessResponse[RecordingResponse]:
    """
    Return a signed S3 URL for a call recording without authentication,
    provided that the call flow has public_recording_enabled=True.

    Security note: To prevent probing/enumerating call session existence across tenants,
    any unknown call_id, session without flow, or non-public flow uniformly returns 403 Forbidden.
    The tenant boundary is derived directly from the session's parent call flow.
    """
    session = db.execute(
        select(CallSession).where(CallSession.id == call_id)
    ).scalar_one_or_none()

    # Collapse non-existent session and unattached flow into uniform 403 to avoid information leakage
    if session is None or not session.call_flow_id:
        raise HTTPException(
            status_code=403,
            detail="Public recording access is not enabled for this call",
        )

    flow = db.execute(
        select(CallFlow).where(
            CallFlow.id == session.call_flow_id,
            CallFlow.tenant_id == session.tenant_id,
            CallFlow.is_deleted.is_(False),
        )
    ).scalar_one_or_none()
    if not flow or not flow.public_recording_enabled:
        raise HTTPException(
            status_code=403,
            detail="Public recording access is not enabled for this call",
        )

    if flow.hipaa_compliance:
        raise HTTPException(
            status_code=403,
            detail="Public recording access is not permitted for HIPAA-compliant flows",
        )

    # Check if a recorded file exists on S3 for this session
    if not session.recording_s3_path:
        if session.recording_error:
            raise HTTPException(
                status_code=404, detail="Recording upload failed for this call"
            )
        if not get_recording_enabled_for_call(db, session):
            raise HTTPException(
                status_code=404, detail="Recording not enabled for this call"
            )
        raise HTTPException(
            status_code=404, detail="Recording not available yet"
        )

    # Generate signed URL
    try:
        from app.services import s3_recording_service

        signed_url = s3_recording_service.generate_signed_url(
            s3_path=session.recording_s3_path,
            expiry_seconds=settings.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
        )
    except Exception as exc:
        logger.error(
            "Failed to generate public signed URL for session %s path %s: %s",
            call_id,
            session.recording_s3_path,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Could not generate recording URL")

    size: int | None = None
    try:
        size = s3_recording_service.get_object_size(session.recording_s3_path)
    except Exception as exc:
        logger.debug(
            "Failed to fetch S3 object size for %s: %s",
            session.recording_s3_path,
            exc,
        )

    return create_success_response(
        RecordingResponse(
            url=signed_url,
            duration=session.duration,
            size=size,
        ),
        "Public recording URL generated",
    )

