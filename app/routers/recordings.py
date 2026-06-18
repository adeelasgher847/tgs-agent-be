"""
GET /api/v1/recordings/{call_id}

Returns a short-lived GCS signed URL for the call recording.

404 cases:
  - call_session not found or wrong tenant
  - recording_enabled was false for that call's number
  - no recording_gcs_path and recording_error=true (upload failed)
  - no recording_gcs_path yet (not yet uploaded)
"""

from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
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
from app.services.role_service import get_user_role_in_tenant
from app.utils.response import create_success_response

router = APIRouter()

# Roles that are blocked from accessing recordings on HIPAA-flagged flows
_HIPAA_BLOCKED_ROLES = frozenset({"readonly", "config"})


def _enforce_hipaa_recording_access(
    principal: Union[User, ApiKeyPrincipal],
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

    flow = (
        db.query(CallFlow)
        .filter(CallFlow.id == call_session.call_flow_id)
        .first()
    )
    if flow is None or not flow.hipaa_compliance:
        return

    # HIPAA flow — check caller's role
    user: User = principal
    if user.current_tenant_id is None:
        return

    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    if role and role.name in _HIPAA_BLOCKED_ROLES:
        raise HTTPException(
            status_code=403,
            detail="Access to HIPAA-protected recordings requires admin or manager role",
        )


@router.get("/{call_id}", response_model=SuccessResponse[RecordingResponse])
async def get_recording(
    call_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> SuccessResponse[RecordingResponse]:
    """
    Return a signed GCS URL for the call recording.

    URL expires in {GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS} seconds (default 3600).
    For HIPAA-flagged flows, readonly and config roles receive 403.
    """
    tenant_id = principal.current_tenant_id

    # Tenant-scoped lookup
    session = (
        db.query(CallSession)
        .filter(CallSession.id == call_id, CallSession.tenant_id == tenant_id)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    # HIPAA RBAC gate — must run before returning any data
    _enforce_hipaa_recording_access(principal, session, db)

    # Check recording was enabled for this call's number
    if not get_recording_enabled_for_call(db, session):
        raise HTTPException(status_code=404, detail="Recording not enabled for this call")

    # No path + error = upload failed
    if not session.recording_gcs_path and session.recording_error:
        raise HTTPException(status_code=404, detail="Recording upload failed for this call")

    # No path yet = not uploaded (may still be processing)
    if not session.recording_gcs_path:
        raise HTTPException(status_code=404, detail="Recording not available yet")

    # Generate signed URL
    try:
        from app.services import gcs_recording_service

        signed_url = gcs_recording_service.generate_signed_url(
            gcs_path=session.recording_gcs_path,
            expiry_seconds=settings.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
        )
    except Exception as exc:
        logger.error(
            "Failed to generate signed URL for session %s path %s: %s",
            call_id,
            session.recording_gcs_path,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Could not generate recording URL")

    # Optionally fetch file size from GCS (best-effort)
    size: int | None = None
    try:
        size = gcs_recording_service.get_object_size(session.recording_gcs_path)
    except Exception:
        pass

    return create_success_response(
        RecordingResponse(
            url=signed_url,
            duration=session.duration,
            size=size,
        ),
        "Recording URL generated",
    )
