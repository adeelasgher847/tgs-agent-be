"""
v2 HIPAA compliance router.

Endpoints:
  PUT  /api/v2/flows/{flow_id}/settings    — toggle hipaa_compliance (admin only)
  GET  /api/v2/workspace/hipaa-status      — HIPAA status summary for workspace
  PUT  /api/v2/workspace/kms-key           — register / update KMS key for CMEK
"""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.tenant import Tenant
from app.models.user import User
from app.services import gcs_recording_service
from app.services.audit_service import log_audit_event
from app.utils.response import create_success_response

flows_router = APIRouter(prefix="/flows", tags=["hipaa"])
workspace_router = APIRouter(prefix="/workspace", tags=["hipaa"])

_KMS_KEY_PREFIX = "projects/"


# ── Schemas ───────────────────────────────────────────────────────────────────


class HipaaSettingsUpdate(BaseModel):
    hipaa_compliance: bool


class KmsKeyUpdate(BaseModel):
    kms_key_name: str

    @field_validator("kms_key_name")
    @classmethod
    def validate_kms_key_format(cls, v: str) -> str:
        if not v.startswith(_KMS_KEY_PREFIX):
            raise ValueError(
                "kms_key_name must be a full Cloud KMS resource name "
                "(e.g. projects/my-project/locations/us-central1/keyRings/..."
                "/cryptoKeys/my-key)"
            )
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_flow_or_404(
    db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID
) -> CallFlow:
    stmt = select(CallFlow).where(
        CallFlow.id == flow_id,
        CallFlow.tenant_id == tenant_id,
        CallFlow.is_deleted.is_(False),
    )
    flow = db.execute(stmt).scalar_one_or_none()
    if flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call flow {flow_id} not found",
        )
    return flow


def _get_tenant_or_404(db: Session, tenant_id: uuid.UUID) -> Tenant:
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    tenant = db.execute(stmt).scalar_one_or_none()
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )
    return tenant


def _get_hipaa_flow_ids(
    db: Session, tenant_id: uuid.UUID
) -> list[uuid.UUID]:
    stmt = select(CallFlow.id).where(
        CallFlow.tenant_id == tenant_id,
        CallFlow.hipaa_compliance.is_(True),
        CallFlow.is_deleted.is_(False),
    )
    return [row.id for row in db.execute(stmt).all()]


def _validate_kms_key_blocking(kms_key_name: str) -> None:
    """Blocking KMS validation — run inside executor via to_thread."""
    from google.cloud import kms  # type: ignore

    client = kms.KeyManagementServiceClient()
    client.get_crypto_key(request={"name": kms_key_name})


async def _validate_kms_key(kms_key_name: str) -> None:
    """
    Verify the KMS key exists and the service account has encrypt/decrypt rights.
    Non-blocking: wraps the blocking gRPC call in asyncio.to_thread.
    Raises HTTPException 400 if validation fails.
    """
    try:
        await asyncio.to_thread(_validate_kms_key_blocking, kms_key_name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"KMS key validation failed: {exc}",
        ) from exc




# ── Routes ────────────────────────────────────────────────────────────────────


@flows_router.put("/{flow_id}/settings")
async def update_flow_hipaa_settings(
    flow_id: uuid.UUID,
    body: HipaaSettingsUpdate,
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Toggle HIPAA compliance on a call flow.  Admin role required.

    When enabled:
      - all call session log entries are DLP-redacted before Cloud Logging
      - GCS recordings use the workspace kms_key_name (CMEK)
      - recording access is restricted to admin and manager roles
    """
    tenant_id = user.current_tenant_id
    flow = _get_flow_or_404(db, flow_id, tenant_id)
    tenant = _get_tenant_or_404(db, tenant_id)

    old_value = bool(flow.hipaa_compliance)
    new_value = body.hipaa_compliance

    # HIPAA cannot be enabled without a signed BAA on file.
    if new_value and not tenant.baa_on_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "A signed Business Associate Agreement (BAA) must be on file "
                "before enabling HIPAA compliance.  Set tenant.baa_on_file=true "
                "after the BAA has been executed."
            ),
        )

    if old_value != new_value:
        flow.hipaa_compliance = new_value
        db.commit()
        db.refresh(flow)

        log_audit_event(
            db,
            request=request,
            tenant_id=tenant_id,
            action="hipaa_flag.updated",
            resource_type="call_flow",
            resource_id=flow_id,
            old_value={"hipaa_compliance": old_value},
            new_value={"hipaa_compliance": new_value},
            actor_user_id=user.id,
        )

    return create_success_response(
        {
            "flow_id": str(flow_id),
            "hipaa_compliance": flow.hipaa_compliance,
        },
        "HIPAA settings updated",
    )


@workspace_router.get("/hipaa-status")
async def get_hipaa_status(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return workspace-level HIPAA status summary."""
    tenant_id = user.current_tenant_id
    tenant = _get_tenant_or_404(db, tenant_id)
    hipaa_flow_ids = _get_hipaa_flow_ids(db, tenant_id)

    return create_success_response(
        {
            "hipaa_enabled_flows": [str(fid) for fid in hipaa_flow_ids],
            "kms_key_configured": bool(tenant.kms_key_name),
            "baa_on_file": bool(tenant.baa_on_file),
        },
        "HIPAA status retrieved",
    )


@workspace_router.put("/kms-key")
async def update_kms_key(
    body: KmsKeyUpdate,
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Register or update the workspace Cloud KMS key for CMEK recording encryption.

    Validates that the key exists and the service account has encrypt/decrypt
    permissions before persisting.  Admin role required.
    """
    tenant_id = user.current_tenant_id
    tenant = _get_tenant_or_404(db, tenant_id)

    await _validate_kms_key(body.kms_key_name)

    tenant.kms_key_name = body.kms_key_name
    db.commit()
    db.refresh(tenant)

    # Apply the KMS key as the bucket-level CMEK default so that recordings
    # written directly by LiveKit (which bypass upload_recording()) are also
    # encrypted automatically, with no application-level changes required.
    try:
        gcs_recording_service.set_bucket_default_kms_key(body.kms_key_name)
    except Exception as exc:
        logger.warning(
            "KMS key persisted but bucket default-key patch failed "
            "(tenant=%s kms_key=%s): %s",
            tenant_id,
            body.kms_key_name,
            exc,
        )

    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="workspace.kms_key_updated",
        resource_type="workspace",
        resource_id=tenant_id,
        new_value={"kms_key_name": body.kms_key_name},
        actor_user_id=user.id,
    )

    return create_success_response(
        {
            "kms_key_name": tenant.kms_key_name,
        },
        "KMS key updated",
    )