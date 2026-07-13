"""Workspace API key management (JWT-authenticated dashboard)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.middleware.api_key_middleware import invalidate_api_key_cache_by_hash
from app.models.user import User
from app.schemas.api_key import ApiKeyCreate, ApiKeyCreated, ApiKeyOut
from app.schemas.base import SuccessResponse
from app.services.api_key_service import (
    create_api_key,
    get_api_key_for_tenant,
    list_api_keys,
    revoke_api_key,
    to_api_key_out,
)
from app.utils.response import create_success_response

router = APIRouter()


@router.post(
    "/",
    response_model=SuccessResponse[ApiKeyCreated],
    status_code=status.HTTP_201_CREATED,
)
def create_workspace_api_key(
    body: ApiKeyCreate,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a workspace API key. The raw secret is returned exactly once."""
    record, raw_key = create_api_key(
        db,
        tenant_id=user.current_tenant_id,
        name=body.name,
    )
    payload = {**to_api_key_out(record), "raw_key": raw_key}
    return create_success_response(
        payload,
        "API key created successfully. Store the raw key now — it will not be shown again.",
        status.HTTP_201_CREATED,
    )


@router.get("/", response_model=SuccessResponse[list[ApiKeyOut]])
def list_workspace_api_keys(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List API keys for the current workspace (masked — no raw secrets)."""
    records = list_api_keys(db, tenant_id=user.current_tenant_id)
    return create_success_response(
        [to_api_key_out(r) for r in records],
        "API keys retrieved successfully",
    )


@router.delete("/{key_id}", response_model=SuccessResponse[ApiKeyOut])
async def revoke_workspace_api_key(
    key_id: uuid.UUID,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Revoke an API key (``is_active=false``) and purge its Redis cache entry."""
    record = get_api_key_for_tenant(
        db,
        key_id=key_id,
        tenant_id=user.current_tenant_id,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    record = revoke_api_key(db, record)
    await invalidate_api_key_cache_by_hash(record.key_hash, record.tenant_id)

    return create_success_response(
        to_api_key_out(record),
        "API key revoked successfully",
    )
