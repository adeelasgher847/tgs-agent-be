from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_config, require_readonly
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.transfer_route import (
    TransferRouteCreate,
    TransferRouteListResponse,
    TransferRouteOut,
    TransferRouteUpdate,
)
from app.services.transfer_route_service import transfer_route_service
from app.utils.response import create_success_response

router = APIRouter()


@router.get("/", response_model=SuccessResponse[TransferRouteListResponse])
def list_transfer_routes(
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
):
    rows = transfer_route_service.list_for_tenant(db, user.current_tenant_id)
    out = [TransferRouteOut.model_validate(r) for r in rows]
    return create_success_response(
        TransferRouteListResponse(data=out, total=len(out)),
        "Transfer routes retrieved successfully",
    )


@router.post("/", response_model=SuccessResponse[TransferRouteOut], status_code=status.HTTP_201_CREATED)
def create_transfer_route(
    body: TransferRouteCreate,
    user: User = Depends(require_config),
    db: Session = Depends(get_db),
):
    row = transfer_route_service.create(db, user.current_tenant_id, body)
    return create_success_response(
        TransferRouteOut.model_validate(row),
        "Transfer route created successfully",
        status.HTTP_201_CREATED,
    )


@router.get("/{route_id}", response_model=SuccessResponse[TransferRouteOut])
def get_transfer_route(
    route_id: uuid.UUID,
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
):
    row = transfer_route_service.get(db, route_id, user.current_tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transfer route not found")
    return create_success_response(TransferRouteOut.model_validate(row), "Transfer route retrieved successfully")


@router.put("/{route_id}", response_model=SuccessResponse[TransferRouteOut])
def update_transfer_route(
    route_id: uuid.UUID,
    body: TransferRouteUpdate,
    user: User = Depends(require_config),
    db: Session = Depends(get_db),
):
    row = transfer_route_service.update(db, route_id, user.current_tenant_id, body)
    return create_success_response(TransferRouteOut.model_validate(row), "Transfer route updated successfully")


@router.delete("/{route_id}", response_model=SuccessResponse[dict])
def delete_transfer_route(
    route_id: uuid.UUID,
    user: User = Depends(require_config),
    db: Session = Depends(get_db),
):
    transfer_route_service.soft_delete(db, route_id, user.current_tenant_id)
    return create_success_response(
        {"id": str(route_id), "soft_deleted": True},
        "Transfer route archived successfully",
    )
