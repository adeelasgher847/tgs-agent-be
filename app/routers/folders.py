from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.folder import AddFlowToFolderRequest, FolderCreate, FolderUpdate
from app.services.folder_service import folder_service

router = APIRouter()


def _workspace_id(principal: Union[User, ApiKeyPrincipal]) -> uuid.UUID:
    return principal.current_tenant_id


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_folder(
    body: FolderCreate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return folder_service.create_folder(db, _workspace_id(principal), body)


@router.get("/")
def list_folders(
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return folder_service.list_folders(db, _workspace_id(principal))


@router.patch("/{folder_id}")
def update_folder(
    folder_id: uuid.UUID,
    body: FolderUpdate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return folder_service.update_folder(db, folder_id, _workspace_id(principal), body)


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_folder(
    folder_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    folder_service.delete_folder(db, folder_id, _workspace_id(principal))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{folder_id}/flows", status_code=status.HTTP_200_OK)
def add_flow_to_folder(
    folder_id: uuid.UUID,
    body: AddFlowToFolderRequest,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return folder_service.add_flow_to_folder(
        db, folder_id, _workspace_id(principal), body.flow_id
    )
