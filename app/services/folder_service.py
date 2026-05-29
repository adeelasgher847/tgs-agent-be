"""Folder service — organise call flows into tenant-scoped folders."""
from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.call_flow_repository import CallFlowRepository
from app.repositories.folder_repository import FolderRepository
from app.schemas.folder import FolderCreate, FolderOut, FolderListResponse, FolderUpdate


class FolderService:
    def _get_folder_or_404(
        self, db: Session, folder_id: uuid.UUID, tenant_id: uuid.UUID
    ):
        repo = FolderRepository(db)
        folder = repo.find_by_id(folder_id)
        if folder is None or folder.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Folder {folder_id} not found",
            )
        return folder

    def _to_out(self, folder) -> dict:
        return FolderOut.model_validate(folder).model_dump(by_alias=True, mode="json")

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create_folder(
        self, db: Session, tenant_id: uuid.UUID, body: FolderCreate
    ) -> dict:
        repo = FolderRepository(db)
        folder = repo.create({"tenant_id": tenant_id, "name": body.name})
        db.commit()
        db.refresh(folder)
        return self._to_out(folder)

    def list_folders(self, db: Session, tenant_id: uuid.UUID) -> dict:
        repo = FolderRepository(db)
        folders = repo.find_by_workspace(tenant_id)
        return {
            "data": [self._to_out(f) for f in folders],
            "total": len(folders),
        }

    def update_folder(
        self,
        db: Session,
        folder_id: uuid.UUID,
        tenant_id: uuid.UUID,
        body: FolderUpdate,
    ) -> dict:
        folder = self._get_folder_or_404(db, folder_id, tenant_id)
        repo = FolderRepository(db)
        folder = repo.update(folder, {"name": body.name})
        db.commit()
        db.refresh(folder)
        return self._to_out(folder)

    def delete_folder(
        self, db: Session, folder_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        folder = self._get_folder_or_404(db, folder_id, tenant_id)
        repo = FolderRepository(db)
        repo.soft_delete(folder)
        db.commit()

    def add_flow_to_folder(
        self,
        db: Session,
        folder_id: uuid.UUID,
        tenant_id: uuid.UUID,
        flow_id: uuid.UUID,
    ) -> dict:
        folder = self._get_folder_or_404(db, folder_id, tenant_id)

        # Verify flow belongs to tenant
        cf_repo = CallFlowRepository(db)
        flow = cf_repo.find_by_id(flow_id)
        if flow is None or flow.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Call flow {flow_id} not found in workspace",
            )

        repo = FolderRepository(db)
        # Idempotent — ignore if already linked
        existing = repo.find_folder_flow(folder_id, flow_id)
        if existing is None:
            repo.add_flow(folder_id, flow_id)
            db.commit()

        return {"folderId": str(folder_id), "flowId": str(flow_id)}


folder_service = FolderService()
