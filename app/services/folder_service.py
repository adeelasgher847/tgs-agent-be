"""Folder service — organise call flows into tenant-scoped folders."""

from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.call_flow_repository import CallFlowRepository
from app.repositories.folder_repository import FolderRepository
from app.schemas.folder import FolderCreate, FolderOut, FolderUpdate
from app.services.call_flow_service import call_flow_service


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
        self._get_folder_or_404(db, folder_id, tenant_id)

        cf_repo = CallFlowRepository(db)
        flow = cf_repo.find_by_id(flow_id, tenant_id=tenant_id)
        if flow is None:
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

    def move_flow_to_folder(
        self,
        db: Session,
        source_folder_id: uuid.UUID,
        target_folder_id: uuid.UUID,
        tenant_id: uuid.UUID,
        flow_id: uuid.UUID,
    ) -> dict:
        """Move a flow from ``source_folder_id`` to ``target_folder_id``.

        A flow can belong to multiple folders simultaneously, but "move"
        means: remove it from the source folder and ensure it's linked to
        the target folder — skipping duplicate-link creation if it's
        already in the target.
        """
        self._get_folder_or_404(db, source_folder_id, tenant_id)
        self._get_folder_or_404(db, target_folder_id, tenant_id)

        if source_folder_id == target_folder_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source and target folder must differ",
            )

        cf_repo = CallFlowRepository(db)
        flow = cf_repo.find_by_id(flow_id, tenant_id=tenant_id)
        if flow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Call flow {flow_id} not found in workspace",
            )

        repo = FolderRepository(db)
        source_link = repo.find_folder_flow(source_folder_id, flow_id)
        if source_link is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Call flow {flow_id} is not in folder {source_folder_id}",
            )

        repo.remove_flow(source_link)

        target_link = repo.find_folder_flow(target_folder_id, flow_id)
        if target_link is None:
            repo.add_flow(target_folder_id, flow_id)

        db.commit()

        return {
            "flowId": str(flow_id),
            "sourceFolderId": str(source_folder_id),
            "targetFolderId": str(target_folder_id),
        }

    def list_flows_in_folder(
        self, db: Session, folder_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> dict:
        self._get_folder_or_404(db, folder_id, tenant_id)

        repo = FolderRepository(db)
        flows = repo.find_flows_by_folder(folder_id, tenant_id)

        cf_repo = CallFlowRepository(db)
        folder_ids_map = cf_repo.find_folder_ids_map([f.id for f in flows])

        return {
            "folderId": str(folder_id),
            "data": [
                call_flow_service.flow_to_list_item(f, folder_ids_map.get(f.id, []))
                for f in flows
            ],
            "total": len(flows),
        }


folder_service = FolderService()
