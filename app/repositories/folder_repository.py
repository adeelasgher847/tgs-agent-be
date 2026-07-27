"""Folder repository — pure SQL access, no business logic."""
from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.folder import Folder
from app.models.folder_flow import FolderFlow


class FolderRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: dict[str, Any]) -> Folder:
        folder = Folder(**data)
        self.db.add(folder)
        self.db.flush()
        self.db.refresh(folder)
        return folder

    def find_by_id(
        self,
        folder_id: uuid.UUID,
        *,
        include_deleted: bool = False,
    ) -> Optional[Folder]:
        stmt = select(Folder).where(Folder.id == folder_id)
        if not include_deleted:
            stmt = stmt.where(Folder.is_deleted == False)  # noqa: E712
        return self.db.execute(stmt).scalar_one_or_none()

    def find_by_workspace(self, workspace_id: uuid.UUID) -> list[Folder]:
        stmt = (
            select(Folder)
            .where(Folder.tenant_id == workspace_id, Folder.is_deleted == False)  # noqa: E712
            .order_by(Folder.created_at.desc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def update(self, folder: Folder, fields: dict[str, Any]) -> Folder:
        for key, value in fields.items():
            setattr(folder, key, value)
        self.db.flush()
        self.db.refresh(folder)
        return folder

    def soft_delete(self, folder: Folder) -> None:
        folder.is_deleted = True
        self.db.flush()

    # ── FolderFlow join table ──────────────────────────────────────────────

    def find_folder_flow(
        self, folder_id: uuid.UUID, flow_id: uuid.UUID
    ) -> Optional[FolderFlow]:
        stmt = select(FolderFlow).where(
            FolderFlow.folder_id == folder_id,
            FolderFlow.flow_id == flow_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def add_flow(self, folder_id: uuid.UUID, flow_id: uuid.UUID) -> FolderFlow:
        link = FolderFlow(folder_id=folder_id, flow_id=flow_id)
        self.db.add(link)
        self.db.flush()
        self.db.refresh(link)
        return link
