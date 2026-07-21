"""Prompt version repository — pure SQL access, no business logic."""
from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.prompt_version import PromptVersion


class PromptVersionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: dict[str, Any]) -> PromptVersion:
        version = PromptVersion(**data)
        self.db.add(version)
        self.db.flush()
        self.db.refresh(version)
        return version

    def find_by_id(self, version_id: uuid.UUID) -> Optional[PromptVersion]:
        stmt = select(PromptVersion).where(
            PromptVersion.id == version_id,
            PromptVersion.is_deleted == False,  # noqa: E712
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def find_by_flow(
        self,
        flow_id: uuid.UUID,
        *,
        order_desc: bool = True,
    ) -> list[PromptVersion]:
        stmt = select(PromptVersion).where(
            PromptVersion.flow_id == flow_id,
            PromptVersion.is_deleted == False,  # noqa: E712
        )
        stmt = stmt.order_by(
            PromptVersion.created_at.desc() if order_desc else PromptVersion.created_at.asc()
        )
        return list(self.db.execute(stmt).scalars().all())

    def count_by_flow(self, flow_id: uuid.UUID) -> int:
        stmt = select(func.count(PromptVersion.id)).where(
            PromptVersion.flow_id == flow_id,
            PromptVersion.is_deleted == False,  # noqa: E712
        )
        return int(self.db.execute(stmt).scalar_one())

    def find_oldest_by_flow(self, flow_id: uuid.UUID) -> Optional[PromptVersion]:
        stmt = (
            select(PromptVersion)
            .where(
                PromptVersion.flow_id == flow_id,
                PromptVersion.is_deleted == False,  # noqa: E712
            )
            .order_by(PromptVersion.created_at.asc())
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def find_oldest_deletable(
        self,
        flow_id: uuid.UUID,
        exclude_version_id: Optional[uuid.UUID],
    ) -> Optional[PromptVersion]:
        """Oldest version by created_at that is NOT the current active version."""
        stmt = select(PromptVersion).where(
            PromptVersion.flow_id == flow_id,
            PromptVersion.is_deleted == False,  # noqa: E712
        )
        if exclude_version_id is not None:
            stmt = stmt.where(PromptVersion.id != exclude_version_id)
        stmt = stmt.order_by(PromptVersion.created_at.asc()).limit(1)
        return self.db.execute(stmt).scalar_one_or_none()

    def soft_delete(self, version: PromptVersion) -> None:
        version.is_deleted = True
        self.db.add(version)
        self.db.flush()

    def delete(self, version: PromptVersion) -> None:
        self.db.delete(version)
        self.db.flush()
