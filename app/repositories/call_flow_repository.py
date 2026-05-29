"""Call flow repository — pure SQL access, no business logic."""
from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.models.call_flow import CallFlow
from app.models.agent import Agent


class CallFlowRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: dict[str, Any]) -> CallFlow:
        flow = CallFlow(**data)
        self.db.add(flow)
        self.db.flush()
        self.db.refresh(flow)
        return flow

    def find_by_id(
        self,
        flow_id: uuid.UUID,
        *,
        include_deleted: bool = False,
        load_relations: bool = False,
    ) -> Optional[CallFlow]:
        stmt = select(CallFlow).where(CallFlow.id == flow_id)
        if not include_deleted:
            stmt = stmt.where(CallFlow.is_deleted == False)  # noqa: E712
        if load_relations:
            stmt = stmt.options(
                joinedload(CallFlow.agent),
                joinedload(CallFlow.prompt_versions),
            )
        return self.db.execute(stmt).unique().scalar_one_or_none()

    def find_by_workspace(
        self,
        workspace_id: uuid.UUID,
        *,
        page: int = 1,
        limit: int = 20,
    ) -> tuple[list[CallFlow], int]:
        offset = (page - 1) * limit
        filters = [
            CallFlow.tenant_id == workspace_id,
            CallFlow.is_deleted == False,  # noqa: E712
        ]
        total = int(
            self.db.execute(
                select(func.count(CallFlow.id)).where(*filters)
            ).scalar_one()
        )
        rows = (
            self.db.execute(
                select(CallFlow)
                .where(*filters)
                .options(joinedload(CallFlow.agent))
                .order_by(CallFlow.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            .unique()
            .scalars()
            .all()
        )
        return list(rows), total

    def update(self, flow: CallFlow, fields: dict[str, Any]) -> CallFlow:
        for key, value in fields.items():
            setattr(flow, key, value)
        self.db.flush()
        self.db.refresh(flow)
        return flow

    def soft_delete(self, flow: CallFlow) -> None:
        flow.is_deleted = True
        self.db.flush()
