"""Agent repository — SQL access for ticket agent-management CRUD."""
from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.models.agent import Agent


class AgentRepository:
    """Sync repository for the ``agent`` table (Sprint 2 ticket CRUD)."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, data: dict[str, Any]) -> Agent:
        agent = Agent(**data)
        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)
        return agent

    def find_by_id(
        self,
        agent_id: uuid.UUID,
        *,
        include_deleted: bool = False,
        load_transfer_route: bool = False,
    ) -> Optional[Agent]:
        stmt = select(Agent).where(Agent.id == agent_id)
        if not include_deleted:
            stmt = stmt.where(Agent.is_deleted == False)  # noqa: E712
        if load_transfer_route:
            stmt = stmt.options(joinedload(Agent.transfer_route))
        return self.db.execute(stmt).scalar_one_or_none()

    def find_by_workspace(
        self,
        workspace_id: uuid.UUID,
        *,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
    ) -> tuple[list[Agent], int]:
        offset = (page - 1) * limit
        filters = [
            Agent.tenant_id == workspace_id,
            Agent.is_deleted == False,  # noqa: E712
        ]
        if search and search.strip():
            term = search.strip().lower()
            filters.append(func.lower(Agent.name).like(f"%{term}%"))

        total = int(
            self.db.execute(select(func.count(Agent.id)).where(*filters)).scalar_one()
        )
        rows = (
            self.db.execute(
                select(Agent)
                .where(*filters)
                .order_by(Agent.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return list(rows), total

    def count_active_by_workspace(self, workspace_id: uuid.UUID) -> int:
        stmt = select(func.count(Agent.id)).where(
            Agent.tenant_id == workspace_id,
            Agent.is_deleted == False,  # noqa: E712
        )
        return int(self.db.execute(stmt).scalar_one())

    def find_by_name_in_workspace(
        self, workspace_id: uuid.UUID, name: str
    ) -> Optional[Agent]:
        normalized = name.strip().lower()
        stmt = select(Agent).where(
            Agent.tenant_id == workspace_id,
            Agent.is_deleted == False,  # noqa: E712
            func.lower(Agent.name) == normalized,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def update(self, agent: Agent, fields: dict[str, Any]) -> Agent:
        for key, value in fields.items():
            setattr(agent, key, value)
        self.db.commit()
        self.db.refresh(agent)
        return agent

    def soft_delete(self, agent: Agent, *, updated_by: Optional[uuid.UUID] = None) -> None:
        agent.is_deleted = True
        if updated_by is not None:
            agent.updated_by = updated_by
        self.db.commit()
