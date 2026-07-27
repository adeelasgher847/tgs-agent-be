"""Workspace (tenant) repository — encapsulates all SQL access for workspace CRUD.

Soft-deleted rows (``deleted_at IS NOT NULL``) are excluded from all lookups.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.tenant import Tenant


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


class WorkspaceRepository:
    """Sync repository for the ``tenant`` table (workspaces)."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------ reads

    def find_by_id(self, workspace_id: uuid.UUID) -> Optional[Tenant]:
        stmt = select(Tenant).where(
            Tenant.id == workspace_id,
            Tenant.deleted_at.is_(None),
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def find_by_name(self, name: str) -> Optional[Tenant]:
        stmt = select(Tenant).where(
            Tenant.name == name,
            Tenant.deleted_at.is_(None),
        )
        return self.db.execute(stmt).scalar_one_or_none()

    # ----------------------------------------------------------------- writes

    def create(self, name: str) -> Tenant:
        tenant = Tenant(
            name=name,
            schema_name=self._next_schema_name(name),
            status="active",
            credits=0,
        )
        self.db.add(tenant)
        self.db.commit()
        self.db.refresh(tenant)
        return tenant

    def update_name(self, tenant: Tenant, name: str) -> Tenant:
        tenant.name = name
        self.db.commit()
        self.db.refresh(tenant)
        return tenant

    def soft_delete(self, tenant: Tenant) -> None:
        tenant.deleted_at = datetime.now(timezone.utc)
        self.db.commit()

    # ---------------------------------------------------------------- helpers

    def _next_schema_name(self, name: str) -> str:
        base = _slugify(name) or f"ws_{uuid.uuid4().hex[:8]}"
        candidate = f"{base}_schema"
        counter = 1
        while self._schema_taken(candidate):
            candidate = f"{base}_{counter}_schema"
            counter += 1
        return candidate

    def _schema_taken(self, schema_name: str) -> bool:
        stmt = select(Tenant.id).where(Tenant.schema_name == schema_name)
        return self.db.execute(stmt).first() is not None
