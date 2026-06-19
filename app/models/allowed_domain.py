from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class AllowedDomain(Base):
    """A domain whitelisted to embed the Web SDK for a workspace.

    Stored in normalized origin form (lowercase, no trailing slash, no :443)
    so it can be compared directly against a normalized Origin header —
    see app/core/origin.py.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False)
    domain = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    workspace = relationship("Tenant")

    __table_args__ = (
        Index("ix_alloweddomain_workspace_id", "workspace_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AllowedDomain id={self.id} workspace={self.workspace_id} domain={self.domain}>"
