from __future__ import annotations

import uuid
from sqlalchemy import Column, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class KnowledgeBase(Base):
    """A named knowledge base scoped to a workspace (tenant)."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    files = relationship("KbFile", back_populates="knowledge_base", cascade="all, delete-orphan")
    chunks = relationship("KbChunk", back_populates="knowledge_base", cascade="all, delete-orphan")


# Legacy alias kept for code that still references KnowledgeBaseDocument.
# Callers should migrate to KnowledgeBase directly.
KnowledgeBaseDocument = KnowledgeBase
