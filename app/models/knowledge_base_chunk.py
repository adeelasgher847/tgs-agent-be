from __future__ import annotations

import uuid
from sqlalchemy import Column, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.db.base_class import Base


class KbChunk(Base):
    """A single text chunk from a knowledge base, with its pgvector embedding."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    kb_id = Column(
        UUID(as_uuid=True),
        ForeignKey("knowledgebase.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_id = Column(
        UUID(as_uuid=True),
        ForeignKey("kbfile.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    # Column named "metadata" in DB; "metadata" is reserved by SQLAlchemy's Declarative API.
    chunk_metadata = Column("metadata", JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    knowledge_base = relationship("KnowledgeBase", back_populates="chunks")
    kb_file = relationship("KbFile", back_populates="chunks")


# Legacy alias
KnowledgeBaseChunk = KbChunk
