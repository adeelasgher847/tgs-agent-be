from __future__ import annotations

import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class KnowledgeBaseDocument(Base):
    """
    Tracks a logical knowledge base document/source inside the app DB.

    The actual embeddings live in Pinecone; this table provides:
    - deterministic document identity for updates
    - version/source metadata
    - chunk inventory for safe vector deletion on re-ingest
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True, index=True)

    title = Column(String(255), nullable=False)
    source_type = Column(String(120), nullable=False)
    source_ref = Column(String(512), nullable=False)  # URL/slug/path; app-defined
    version = Column(String(120), nullable=False, default="v1")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    is_active = Column(Boolean, nullable=False, default=True, server_default="true")

    chunks = relationship("KnowledgeBaseChunk", back_populates="document", cascade="all, delete-orphan")

