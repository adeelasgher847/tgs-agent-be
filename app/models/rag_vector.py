from __future__ import annotations

"""
RAG vector-store models backed by pgvector.

These models live in a separate SQLAlchemy Base (VectorBase) so they can
be bound to a dedicated VECTOR_DB_URL and kept out of the main Alembic
migration flow for the primary application database.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

from pgvector.sqlalchemy import Vector

from app.core.config import settings


VectorBase = declarative_base()


class RagDocument(VectorBase):
    __tablename__ = "rag_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Multi-tenant isolation
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    # Optional per-agent scoping; NULL = shared for tenant
    agent_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    title = Column(String(255), nullable=True)
    source_type = Column(String(50), nullable=False)  # e.g. 'db', 'file', 'url', 'manual'
    source_ref = Column(String(512), nullable=True)   # e.g. table+id, file path, URL
    language = Column(String(16), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("RagChunk", back_populates="document", cascade="all, delete-orphan")


class RagChunk(VectorBase):
    __tablename__ = "rag_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    document_id = Column(UUID(as_uuid=True), ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False)

    # Duplicate tenant/agent for faster filtering without join in some queries
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    chunk_index = Column(Integer, nullable=False)  # position within document
    text = Column(Text, nullable=False)

    # pgvector column; dimension should match the embedding model used
    embedding = Column(Vector(settings.VECTOR_DIMENSION), nullable=False)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("RagDocument", back_populates="chunks")

