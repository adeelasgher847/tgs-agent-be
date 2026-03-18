from __future__ import annotations

import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class KnowledgeBaseChunk(Base):
    """
    Inventories Pinecone vector IDs for each document chunk.

    This enables:
    - safe deletion of stale vectors on re-ingest
    - stable chunk references for auditing/debugging
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("knowledgebasedocument.id"), nullable=False, index=True)

    chunk_index = Column(Integer, nullable=False, index=True)
    vector_id = Column(String(300), nullable=False, unique=True, index=True)

    # Optional: keep a short preview to make debugging easier.
    text_preview = Column(String(700), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("KnowledgeBaseDocument", back_populates="chunks")

