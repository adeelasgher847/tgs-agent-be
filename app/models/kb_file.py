from __future__ import annotations

import uuid
from sqlalchemy import Column, Text, BigInteger, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class KbFile(Base):
    """Tracks an uploaded file that has been (or is being) ingested into a knowledge base."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    kb_id = Column(
        UUID(as_uuid=True),
        ForeignKey("knowledgebase.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename = Column(Text, nullable=False)
    size_bytes = Column(BigInteger, nullable=True)
    file_type = Column(Text, nullable=True)
    gcs_path = Column(Text, nullable=True)

    # processing | ready | error
    status = Column(Text, nullable=False, default="processing")
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    knowledge_base = relationship("KnowledgeBase", back_populates="files")
    chunks = relationship("KbChunk", back_populates="kb_file")
