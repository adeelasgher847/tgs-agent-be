from sqlalchemy import Column, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class PromptVersion(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    flow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callflow.id", ondelete="CASCADE"),
        nullable=False,
    )
    prompt_text = Column(Text, nullable=False)
    # DB-only: sanitized copy for Gemini — never returned in API responses
    gemini_prompt = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    call_flow = relationship(
        "CallFlow",
        foreign_keys=[flow_id],
        back_populates="prompt_versions",
    )

    __table_args__ = (
        Index("ix_promptversion_flow_created", "flow_id", "created_at"),
    )
