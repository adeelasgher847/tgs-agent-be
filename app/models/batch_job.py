from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class BatchJob(Base):
    """Tracks a batch outbound-call campaign."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id", ondelete="SET NULL"), nullable=True)

    # pending → processing → completed | cancelled | failed
    status = Column(String(20), nullable=False, default="pending", server_default="pending")

    # Denormalized counts kept in sync by the worker on every state transition.
    total_count = Column(Integer, nullable=False, default=0, server_default="0")
    waiting_count = Column(Integer, nullable=False, default=0, server_default="0")
    active_count = Column(Integer, nullable=False, default=0, server_default="0")
    completed_count = Column(Integer, nullable=False, default=0, server_default="0")
    failed_count = Column(Integer, nullable=False, default=0, server_default="0")

    # Answering Machine Detection (AMD) behaviour for this batch
    voicemail_action = Column(Text, nullable=False, default="skip", server_default="skip")
    voicemail_message = Column(Text, nullable=True)
    voicemail_skipped_count = Column(Integer, nullable=False, default=0, server_default="0")
    voicemail_message_left_count = Column(Integer, nullable=False, default=0, server_default="0")

    # Rotated outbound caller-ID number actually used for this batch (set when
    # the agent's bound number was spam-flagged and a clean replacement was found).
    actual_from_number = Column(String(20), nullable=True)

    s3_path = Column(Text, nullable=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    workspace = relationship("Tenant")
    agent = relationship("Agent")
    records = relationship("BatchCallRecord", back_populates="batch_job", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_batchjob_workspace_id", "workspace_id"),
        Index("ix_batchjob_agent_id", "agent_id"),
        Index("ix_batchjob_status", "status"),
        CheckConstraint(
            "voicemail_action IN ('skip', 'leave_message', 'continue')",
            name="ck_batchjob_voicemail_action",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BatchJob id={self.id} status={self.status} total={self.total_count}>"
