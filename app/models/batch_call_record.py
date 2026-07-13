from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class BatchCallRecord(Base):
    """One call entry within a BatchJob."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey("batchjob.id", ondelete="CASCADE"), nullable=False)

    phone_number = Column(String(50), nullable=False)
    variables = Column(JSONB, nullable=True)

    # waiting → active → completed | failed | cancelled
    status = Column(String(20), nullable=False, default="waiting", server_default="waiting")

    # FK to callsession set once Twilio call is initiated
    call_id = Column(UUID(as_uuid=True), ForeignKey("callsession.id", ondelete="SET NULL"), nullable=True)

    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    last_error = Column(Text, nullable=True)
    # Populated when a retry is scheduled (e.g. busy/no_answer with 30-min gap)
    next_attempt_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())

    # Relationships
    batch_job = relationship("BatchJob", back_populates="records")
    call_session = relationship("CallSession")

    __table_args__ = (
        Index("ix_batchcallrecord_batch_job_id", "batch_job_id"),
        Index("ix_batchcallrecord_status", "status"),
        Index("ix_batchcallrecord_call_id", "call_id"),
        Index("ix_batchcallrecord_job_status_pickup", "batch_job_id", "status", "next_attempt_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BatchCallRecord id={self.id} phone={self.phone_number} status={self.status}>"
