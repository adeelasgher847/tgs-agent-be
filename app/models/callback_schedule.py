from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class CallbackSchedule(Base):
    """
    Tracks each retry attempt for a missed/busy outbound call.
    Table name is auto-derived from base class: 'callbackschedule'.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # The original missed call that triggered the callback chain
    original_call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callsession.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # The agent responsible for placing the callback
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Destination number to dial
    phone_number = Column(Text, nullable=False)
    # 1-based attempt counter (1 = first retry, 2 = second, …)
    attempt_number = Column(Integer, nullable=False, default=1, server_default="1")
    # When the system should dispatch this callback (UTC stored, tz-aware)
    scheduled_at = Column(DateTime(timezone=True), nullable=False)
    # IANA timezone used to evaluate business-hours window
    timezone = Column(Text, nullable=False)
    # Lifecycle state: pending → executed | cancelled | exhausted
    status = Column(String(20), nullable=False, default="pending", server_default="pending")
    # Populated when the callback call is actually dispatched
    executed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # ARQ job ID for the deferred execute_callback task. NULL means the job has
    # not yet been submitted to Redis (recovery cron will pick it up within 60 s).
    arq_job_id = Column(String(255), nullable=True, index=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    original_call = relationship(
        "CallSession",
        foreign_keys=[original_call_id],
        back_populates="callback_schedules",
    )
    agent = relationship("Agent", back_populates="callback_schedules")

    # ── Constraints & indexes ──────────────────────────────────────────────────
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','executed','cancelled','exhausted')",
            name="ck_callbackschedule_status",
        ),
        # Polling query: WHERE status = 'pending' AND scheduled_at <= now()
        Index("ix_callbackschedule_status_scheduled_at", "status", "scheduled_at"),
    )
