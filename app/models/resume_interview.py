import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.db.base_class import Base


class ResumeInterview(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resume.id"), nullable=False, index=True)
    job_description_id = Column(
        UUID(as_uuid=True),
        ForeignKey("jobdescription.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=False, index=True)
    call_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callsession.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    candidate_phone = Column(String(64), nullable=False)
    scheduled_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(40), nullable=False, index=True, server_default="SCHEDULE_REQUESTED")

    crm_type = Column(String(20), nullable=True)
    crm_item_id = Column(String(128), nullable=True, index=True)
    crm_batch_id = Column(String(128), nullable=True, index=True)
    phone_number_id = Column(UUID(as_uuid=True), ForeignKey("phonenumber.id"), nullable=True)
    twilio_call_sid = Column(String(255), nullable=True, index=True)

    attempt_count = Column(Integer, nullable=False, server_default="0")
    last_error = Column(Text, nullable=True)
    metadata_json = Column(JSONB, nullable=True)

    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ResumeInterviewEvent(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    resume_interview_id = Column(
        UUID(as_uuid=True),
        ForeignKey("resumeinterview.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type = Column(String(80), nullable=False, index=True)
    event_payload = Column(JSONB, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
