from sqlalchemy import Column, String, Text, DateTime, Integer, Index, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class Appointment(Base):
    """Customer appointments booked via voice agent or web."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True, index=True)
    call_session_id = Column(UUID(as_uuid=True), ForeignKey("callsession.id"), nullable=True)

    # Customer info collected during the call
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=False)
    customer_email = Column(String(255), nullable=True)
    appointment_reason = Column(Text, nullable=True)

    # Slot timing
    slot_start = Column(DateTime(timezone=True), nullable=False, index=True)
    slot_end = Column(DateTime(timezone=True), nullable=False)
    duration_minutes = Column(Integer, nullable=False, server_default="30")

    # Lifecycle: pending / confirmed / cancelled / completed / no_show
    status = Column(String(20), nullable=False, server_default="pending")

    # Booking source: voice_agent / web / api
    created_via = Column(String(20), nullable=False, server_default="web")

    notes = Column(Text, nullable=True)
    cancellation_reason = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by_user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    customer_notified_on_review_at = Column(DateTime(timezone=True), nullable=True)

    # Inbound staff WhatsApp (WATI) prompt + customer SMS after staff confirms
    staff_whatsapp_ack_token = Column(String(32), nullable=True, unique=True, index=True)
    staff_whatsapp_prompt_sent_at = Column(DateTime(timezone=True), nullable=True)
    customer_sms_confirmed_notified_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    tenant = relationship("Tenant", back_populates="appointments")
    agent = relationship("Agent", back_populates="appointments")

    __table_args__ = (
        Index(
            "uq_appointment_tenant_slot_active",
            "tenant_id",
            "slot_start",
            unique=True,
            postgresql_where=text("status != 'cancelled'"),
            sqlite_where=text("status != 'cancelled'"),
        ),
    )
