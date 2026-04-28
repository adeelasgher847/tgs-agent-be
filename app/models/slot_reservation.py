"""Temporary in-call slot holds until post-call booking or release."""

import uuid
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class SlotReservation(Base):
    """
    A caller holds a calendar slot for the duration of a voice call.
    Other booking flows must not see this window as free while status is active.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    call_session_id = Column(
        UUID(as_uuid=True), ForeignKey("callsession.id"), nullable=False, index=True
    )
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True, index=True)

    slot_start = Column(DateTime(timezone=True), nullable=False, index=True)
    slot_end = Column(DateTime(timezone=True), nullable=False)

    # active: holding the slot; consumed: finalized into an appointment; released: call ended or superseded
    status = Column(String(20), nullable=False, server_default="active", index=True)

    # customer_name, customer_phone, customer_email, appointment_reason, notes, etc.
    metadata_json = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    call_session = relationship("CallSession", back_populates="slot_reservations")
    tenant = relationship("Tenant")
    agent = relationship("Agent")

    __table_args__ = (
        Index(
            "ix_slotreservation_tenant_active_time",
            "tenant_id",
            "status",
            "slot_start",
        ),
    )
