from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class CallSession(Base):
    """Database model for call sessions"""
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False)
    
    # Call metadata
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), nullable=False, default="active")  # active, completed, failed, busy
    duration = Column(Integer, nullable=True)  # duration in seconds
    
    # Call type and classification
    call_type = Column(String(20), nullable=False, server_default="inbound")  # inbound, outbound, web
    success_evaluation = Column(String(20), nullable=True)  # success, fail, null
    ended_reason = Column(String(255), nullable=True)  # Customer Ended Call, Call.start.error, etc.
    
    # Cost and billing
    cost = Column(Float, nullable=True, server_default="0.0")  # Cost in USD
    cost_currency = Column(String(3), nullable=True, server_default="USD")
    
    # Call content
    call_transcript = Column(JSONB, nullable=True)  # Store as JSON array of messages
    response_times = Column(JSONB, nullable=True)  # Store response times for each interaction
    recording_url = Column(String(500), nullable=True)  # Legacy Twilio recording URL (kept for compat)

    # GCS recording (Sprint 4 — replaces Twilio recording for LiveKit calls)
    recording_gcs_path = Column(String(500), nullable=True)  # e.g. recordings/{workspaceId}/{callId}/{date}.opus
    recording_error = Column(Boolean, nullable=False, server_default="false")
    
    # Phone numbers and external IDs
    twilio_call_sid = Column(String(255), nullable=True, index=True)
    from_number = Column(String(50), nullable=True)
    to_number = Column(String(50), nullable=True)
    assistant_phone_number = Column(String(50), nullable=True)  # The phone number assigned to the assistant
    customer_phone_number = Column(String(50), nullable=True)  # The customer's phone number
    
    # Additional metadata
    call_metadata = Column(JSONB, nullable=True)  # Store additional call metadata
    transferred = Column(Boolean, nullable=False, server_default="false")  # Whether call was transferred
    
    # Optional link to the call flow that triggered this session
    call_flow_id = Column(UUID(as_uuid=True), ForeignKey("callflow.id"), nullable=True, index=True)

    # Smart Callback: points to the original missed call in a retry chain
    parent_call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callsession.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="call_sessions")
    agent = relationship("Agent", back_populates="call_sessions")
    tenant = relationship("Tenant", back_populates="call_sessions")
    call_logs = relationship("CallLog", back_populates="call_session", cascade="all, delete-orphan")
    transcript_messages = relationship("TranscriptMessage", back_populates="call_session", cascade="all, delete-orphan")
    slot_reservations = relationship("SlotReservation", back_populates="call_session", cascade="all, delete-orphan")
    call_flow = relationship("CallFlow", back_populates="call_sessions")
    # Self-referential: retry calls point back to the original missed call
    parent_call = relationship("CallSession", remote_side="CallSession.id", foreign_keys=[parent_call_id])
    callback_schedules = relationship("CallbackSchedule", back_populates="original_call", foreign_keys="CallbackSchedule.original_call_id")
    
    def __repr__(self):
        return f"<CallSession(id={self.id}, status={self.status})>"
