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
    recording_url = Column(String(500), nullable=True)  # URL to the call recording
    
    # Dialer Type
    dialer_type = Column(String(20), nullable=False, default="twilio", server_default="twilio")  # twilio, vicidial
    
    # Phone numbers and external IDs
    twilio_call_sid = Column(String(255), nullable=True, index=True)  # Twilio call SID
    vicidial_call_id = Column(String(255), nullable=True, index=True)  # Vicidial call ID
    vicidial_lead_id = Column(String(255), nullable=True)  # Vicidial lead ID
    from_number = Column(String(50), nullable=True)
    to_number = Column(String(50), nullable=True)
    assistant_phone_number = Column(String(50), nullable=True)  # The phone number assigned to the assistant
    customer_phone_number = Column(String(50), nullable=True)  # The customer's phone number
    
    # Additional metadata
    call_metadata = Column(JSONB, nullable=True)  # Store additional call metadata
    transferred = Column(Boolean, nullable=False, server_default="false")  # Whether call was transferred
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="call_sessions")
    agent = relationship("Agent", back_populates="call_sessions")
    tenant = relationship("Tenant", back_populates="call_sessions")
    call_logs = relationship("CallLog", back_populates="call_session", cascade="all, delete-orphan")
    transcript_messages = relationship("TranscriptMessage", back_populates="call_session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<CallSession(id={self.id}, status={self.status})>"
