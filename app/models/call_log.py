from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class CallLog(Base):
    """Database model for call logs - separate from call sessions for better logging"""
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    call_session_id = Column(UUID(as_uuid=True), ForeignKey("callsession.id"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    
    # Call identification
    call_id = Column(String(255), nullable=False, index=True)  # Shortened ID for display
    external_call_id = Column(String(255), nullable=True, index=True)  # Twilio SID or other external ID
    
    # Call classification
    call_type = Column(String(20), nullable=False, server_default="inbound")  # inbound, outbound, web
    success_evaluation = Column(String(20), nullable=True)  # success, fail, null
    ended_reason = Column(String(255), nullable=True)  # Customer Ended Call, Call.start.error, etc.
    transferred = Column(Boolean, nullable=False, server_default="false")  # Whether call was transferred
    
    # Phone numbers
    assistant_phone_number = Column(String(50), nullable=True)
    customer_phone_number = Column(String(50), nullable=True)
    
    # Cost and billing
    cost = Column(Float, nullable=True, server_default="0.0")  # Cost in USD
    cost_currency = Column(String(3), nullable=True, server_default="USD")
    
    # Call timing
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Integer, nullable=True)  # duration in seconds
    
    # Additional metadata
    call_metadata = Column(JSONB, nullable=True)  # Store additional call metadata
    notes = Column(Text, nullable=True)  # Manual notes about the call
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    call_session = relationship("CallSession", back_populates="call_logs")
    tenant = relationship("Tenant", back_populates="call_logs")
    
    def __repr__(self):
        return f"<CallLog(id={self.id}, call_id={self.call_id}, type={self.call_type})>"
