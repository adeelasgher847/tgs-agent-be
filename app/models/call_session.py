from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float
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
    
    # Call content
    call_transcript = Column(JSONB, nullable=True)  # Store as JSON array of messages
    response_times = Column(JSONB, nullable=True)  # Store response times for each interaction
    
    # Twilio specific fields
    twilio_call_sid = Column(String(255), nullable=True, index=True)
    from_number = Column(String(50), nullable=True)
    to_number = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="call_sessions")
    agent = relationship("Agent", back_populates="call_sessions")
    tenant = relationship("Tenant", back_populates="call_sessions")
    
    def __repr__(self):
        return f"<CallSession(id={self.id}, status={self.status})>"
