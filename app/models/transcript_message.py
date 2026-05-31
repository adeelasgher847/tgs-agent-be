from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class TranscriptMessage(Base):
    """Database model for individual transcript messages"""
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    call_session_id = Column(UUID(as_uuid=True), ForeignKey("callsession.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # "agent" or "client"
    message = Column(Text, nullable=False)  # The actual message content
    message_type = Column(String(50), nullable=True)  # "speech", "timeout", "error", etc.
    sequence_number = Column(Integer, nullable=False)  # Order of message within the call session
    
    # Metadata
    confidence = Column(Float, nullable=True)  # Speech recognition confidence
    duration = Column(Float, nullable=True)  # Message duration in seconds
    response_time = Column(Float, nullable=True)  # Time taken to generate response
    
    # Additional data
    message_metadata = Column(JSONB, nullable=True)  # Store additional message metadata
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    call_session = relationship("CallSession", back_populates="transcript_messages")
    agent = relationship("Agent", back_populates="transcript_messages")
    user = relationship("User", back_populates="transcript_messages")
    
    def __repr__(self):
        return f"<TranscriptMessage(id={self.id}, role={self.role}, call_session_id={self.call_session_id})>"
