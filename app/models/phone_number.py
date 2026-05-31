from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class PhoneNumber(Base):
    """Simple phone number model"""
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    
    # Basic phone number info
    phone_number = Column(String(20), nullable=False, index=True)  # +1234567890
    label = Column(String(100), nullable=True)  # Custom label
    status = Column(String(20), nullable=False, default="active")  # active, inactive
    
    # Twilio integration
    twilio_phone_number_sid = Column(String(255), nullable=True, index=True)
    twilio_account_sid = Column(String(255), nullable=True)  # Custom Twilio Account SID
    twilio_auth_token = Column(String(500), nullable=True)  # Custom Twilio Auth Token (encrypted)
    
    # Assistant assignment
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="phone_numbers")
    
    # Global unique constraint: a phone number can belong to only one tenant.
    __table_args__ = (
        UniqueConstraint('phone_number', name='uq_phone_number_global'),
    )
    
    def __repr__(self):
        return f"<PhoneNumber(id={self.id}, number={self.phone_number}, label={self.label})>"
