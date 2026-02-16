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
    
    # Dialer Type
    dialer_type = Column(String(20), nullable=False, default="twilio", server_default="twilio")  # twilio, vicidial
    
    # Carrier Reference (for Vicidial)
    carrier_id = Column(UUID(as_uuid=True), ForeignKey("carrier.id"), nullable=True, index=True)
    caller_id_number = Column(String(20), nullable=True)  # User's caller ID number
    
    # Twilio integration
    twilio_phone_number_sid = Column(String(255), nullable=True, index=True)
    twilio_account_sid = Column(String(255), nullable=True)  # Custom Twilio Account SID
    twilio_auth_token = Column(String(500), nullable=True)  # Custom Twilio Auth Token (encrypted)
    
    # Vicidial integration
    vicidial_cid_group_id = Column(String(50), nullable=True)  # CID Group ID in Vicidial
    vicidial_campaign_id = Column(String(50), nullable=True)  # Campaign ID in Vicidial
    
    # Assistant assignment
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="phone_numbers")
    carrier = relationship("Carrier", back_populates="phone_numbers")
    
    # Composite unique constraint: phone_number must be unique within each tenant
    __table_args__ = (
        UniqueConstraint('tenant_id', 'phone_number', name='uq_phone_number_per_tenant'),
    )
    
    def __repr__(self):
        return f"<PhoneNumber(id={self.id}, number={self.phone_number}, label={self.label}, dialer_type={self.dialer_type})>"
