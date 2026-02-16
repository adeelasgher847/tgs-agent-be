from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Carrier(Base):
    """Carrier configuration for Vicidial (tenant-specific)"""
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=True, index=True)  # Nullable for global carriers
    
    # Basic Info
    name = Column(String(100), nullable=False)  # "Vonage US", "Telenyx EU"
    provider = Column(String(50), nullable=False)  # 'Vonage', 'Telenyx', 'Twilio'
    status = Column(String(20), nullable=False, default="active")  # active, inactive
    description = Column(Text, nullable=True)
    
    # SIP Credentials (encrypted)
    sip_username = Column(String(255), nullable=True)  # SIP username (encrypted)
    sip_password = Column(String(500), nullable=True)  # SIP password (encrypted)
    sip_server = Column(String(255), nullable=True)  # sip.vonage.com, sip.telnyx.com
    sip_port = Column(Integer, nullable=True, default=5060)
    
    # Vicidial Integration
    vicidial_carrier_id = Column(String(50), nullable=True)  # Carrier ID in Vicidial
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="carriers")
    phone_numbers = relationship("PhoneNumber", back_populates="carrier")
    
    def __repr__(self):
        return f"<Carrier(id={self.id}, name={self.name}, provider={self.provider})>"
