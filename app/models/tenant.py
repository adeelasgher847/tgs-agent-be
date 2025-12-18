from sqlalchemy import Column, String, DateTime, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Tenant(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, index=True, nullable=False)
    schema_name = Column(String, unique=False, nullable=False)
    status = Column(String, nullable=False, default="pending_payment")  # pending_payment, active, inactive
    stripe_customer_id = Column(String, nullable=True, index=True)
    stripe_subscription_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    credits = Column(Numeric(10, 4), default=0, nullable=False)  # Float credits with 4 decimal precision
    
    # Relationships
    users = relationship("User", secondary="user_tenant_association", back_populates="tenants") 
    agents = relationship("Agent", back_populates="tenant")
    call_sessions = relationship("CallSession", back_populates="tenant")
    call_logs = relationship("CallLog", back_populates="tenant")
    phone_numbers = relationship("PhoneNumber", back_populates="tenant")
    subscription = relationship("Subscription", back_populates="tenant", uselist=False)
    crm_configs = relationship("TenantCRMConfig", back_populates="tenant", cascade="all, delete-orphan")
