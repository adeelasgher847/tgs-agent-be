from sqlalchemy import Column, String, Integer, Boolean, DateTime, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Plan(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, unique=True, index=True, nullable=False)  # "free", "starter", "pro"
    display_name = Column(String, nullable=False)  # "Free", "Starter", "Pro"
    description = Column(String, nullable=True)
    
    # Simple Pricing (like Vapi)
    price_monthly = Column(Integer, nullable=True)  # Price in cents (0 for free)
    
    # Stripe
    stripe_price_id = Column(String, nullable=True)
    # CRM plan: monday, clickup, jira, trello (nullable for non-CRM plans)
    crm_type = Column(String(50), nullable=True, index=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_popular = Column(Boolean, default=False)  # Show as "Popular"

    # Studio/Agency (core product) tier fields — nullable, unused by CRM-addon plans
    included_minutes = Column(Integer, nullable=True)  # total inbound+outbound minutes per billing cycle
    monthly_credits = Column(Numeric(10, 2), nullable=True)  # credit grant per renewal; 1 credit = $1
    max_subaccounts = Column(Integer, nullable=True)  # cap on whitelabel sub-accounts; NULL = unlimited
    free_phone_numbers = Column(Integer, nullable=True)  # cap on phone numbers included free
    features = Column(JSONB, nullable=True)  # list of display strings, e.g. ["WhiteLabel workspaces", ...]

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
