from sqlalchemy import Column, String, Integer, Boolean, DateTime, Float
from sqlalchemy.dialects.postgresql import UUID
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
    price_per_minute = Column(Float, default=0.05)  # $0.05 per minute (Vapi's rate)
    
    # Simple Credit System
    credits = Column(Integer, nullable=True, default=0)  # Credits included with this plan
    
    # Simple Limits
    agent_limit = Column(Integer, default=0)  # Max agents
    monthly_calls_limit = Column(Integer, default=0)  # Keep existing column
    included_minutes = Column(Integer, default=0)  # Free minutes per month
    
    # Stripe
    stripe_price_id = Column(String, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_popular = Column(Boolean, default=False)  # Show as "Popular"
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
