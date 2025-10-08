from sqlalchemy import Column, String, Integer, Boolean, DateTime
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
