from sqlalchemy import Column, String, Integer, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Plan(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, unique=True, index=True, nullable=False)  # "free", "pro"
    display_name = Column(String, nullable=False)  # "Free Plan", "Pro Plan"
    description = Column(String, nullable=True)
    price_monthly = Column(Integer, nullable=True)  # Price in cents
    price_yearly = Column(Integer, nullable=True)  # Price in cents
    agent_limit = Column(Integer, nullable=False, default=0)
    monthly_calls_limit = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, default=True, nullable=False)
    stripe_price_id = Column(String, nullable=True)  # Stripe price ID
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
