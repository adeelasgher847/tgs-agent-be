from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Subscription(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    plan_id = Column(UUID(as_uuid=True), ForeignKey('plan.id'), nullable=False)
    stripe_subscription_id = Column(String, unique=True, nullable=True, index=True)
    stripe_customer_id = Column(String, nullable=True, index=True)
    stripe_session_id = Column(String, nullable=True, index=True)
    crm_type = Column(String(50), nullable=True, index=True)  # monday, clickup, jira, trello
    status = Column(String, nullable=False, default="active")  # active, canceled, past_due, unpaid
    current_period_start = Column(DateTime(timezone=True), nullable=True)
    current_period_end = Column(DateTime(timezone=True), nullable=True)
    cancel_at_period_end = Column(Boolean, default=False, nullable=False)
    canceled_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    plan = relationship("Plan", back_populates="subscriptions")
    __table_args__ = (
        UniqueConstraint('user_id', 'crm_type', name='uq_user_crm_subscription'),
    )
