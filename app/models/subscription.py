from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, UniqueConstraint, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Subscription(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    plan_id = Column(UUID(as_uuid=True), ForeignKey('plan.id'), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id'), nullable=True, index=True)
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
    tenant = relationship("Tenant", back_populates="core_subscriptions")
    __table_args__ = (
        UniqueConstraint('user_id', 'crm_type', name='uq_user_crm_subscription'),
        # One active core (tenant-scoped, non-CRM) subscription per tenant. Legacy/future
        # CRM-addon rows keep tenant_id NULL and are excluded via the partial predicate,
        # since plain UniqueConstraint doesn't dedupe NULLs the way we need here.
        Index(
            "uq_tenant_core_subscription",
            "tenant_id",
            unique=True,
            postgresql_where=text("crm_type IS NULL AND tenant_id IS NOT NULL"),
        ),
    )
