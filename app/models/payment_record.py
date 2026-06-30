"""
PaymentRecord — persists Stripe PaymentIntent state for in-call payments.

One row is created when the agent initiates a payment session and is updated
by the stripe-webhook endpoint on payment_intent.succeeded / payment_intent.payment_failed.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class PaymentRecord(Base):
    """Tracks Stripe PaymentIntent state for in-call payments."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Workspace that owns this payment record
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The live call session that triggered this payment (nullable — future off-call use)
    call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("callsession.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Stripe identifiers
    payment_intent_id = Column(String(255), nullable=False, unique=True, index=True)

    # Payment details
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(10), nullable=False, server_default="usd")
    description = Column(Text, nullable=True)

    # Lifecycle status: pending | succeeded | failed
    status = Column(String(50), nullable=False, server_default="pending")

    # Card details — populated on payment_intent.succeeded from charge data
    card_last4 = Column(String(4), nullable=True)
    card_brand = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    workspace = relationship("Tenant", foreign_keys=[workspace_id])
    call_session = relationship("CallSession", foreign_keys=[call_id])

    __table_args__ = (
        Index("idx_paymentrecord_workspace_created", "workspace_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<PaymentRecord(id={self.id}, pi={self.payment_intent_id}, "
            f"status={self.status}, amount={self.amount_cents} {self.currency})>"
        )
