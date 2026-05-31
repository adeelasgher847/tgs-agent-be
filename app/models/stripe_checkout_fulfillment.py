from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class StripeCheckoutFulfillment(Base):
    """One row per fulfilled Stripe Checkout session (idempotent webhook processing)."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    checkout_session_id = Column(String(255), nullable=False, unique=True, index=True)
    stripe_event_id = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
