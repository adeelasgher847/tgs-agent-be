from sqlalchemy import Boolean, Column, DateTime, Numeric, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class WalletAutoRechargeConfig(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    enabled = Column(Boolean, server_default="false", nullable=False)
    min_balance = Column(Numeric(10, 4), server_default="10.00", nullable=False)
    recharge_amount = Column(Numeric(10, 4), server_default="20.00", nullable=False)
    stripe_payment_method_id = Column(String, nullable=True)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    # Minimal durable trail of the LAST auto-recharge attempt (not a full
    # ledger — just enough for support to see what happened without cross-
    # referencing logs). Written in two phases by
    # CreditService._maybe_trigger_wallet_auto_recharge /
    # _execute_auto_recharge_charge: (1) last_trigger_status="pending" is set
    # atomically with the cooldown claim UPDATE, before the Stripe call is
    # made, so a crash mid-flight still leaves a durable "we attempted this"
    # record; (2) updated to "succeeded"/"failed" (+ last_payment_intent_id /
    # last_trigger_error) once the Stripe call resolves.
    last_payment_intent_id = Column(String, nullable=True)
    last_trigger_status = Column(String, nullable=True)
    last_trigger_error = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    tenant = relationship("Tenant", back_populates="wallet_auto_recharge_config")
