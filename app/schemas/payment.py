"""
Pydantic schemas for in-call Stripe payment endpoints.
"""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CreatePaymentSessionRequest(BaseModel):
    """Body for POST /api/v1/payments/session."""

    call_id: Optional[uuid.UUID] = Field(
        default=None,
        description="The live call session that triggered this payment (optional).",
    )
    amount_cents: int = Field(
        ...,
        gt=0,
        description="Amount in the smallest currency unit (e.g. cents for USD).",
        examples=[5000],
    )
    currency: str = Field(
        default="usd",
        min_length=3,
        max_length=3,
        description="Three-letter ISO 4217 currency code (lowercase).",
        examples=["usd"],
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Free-text description shown to the payer.",
        examples=["Service fee for call session"],
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CreatePaymentSessionResponse(BaseModel):
    """Response from POST /api/v1/payments/session."""

    payment_intent_id: str = Field(
        ..., description="Stripe PaymentIntent ID (pi_...)."
    )
    client_secret: str = Field(
        ..., description="Stripe client_secret for client-side confirmation."
    )
    payment_url: str = Field(
        ...,
        description=(
            "URL for the caller-facing payment page. "
            "Pass this to the agent prompt so it can direct the caller."
        ),
    )
    agent_context: str = Field(
        ...,
        description="Ready-to-use sentence to inject into the agent's prompt context.",
    )


class PaymentRecordOut(BaseModel):
    """Serialised PaymentRecord for list/detail responses."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    call_id: Optional[uuid.UUID]
    payment_intent_id: str
    amount_cents: int
    currency: str
    description: Optional[str]
    status: str
    card_last4: Optional[str]
    card_brand: Optional[str]
    created_at: str  # ISO-8601 string
    updated_at: Optional[str]

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_model(cls, record) -> "PaymentRecordOut":
        return cls(
            id=record.id,
            workspace_id=record.workspace_id,
            call_id=record.call_id,
            payment_intent_id=record.payment_intent_id,
            amount_cents=record.amount_cents,
            currency=record.currency,
            description=record.description,
            status=record.status,
            card_last4=record.card_last4,
            card_brand=record.card_brand,
            created_at=record.created_at.isoformat() if record.created_at else "",
            updated_at=record.updated_at.isoformat() if record.updated_at else None,
        )


class PaginatedPaymentResponse(BaseModel):
    """Paginated list of payment records."""

    items: list[PaymentRecordOut]
    total: int
    page: int
    per_page: int
    pages: int
