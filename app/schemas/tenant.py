from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from decimal import Decimal
import uuid

class TenantBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    # credits: int = Field(default=0, ge=0)  # New field for credit system

class TenantCreate(TenantBase):
    # Only name required, schema_name will be set automatically
    pass

class TenantOut(TenantBase):
    id: uuid.UUID
    schema_name: str
    status: str
    stripe_customer_id: str | None = Field(default=None, exclude=True)
    stripe_subscription_id: str | None = Field(default=None, exclude=True)
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class TenantCreateResponse(BaseModel):
    tenant_id: uuid.UUID
    tenant: TenantOut


class TenantPlanOut(BaseModel):
    """Response shape for GET /api/v1/tenants/plan — current plan + entitlements."""

    plan_name: str | None = None
    display_name: str | None = None
    included_minutes: int | None = None
    minutes_used_this_cycle: Decimal
    minutes_remaining: Decimal
    monthly_credits: Decimal | None = None
    free_phone_numbers: int | None = None
    max_subaccounts: int | None = None
    features: list[str] = Field(default_factory=list)
    credits_balance: Decimal

    # Subscription/billing summary (image 3/4 of the billing dashboard).
    price_monthly: Decimal | None = None
    per_minute_rate: Decimal
    subscription_status: str | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False

    model_config = ConfigDict(from_attributes=True)


class BillingPortalSessionOut(BaseModel):
    """Response shape for POST /api/v1/tenants/billing/portal-session."""

    url: str


class CancelPlanOut(BaseModel):
    """Response shape for POST /api/v1/tenants/plan/cancel."""

    status: str
    cancel_at_period_end: bool
    current_period_end: datetime | None = None


class InvoiceOut(BaseModel):
    """Single Stripe invoice entry for the tenant's billing history."""

    id: str
    number: str | None = None
    description: str | None = None
    status: str | None = None
    currency: str | None = None
    amount_due: Decimal | None = None
    amount_paid: Decimal | None = None
    created_at: datetime | None = None
    hosted_invoice_url: str | None = None
    invoice_pdf: str | None = None


class InvoiceListOut(BaseModel):
    """Response shape for GET /api/v1/tenants/billing/invoices."""

    invoices: list[InvoiceOut] = Field(default_factory=list)