from pydantic import BaseModel, ConfigDict, Field, model_validator
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

    # Daily LLM token budget cap (observe-only when None). See
    # Tenant.llm_token_budget_daily.
    llm_token_budget_daily: int | None = None

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


class AutoRechargeSettingsOut(BaseModel):
    """Response shape for GET /api/v1/tenants/billing/auto-recharge."""

    enabled: bool
    min_balance: Decimal
    recharge_amount: Decimal
    stripe_payment_method_id: str | None = None
    last_triggered_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AutoRechargeSettingsUpsert(BaseModel):
    """Request body for PUT /api/v1/tenants/billing/auto-recharge."""

    enabled: bool = Field(
        ...,
        description=(
            "When true, requires `stripe_payment_method_id` to also be set — "
            "auto-recharge can never be enabled without a saved payment "
            "method to charge off-session."
        ),
    )
    min_balance: Decimal = Field(..., ge=0)
    recharge_amount: Decimal = Field(..., gt=0)
    stripe_payment_method_id: str | None = Field(
        default=None,
        description=(
            "Stripe PaymentMethod id to charge off-session when the balance "
            "drops below min_balance. Required whenever `enabled=true`."
        ),
    )

    @model_validator(mode="after")
    def _validate_recharge_config(self) -> "AutoRechargeSettingsUpsert":
        if self.enabled and not self.stripe_payment_method_id:
            raise ValueError(
                "auto-recharge requires a saved payment method — set "
                "stripe_payment_method_id or leave enabled=false"
            )

        # Guardrail against a config that would immediately re-trigger every
        # single cooldown window forever: if the recharge only tops the
        # balance up to less than half of min_balance above zero, the very
        # next deduction after the recharge lands is highly likely to drop
        # the tenant back below min_balance again, re-firing another
        # off-session charge as soon as the cooldown clears. Requiring
        # recharge_amount >= 0.5 * min_balance is a conservative floor (not
        # a guarantee of no re-fire, since usage rate varies) that at least
        # rules out configs that are clearly too small relative to their own
        # threshold to be a "top-up" rather than a "trickle recharge." Only
        # enforced when enabled=True — a disabled config can hold any values
        # since it will never actually fire a charge.
        if self.enabled:
            min_recharge = self.min_balance * Decimal("0.5")
            if self.recharge_amount < min_recharge:
                raise ValueError(
                    f"recharge_amount ({self.recharge_amount}) is too small relative to "
                    f"min_balance ({self.min_balance}) — must be at least half of "
                    f"min_balance ({min_recharge}) to avoid re-triggering on almost "
                    "every cooldown window"
                )
        return self
