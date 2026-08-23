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

    model_config = ConfigDict(from_attributes=True)