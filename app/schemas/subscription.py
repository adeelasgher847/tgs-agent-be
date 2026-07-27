from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
import uuid
from app.schemas.plan import PlanOut

class SubscriptionBase(BaseModel):
    status: str = Field(..., pattern="^(active|canceled|past_due|unpaid)$")
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False

class SubscriptionCreate(SubscriptionBase):
    user_id: uuid.UUID
    plan_id: uuid.UUID
    crm_type: str | None = None
    stripe_subscription_id: str | None = None
    stripe_customer_id: str | None = None

class SubscriptionUpdate(BaseModel):
    status: str | None = None
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool | None = None
    canceled_at: datetime | None = None
    stripe_subscription_id: str | None = None
    stripe_customer_id: str | None = None

class SubscriptionOut(SubscriptionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    plan_id: uuid.UUID
    crm_type: str | None = None
    stripe_subscription_id: str | None = None
    stripe_customer_id: str | None = None
    canceled_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None
    plan: PlanOut | None = None
    
    model_config = ConfigDict(from_attributes=True)

class SubscriptionWithUsage(SubscriptionOut):
    current_usage: dict | None = None  # Will be populated with current month usage


class UserSubscriptionSummary(BaseModel):
    """Minimal subscription info for current user: status, period start/end, crm_type."""
    status: str
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    crm_type: str | None = None

    model_config = ConfigDict(from_attributes=True)
