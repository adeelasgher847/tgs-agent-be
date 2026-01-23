from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
import uuid
from app.schemas.plan import PlanOut

class SubscriptionBase(BaseModel):
    status: str = Field(..., pattern="^(active|canceled|past_due|unpaid)$")
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: bool = False

class SubscriptionCreate(SubscriptionBase):
    user_id: uuid.UUID
    plan_id: uuid.UUID
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None

class SubscriptionUpdate(BaseModel):
    status: Optional[str] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: Optional[bool] = None
    canceled_at: Optional[datetime] = None
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None

class SubscriptionOut(SubscriptionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    plan_id: uuid.UUID
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    canceled_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    plan: Optional[PlanOut] = None
    
    model_config = ConfigDict(from_attributes=True)

class SubscriptionWithUsage(SubscriptionOut):
    current_usage: Optional[dict] = None  # Will be populated with current month usage
