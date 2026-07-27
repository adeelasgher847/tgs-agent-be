from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
import uuid

class PlanBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    
    # Simple Pricing (like Vapi)
    price_monthly: int | None = None  # Price in cents (0 for free)
    
    # Stripe
    stripe_price_id: str | None = None
    # CRM plan: monday, clickup, jira, trello (optional)
    crm_type: str | None = None
    
    # Status
    is_active: bool = True
    is_popular: bool = False

class PlanCreate(PlanBase):
    pass

class PlanUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    price_monthly: int | None = None
    stripe_price_id: str | None = None
    crm_type: str | None = None
    is_active: bool | None = None
    is_popular: bool | None = None

class PlanOut(PlanBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None = None
    
    model_config = ConfigDict(from_attributes=True)
