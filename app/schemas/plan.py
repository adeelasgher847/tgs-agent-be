from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
import uuid

class PlanBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    
    # Simple Pricing (like Vapi)
    price_monthly: Optional[int] = None  # Price in cents (0 for free)
    
    # Stripe
    stripe_price_id: Optional[str] = None
    crm_type: Optional[str] = None
    
    # Status
    is_active: bool = True
    is_popular: bool = False

class PlanCreate(PlanBase):
    pass

class PlanUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    price_monthly: Optional[int] = None
    stripe_price_id: Optional[str] = None
    crm_type: Optional[str] = None
    is_active: Optional[bool] = None
    is_popular: Optional[bool] = None

class PlanOut(PlanBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
