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
    price_per_minute: float = Field(default=0.05, ge=0)  # $0.05 per minute
    
    # Simple Credit System
    credits: Optional[int] = Field(default=0, ge=0)  # Credits included with this plan
    
    # Simple Limits
    agent_limit: int = Field(default=0, ge=0)  # Max agents
    monthly_calls_limit: int = Field(default=0, ge=0)  # Keep existing
    included_minutes: int = Field(default=0, ge=0)  # Free minutes per month
    
    # Stripe
    stripe_price_id: Optional[str] = None
    
    # Status
    is_active: bool = True
    is_popular: bool = False

class PlanCreate(PlanBase):
    pass

class PlanUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    price_monthly: Optional[int] = None
    price_per_minute: Optional[float] = None
    credits: Optional[int] = None
    agent_limit: Optional[int] = None
    monthly_calls_limit: Optional[int] = None
    included_minutes: Optional[int] = None
    stripe_price_id: Optional[str] = None
    is_active: Optional[bool] = None
    is_popular: Optional[bool] = None

class PlanOut(PlanBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
