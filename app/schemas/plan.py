from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
import uuid

class PlanBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    price_monthly: Optional[int] = None  # Price in cents
    price_yearly: Optional[int] = None   # Price in cents
    agent_limit: int = Field(..., ge=0)
    monthly_calls_limit: int = Field(..., ge=0)
    is_active: bool = True

class PlanCreate(PlanBase):
    stripe_price_id: Optional[str] = None

class PlanUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    price_monthly: Optional[int] = None
    price_yearly: Optional[int] = None
    agent_limit: Optional[int] = None
    monthly_calls_limit: Optional[int] = None
    is_active: Optional[bool] = None
    stripe_price_id: Optional[str] = None

class PlanOut(PlanBase):
    id: uuid.UUID
    stripe_price_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
