from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
import uuid

class UsageRecordBase(BaseModel):
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2020)
    calls_used: int = Field(default=0, ge=0)
    agents_created: int = Field(default=0, ge=0)

class UsageRecordCreate(UsageRecordBase):
    subscription_id: uuid.UUID

class UsageRecordUpdate(BaseModel):
    calls_used: Optional[int] = None
    agents_created: Optional[int] = None

class UsageRecordOut(UsageRecordBase):
    id: uuid.UUID
    subscription_id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
