from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
import uuid

class TenantBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class TenantCreate(TenantBase):
    # Only name required, schema_name will be set automatically
    pass

class TenantOut(TenantBase):
    id: uuid.UUID
    schema_name: str
    status: str
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class TenantCreateResponse(BaseModel):
    tenant_id: uuid.UUID
    tenant: TenantOut