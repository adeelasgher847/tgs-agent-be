from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class TenantBase(BaseModel):
    name: str

class TenantCreate(TenantBase):
    # Only name required, schema_name will be set automatically
    pass

class TenantOut(TenantBase):
    id: int
    schema_name: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class TenantCreateResponse(BaseModel):
    tenant_id: int
    message: str
    tenant: TenantOut