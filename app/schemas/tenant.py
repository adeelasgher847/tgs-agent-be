from pydantic import BaseModel
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

    class Config:
        from_attributes = True  # Updated from orm_mode for newer Pydantic versions

class TenantCreateResponse(BaseModel):
    tenant_id: int
    message: str
    tenant: TenantOut