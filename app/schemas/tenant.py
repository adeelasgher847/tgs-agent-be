from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TenantBase(BaseModel):
    name: str

class TenantCreate(TenantBase):
    # Only name required, schema_name and admin_id will be set automatically
    pass

class TenantOut(TenantBase):
    id: int
    schema_name: str
    admin_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class TenantCreateResponse(BaseModel):
    tenant_id: int
    message: str
    tenant: TenantOut