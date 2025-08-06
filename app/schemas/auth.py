from pydantic import BaseModel
from typing import Optional, List
import uuid

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    email: str
    tenant_id: Optional[uuid.UUID] = None
    tenant_ids: Optional[List[uuid.UUID]] = None

class TokenData(BaseModel):
    user_id: Optional[uuid.UUID] = None
    email: Optional[str] = None
    tenant_ids: List[uuid.UUID] = []
    tenant_id: Optional[uuid.UUID] = None

class SwitchTenantRequest(BaseModel):
    tenant_id: uuid.UUID 