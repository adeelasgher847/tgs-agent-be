from pydantic import BaseModel
from typing import Optional, List

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    tenant_id: Optional[int] = None  # Changed from current_tenant_id

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    tenant_ids: List[int] = []  # Will be populated from database
    tenant_id: Optional[int] = None  # Changed from current_tenant_id

class SwitchTenantRequest(BaseModel):
    tenant_id: int 