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
    tenant_ids: List[int]
    current_tenant_id: Optional[int] = None

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    tenant_ids: List[int] = []
    current_tenant_id: Optional[int] = None

class SwitchTenantRequest(BaseModel):
    tenant_id: int 