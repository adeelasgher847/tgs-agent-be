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
    tenant_id: Optional[int] = None
    tenant_ids: Optional[List[int]] = None  # Fix type

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    tenant_ids: List[int] = []  # Keep as is for now
    tenant_id: Optional[int] = None

class SwitchTenantRequest(BaseModel):
    tenant_id: int 