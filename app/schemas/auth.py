from pydantic import BaseModel, EmailStr
from typing import Optional, List
import uuid

class LoginRequest(BaseModel):
    email: str
    password: str

class RoleInfo(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    email: str
    tenant_id: Optional[uuid.UUID] = None
    tenant_ids: Optional[List[uuid.UUID]] = None
    role: Optional[RoleInfo] = None
    refresh_token: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ForgotPasswordResponse(BaseModel):
    message: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class ResetPasswordResponse(BaseModel):
    message: str

class TokenData(BaseModel):
    user_id: Optional[uuid.UUID] = None
    email: Optional[str] = None
    tenant_ids: List[uuid.UUID] = []
    tenant_id: Optional[uuid.UUID] = None

class SwitchTenantRequest(BaseModel):
    tenant_id: uuid.UUID

class RefreshRequest(BaseModel):
    refresh_token: str
    access_token: Optional[str] = None