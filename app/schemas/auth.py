from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from enum import Enum
import uuid

class LoginRequest(BaseModel):
    email: str
    password: str

class RoleInfo(BaseModel):
    id: uuid.UUID=Field(exclude=True) # exclude from response
    name: str
    description: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    email: str
    tenant_id: Optional[uuid.UUID] = None
    tenant_ids: Optional[List[uuid.UUID]] =  Field(default=None, exclude=True)
    role: Optional[RoleInfo] = None
    refresh_token: Optional[str] = None

class Provider(str, Enum):
    google = "google"  

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ForgotPasswordResponse(BaseModel):
    message: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class ResetPasswordResponse(BaseModel):
    message: str

class SwitchTenantRequest(BaseModel):
    tenant_id: uuid.UUID

class RefreshRequest(BaseModel):
    refresh_token: str
    access_token: Optional[str] = None

class GoogleLoginRequest(BaseModel):
    google_token: str
    provider: Provider = Provider.google

class RegisterRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: EmailStr
    password: Optional[str] = None
    phone: Optional[str] = None
    provider: Optional[str] = None
    google_token: Optional[str] = None
    provider: Optional[dict] = None    