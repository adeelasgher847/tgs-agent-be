from pydantic import BaseModel, EmailStr, Field
from typing import List
from enum import Enum
import uuid

class LoginRequest(BaseModel):
    email: str
    password: str

class RoleInfo(BaseModel):
    id: uuid.UUID=Field(exclude=True) # exclude from response
    name: str
    description: str | None = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    email: str
    tenant_id: uuid.UUID | None = None
    product_id: uuid.UUID | None = None
    tenant_ids: List[uuid.UUID] | None =  Field(default=None, exclude=True)
    role: RoleInfo | None = None
    refresh_token: str | None = None

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
    access_token: str | None = None

class GoogleLoginRequest(BaseModel):
    google_token: str
    provider: Provider = Provider.google

class RegisterRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: EmailStr
    password: str | None = None
    phone: str | None = None
    provider: str | None = None
    google_token: str | None = None
    provider: dict | None = None    