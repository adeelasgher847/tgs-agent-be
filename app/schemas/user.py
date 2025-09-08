from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from pydantic import EmailStr
from datetime import datetime
import uuid


class UserBase(BaseModel):
    first_name: str = Field(..., min_length=1, description="First name is required")
    last_name: str = Field(..., min_length=1, description="Last name is required")
    email: EmailStr = Field(..., description="Valid email address is required")
    phone: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters long")
    # role_id: Optional[int] = None


class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, description="First name")
    last_name: Optional[str] = Field(None, min_length=1, description="Last name")
    email: Optional[EmailStr] = Field(None, description="Valid email address")
    phone: Optional[str] = Field(None, description="Phone number")

class UserOut(UserBase):
    id: uuid.UUID
    role_id: Optional[uuid.UUID] = None
    join_date: datetime
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class RoleInfo(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class TenantInfo(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserProfile(UserBase):
    id: uuid.UUID
    role_id: Optional[uuid.UUID] = None
    current_tenant_id: Optional[uuid.UUID] = None
    join_date: datetime
    created_at: datetime
    role: Optional[RoleInfo] = None
    current_tenant: Optional[TenantInfo] = None
    tenants: list[TenantInfo] = []
    
    model_config = ConfigDict(from_attributes=True) 

