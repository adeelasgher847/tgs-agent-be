from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import EmailStr
from datetime import datetime
import uuid


class UserBase(BaseModel):
    first_name: str = Field(..., min_length=1, description="First name is required")
    last_name: str = Field(..., min_length=1, description="Last name is required")
    email: EmailStr = Field(..., description="Valid email address is required")
    phone: str | None = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters long")
    # role_id: Optional[int] = None

    @model_validator(mode='before')
    def trim_user_names(cls, values):
        if 'first_name' in values and values['first_name']:
            values['first_name'] = " ".join(values['first_name'].split())
        if 'last_name' in values and values['last_name']:
            values['last_name'] = " ".join(values['last_name'].split())
        return values


class UserUpdate(BaseModel):
    first_name: str | None = Field(None, min_length=1, description="First name")
    last_name: str | None = Field(None, min_length=1, description="Last name")
    email: EmailStr | None = Field(None, description="Valid email address")
    phone: str | None = Field(None, description="Phone number")

class UserOut(UserBase):
    id: uuid.UUID
    role_id: uuid.UUID | None = None
    join_date: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AcceptInviteOut(UserOut):
    """UserOut plus the session tokens issued on invite acceptance."""
    access_token: str
    refresh_token: str


class RoleInfo(BaseModel):
    id: uuid.UUID = Field(exclude=True)
    name: str
    description: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class TenantInfo(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    
    model_config = ConfigDict(from_attributes=True)


class UserProfile(UserBase):
    id: uuid.UUID
    role_id: uuid.UUID | None = None
    current_tenant_id: uuid.UUID | None = None
    join_date: datetime
    created_at: datetime
    role: RoleInfo | None = None
    # role: Optional[RoleInfo]  = None
    current_tenant: TenantInfo | None = None
    tenants: list[TenantInfo] = []
    
    model_config = ConfigDict(from_attributes=True) 


class TenantMember(BaseModel):
    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    role: RoleInfo | None = None
    join_date: datetime
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)