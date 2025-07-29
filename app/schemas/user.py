from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from pydantic import EmailStr
from datetime import datetime


class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None


class UserCreate(UserBase):
    password: str
    # role_id: Optional[int] = None

class UserOut(UserBase):
    id: int
    role_id: Optional[int] = None
    join_date: datetime
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True) 


class UserLogin(BaseModel):
    email: str
    password: str