from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None

class UserCreate(UserBase):
    password: str
    tenant_id: Optional[int] = None

class UserOut(UserBase):
    id: int
    tenant_id: Optional[int] = None
    join_date: datetime     
    created_at: datetime

    class Config:
        orm_mode = True 