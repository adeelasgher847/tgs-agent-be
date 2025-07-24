from pydantic import BaseModel
from typing import Optional

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    role_id: Optional[int] = None

class UserOut(UserBase):
    id: int
    role_id: Optional[int] = None

    class Config:
        orm_mode = True 