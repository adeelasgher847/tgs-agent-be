from pydantic import BaseModel, ConfigDict
from typing import Optional

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    role_id: Optional[int] = None

class UserOut(UserBase):
    id: int
    role_id: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True) 