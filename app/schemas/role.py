from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None

class RoleCreate(RoleBase):
    pass

class RoleOut(RoleBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True 