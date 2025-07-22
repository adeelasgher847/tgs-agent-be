from pydantic import BaseModel

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    tenant_id: int

class UserOut(UserBase):
    id: int
    tenant_id: int

    class Config:
        orm_mode = True 