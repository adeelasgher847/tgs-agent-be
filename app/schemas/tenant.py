from pydantic import BaseModel

class TenantBase(BaseModel):
    name: str
    schema_name: str

class TenantCreate(TenantBase):
    pass

class TenantOut(TenantBase):
    id: int

    class Config:
        orm_mode = True 