from pydantic import BaseModel, EmailStr

class RegisterSchema(BaseModel):
    email: EmailStr
    password: str
    tenant_id: int

class Token(BaseModel):
    access_token: str
    token_type: str
