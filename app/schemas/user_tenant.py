from pydantic import BaseModel, ConfigDict
import uuid

class UserTenantAssociationBase(BaseModel):
    role_id: uuid.UUID | None = None

class UserTenantAssociationCreate(UserTenantAssociationBase):
    user_id: uuid.UUID
    tenant_id: uuid.UUID

class UserTenantAssociationOut(UserTenantAssociationBase):
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    
    model_config = ConfigDict(from_attributes=True)
