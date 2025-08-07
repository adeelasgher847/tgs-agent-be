from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid


class AgentBase(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    voice_type: Optional[str] = None
    fallback_response: Optional[str] = None


class AgentCreate(AgentBase):
    # tenant_id is automatically added from current tenant context
    pass


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    voice_type: Optional[str] = None
    fallback_response: Optional[str] = None


class AgentOut(AgentBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: uuid.UUID
    updated_by: uuid.UUID

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    data: list[AgentOut]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool
