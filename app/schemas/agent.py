from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
from enum import Enum

class LanguageEnum(str, Enum):
    en = "en"
    ur = "ur"
    es = "es"
    hi = "hi"
    ar = "ar"
    zh = "zh"

class VoiceTypeEnum(str, Enum):
    male = "male"
    female = "female"
    
class AgentBase(BaseModel):
    name: str =  Field(..., min_length=1, max_length=100)
    system_prompt: Optional[str] = Field(None, max_length=1000)
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = Field(None, max_length=1000)


class AgentCreate(AgentBase):
    # tenant_id is automatically added from current tenant context
    pass


class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    system_prompt: Optional[str] = Field(None, max_length=1000)
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = Field(None, max_length=1000)

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