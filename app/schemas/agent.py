from pydantic import BaseModel, validator, root_validator
from typing import Optional
from datetime import datetime
import uuid
from enum import Enum

class LanguageEnum(str, Enum):
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"

class VoiceTypeEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"



class AgentBase(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = None

    @validator("name")
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Agent name must not be empty.")
        return v.strip()

    @validator("system_prompt")
    def prompt_length_limit(cls, v):
        if v is not None and len(v) > 1000:
            raise ValueError("System prompt must not exceed 1000 characters.")
        return v

    @validator("fallback_response")
    def fallback_length_limit(cls, v):
        if v is not None and len(v) > 1000:
            raise ValueError("Fallback response must not exceed 1000 characters.")
        return v

class AgentCreate(AgentBase):
    pass

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = None

    @validator("name")
    def name_must_not_be_empty(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Agent name must not be empty.")
        return v.strip() if v else v

    @validator("system_prompt")
    def prompt_length_limit(cls, v):
        if v is not None and len(v) > 1000:
            raise ValueError("System prompt must not exceed 1000 characters.")
        return v

    @validator("fallback_response")
    def fallback_length_limit(cls, v):
        if v is not None and len(v) > 1000:
            raise ValueError("Fallback response must not exceed 1000 characters.")
        return v

class AgentOut(AgentBase):
    id: uuid.UUID
    tenant_id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

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