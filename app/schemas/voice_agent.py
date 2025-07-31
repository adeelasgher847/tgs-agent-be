from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class VoiceAgentBase(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    voice_type: Optional[str] = None
    fallback_response: Optional[str] = None


class VoiceAgentCreate(VoiceAgentBase):
    tenant_id: int


class VoiceAgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    voice_type: Optional[str] = None
    fallback_response: Optional[str] = None


class VoiceAgentOut(VoiceAgentBase):
    id: int
    tenant_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
