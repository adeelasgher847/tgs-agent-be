from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
from enum import Enum
from app.core.config import settings

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
    model_id: Optional[uuid.UUID] = None
    pass


class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    system_prompt: Optional[str] = Field(None, max_length=1000)
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = Field(None, max_length=1000)
    model_id: Optional[uuid.UUID] = None

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
    
    
class GeminiClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY

    def create_agent(self, name: str) -> str:
        # TODO: Replace with actual Gemini API endpoint payload and headers
        if not self.api_key:
            # For now, just simulate and return a fake id if key missing; or raise
            return f"gemini_{name.lower().replace(' ', '_')}"
        # Example stub:
        # url = "https://generativelanguage.googleapis.com/v1/agents"
        # headers = {"Authorization": f"Bearer {self.api_key}"}
        # payload = {"displayName": name}
        # r = httpx.post(url, json=payload, headers=headers, timeout=15)
        # r.raise_for_status()
        # return r.json().get("name")  # or appropriate id field
        return f"gemini_{name.lower().replace(' ', '_')}"

gemini_client = GeminiClient()