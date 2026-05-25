from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import settings
from app.models.agent import Agent


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


class TtsProviderEnum(str, Enum):
    rime = "rime"
    elevenlabs = "11labs"
    elevenlabs_byo = "11labs_byo"


class AgentStatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    draft = "draft"
    pending = "pending"  # no phone number bound yet (telephony ticket)
    ready = "ready"  # bound to a number, callable


class TtsModelSchema(BaseModel):
    """Ticket ``ttsModel`` fragment."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: TtsProviderEnum
    voice_id: str = Field(..., min_length=1, max_length=255, alias="voiceId")
    language: str = Field(..., min_length=2, max_length=20)

    @field_validator("voice_id", "language")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class AgentCreate(BaseModel):
    """Create body — ticket fields required; voice-platform fields optional."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=3, max_length=80)
    llm_model: str = Field(..., min_length=1, max_length=100, alias="llmModel")
    tts_model: TtsModelSchema = Field(..., alias="ttsModel")
    status: AgentStatusEnum = AgentStatusEnum.active
    eleven_labs_api_key: Optional[str] = Field(
        default=None,
        alias="elevenLabsApiKey",
        min_length=1,
        max_length=500,
    )

    # Optional voice-runtime fields (dashboard / calls)
    system_prompt: Optional[str] = None
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = None
    model_id: Optional[uuid.UUID] = None
    provider_id: Optional[uuid.UUID] = None
    agent_temperature: Optional[int] = Field(None, ge=0, le=100)
    agent_max_tokens: Optional[int] = Field(None, gt=0)
    tts_provider_id: Optional[uuid.UUID] = None
    tts_voice_id: Optional[uuid.UUID] = None
    tts_settings_json: Optional[Dict[str, Any]] = None
    greeting_message: Optional[str] = None
    is_inbound_agent: bool = False
    is_follow_up_agent: bool = False
    transfer_route_id: Optional[uuid.UUID] = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not 3 <= len(normalized) <= 80:
            raise ValueError("name must be 3-80 characters after trimming whitespace")
        return normalized

    @model_validator(mode="after")
    def _byo_key_rules(self) -> "AgentCreate":
        is_byo = self.tts_model.provider == TtsProviderEnum.elevenlabs_byo
        if is_byo and not (self.eleven_labs_api_key and self.eleven_labs_api_key.strip()):
            raise ValueError("elevenLabsApiKey is required when ttsModel.provider is '11labs_byo'")
        if not is_byo and self.eleven_labs_api_key is not None:
            raise ValueError("elevenLabsApiKey is only allowed when ttsModel.provider is '11labs_byo'")
        return self


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Optional[str] = Field(None, min_length=3, max_length=80)
    llm_model: Optional[str] = Field(None, min_length=1, max_length=100, alias="llmModel")
    tts_model: Optional[TtsModelSchema] = Field(None, alias="ttsModel")
    status: Optional[AgentStatusEnum] = None
    eleven_labs_api_key: Optional[str] = Field(
        default=None,
        alias="elevenLabsApiKey",
        min_length=1,
        max_length=500,
    )

    system_prompt: Optional[str] = None
    language: Optional[LanguageEnum] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = None
    model_id: Optional[uuid.UUID] = None
    provider_id: Optional[uuid.UUID] = None
    agent_temperature: Optional[int] = Field(None, ge=0, le=100)
    agent_max_tokens: Optional[int] = Field(None, gt=0)
    tts_provider_id: Optional[uuid.UUID] = None
    tts_voice_id: Optional[uuid.UUID] = None
    tts_settings_json: Optional[Dict[str, Any]] = None
    greeting_message: Optional[str] = None
    is_inbound_agent: Optional[bool] = None
    is_follow_up_agent: Optional[bool] = None
    transfer_route_id: Optional[uuid.UUID] = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not 3 <= len(normalized) <= 80:
            raise ValueError("name must be 3-80 characters after trimming whitespace")
        return normalized

    @model_validator(mode="after")
    def _byo_key_rules(self) -> "AgentUpdate":
        if self.tts_model is not None:
            is_byo = self.tts_model.provider == TtsProviderEnum.elevenlabs_byo
            if is_byo and self.eleven_labs_api_key is not None and not self.eleven_labs_api_key.strip():
                raise ValueError("elevenLabsApiKey must be a non-empty string")
            if not is_byo and self.eleven_labs_api_key is not None:
                raise ValueError("elevenLabsApiKey is only allowed when ttsModel.provider is '11labs_byo'")
        return self


class AgentOut(BaseModel):
    """API agent projection — ticket fields + optional voice fields; never exposes BYO key."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    name: str
    llm_model: Optional[str] = Field(default=None, serialization_alias="llmModel")
    tts_model: Optional[TtsModelSchema] = Field(default=None, serialization_alias="ttsModel")
    status: AgentStatusEnum
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, serialization_alias="updatedAt")

    tenant_id: Optional[uuid.UUID] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    voice_type: Optional[str] = None
    model_id: Optional[uuid.UUID] = None
    tts_provider_id: Optional[uuid.UUID] = None
    tts_voice_id: Optional[uuid.UUID] = None
    is_inbound_agent: Optional[bool] = None
    is_follow_up_agent: Optional[bool] = None


def agent_to_out(agent: Agent) -> AgentOut:
    """Map ORM row to response; omit ``encrypted_elevenlabs_api_key``."""
    tts_model: Optional[TtsModelSchema] = None
    if agent.tts_provider_slug and agent.tts_voice_external_id and agent.tts_language:
        try:
            tts_model = TtsModelSchema(
                provider=TtsProviderEnum(agent.tts_provider_slug),
                voice_id=agent.tts_voice_external_id,
                language=agent.tts_language,
            )
        except ValueError:
            tts_model = None

    try:
        status = AgentStatusEnum(agent.status or "active")
    except ValueError:
        status = AgentStatusEnum.active

    return AgentOut(
        id=agent.id,
        name=agent.name,
        llm_model=agent.llm_model,
        tts_model=tts_model,
        status=status,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        tenant_id=agent.tenant_id,
        system_prompt=agent.system_prompt,
        language=agent.language,
        voice_type=agent.voice_type,
        model_id=agent.model_id,
        tts_provider_id=agent.tts_provider_id,
        tts_voice_id=agent.tts_voice_id,
        is_inbound_agent=agent.is_inbound_agent,
        is_follow_up_agent=agent.is_follow_up_agent,
    )


class AgentListResponse(BaseModel):
    data: list[AgentOut]
    total: int
    page: int
    page_size: int = Field(..., serialization_alias="pageSize")


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY

    def create_agent(self, name: str) -> str:
        if not self.api_key:
            return f"gemini_{name.lower().replace(' ', '_')}"
        return f"gemini_{name.lower().replace(' ', '_')}"


gemini_client = GeminiClient()
