from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import settings
from app.models.agent import Agent

# Legacy ticket slugs stored before provider enum rename.
_TTS_SLUG_ALIASES: dict[str, str] = {
    "11labs": "elevenlabs",
    "11labs_byo": "elevenlabs_byo",
}

_DEFAULT_STT_PROVIDER = "deepgram"
_DEFAULT_STT_MODEL_ID = "nova-3"
_DEFAULT_STT_LANGUAGE_CODE = "en"


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
    elevenlabs = "elevenlabs"
    elevenlabs_byo = "elevenlabs_byo"


class AgentStatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    draft = "draft"
    pending = "pending"  # no phone number bound yet (telephony ticket)
    ready = "ready"  # bound to a number, callable
    error = "error"  # provisioning failure (v2)


def normalize_tts_provider_slug(raw: str) -> str:
    """Map API / legacy slugs to canonical DB slug."""
    key = (raw or "").strip().lower()
    return _TTS_SLUG_ALIASES.get(key, key)


def tts_slug_to_api_provider(slug: str) -> TtsProviderEnum:
    """Map stored slug to API enum (supports legacy rows)."""
    canonical = normalize_tts_provider_slug(slug)
    return TtsProviderEnum(canonical)


class SttProviderEnum(str, Enum):
    deepgram = "deepgram"
    google = "google"
    elevenlabs = "elevenlabs"


class SttModelSchema(BaseModel):
    """Ticket ``sttModel`` fragment — validated against STT catalog in service layer."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: SttProviderEnum
    model_id: str = Field(..., min_length=1, max_length=255, alias="modelId")
    language_code: str = Field(..., min_length=2, max_length=20, alias="languageCode")

    @field_validator("model_id")
    @classmethod
    def _strip_model_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("language_code")
    @classmethod
    def _strip_language_code(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class SttSettingsJsonSchema(BaseModel):
    """Optional STT tuning stored on ``agent.stt_settings_json``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    silence_threshold_ms: Optional[int] = Field(
        default=1500,
        ge=300,
        le=5000,
        alias="silenceThresholdMs",
        description="Silence duration (ms) before isSilence=true is emitted.",
    )


class TtsSettingsJsonSchema(BaseModel):
    """
    Optional TTS voice tuning stored on ``agent.tts_settings_json``.

    Omit the whole object at create/update to keep normal call-time defaults
    (speed 1.0, volume 1.0). Shown in Swagger as the default example shape.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    speed: float = Field(
        default=1.0,
        ge=settings.TTS_SPEED_MIN,
        le=settings.TTS_SPEED_MAX,
        description="Speech rate. 1.0 = normal, lower = slower, higher = faster.",
    )
    volume: float = Field(
        default=1.0,
        ge=settings.TTS_VOLUME_MIN,
        le=settings.TTS_VOLUME_MAX,
        description="Output loudness. 1.0 = normal, 0 = silence, up to max = louder.",
    )


class TtsModelSchema(BaseModel):
    """Ticket ``ttsModel`` fragment — validated against TTS catalog in service layer."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: TtsProviderEnum
    voice_id: str = Field(..., min_length=1, max_length=255, alias="voiceId")
    language: LanguageEnum
    tts_voice_id: Optional[uuid.UUID] = Field(None, alias="ttsVoiceId")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        if isinstance(value, str):
            canonical = normalize_tts_provider_slug(value)
            return TtsProviderEnum(canonical).value
        return value

    @field_validator("voice_id")
    @classmethod
    def _strip_voice_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class AgentCreate(BaseModel):
    """Create body — ``llmModel`` + ``ttsModel`` required; no catalog UUIDs in API."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=3, max_length=80)
    llm_model: str = Field(..., min_length=1, max_length=100, alias="llmModel")
    tts_model: TtsModelSchema = Field(..., alias="ttsModel")
    stt_model: Optional[SttModelSchema] = Field(
        default=None,
        alias="sttModel",
        description=(
            "STT provider + model. Defaults to deepgram/nova-3/en when omitted. "
            "Use provider='google', modelId='chirp-3', languageCode='en-AU' for Google STT."
        ),
    )
    stt_settings: Optional[SttSettingsJsonSchema] = Field(
        default=None,
        alias="sttSettings",
        description="Optional STT tuning (e.g. silenceThresholdMs).",
    )
    status: AgentStatusEnum = AgentStatusEnum.pending
    eleven_labs_api_key: Optional[str] = Field(
        default=None,
        alias="elevenLabsApiKey",
        min_length=1,
        max_length=500,
    )

    # Optional voice-runtime fields (dashboard / calls)
    system_prompt: Optional[str] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = None
    agent_temperature: Optional[int] = Field(None, ge=0, le=100)
    agent_max_tokens: Optional[int] = Field(None, gt=0)
    tts_settings_json: Optional[TtsSettingsJsonSchema] = Field(
        default=None,
        description=(
            "Optional TTS tuning. Example uses normal speed/volume (1.0). "
            "Omit entirely to use the same defaults at call time."
        ),
        json_schema_extra={"example": {"speed": 1.0, "volume": 1.0}},
    )
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

    @field_validator("llm_model")
    @classmethod
    def _strip_llm_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("llmModel must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _byo_key_rules(self) -> "AgentCreate":
        is_byo = self.tts_model.provider == TtsProviderEnum.elevenlabs_byo
        if is_byo and not (self.eleven_labs_api_key and self.eleven_labs_api_key.strip()):
            raise ValueError(
                "elevenLabsApiKey is required when ttsModel.provider is 'elevenlabs_byo'"
            )
        if not is_byo and self.eleven_labs_api_key is not None:
            raise ValueError(
                "elevenLabsApiKey is only allowed when ttsModel.provider is 'elevenlabs_byo'"
            )
        return self


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Optional[str] = Field(None, min_length=3, max_length=80)
    llm_model: Optional[str] = Field(None, min_length=1, max_length=100, alias="llmModel")
    tts_model: Optional[TtsModelSchema] = Field(None, alias="ttsModel")
    stt_model: Optional[SttModelSchema] = Field(None, alias="sttModel")
    stt_settings: Optional[SttSettingsJsonSchema] = Field(None, alias="sttSettings")
    status: Optional[AgentStatusEnum] = None
    eleven_labs_api_key: Optional[str] = Field(
        default=None,
        alias="elevenLabsApiKey",
        min_length=1,
        max_length=500,
    )

    system_prompt: Optional[str] = None
    voice_type: Optional[VoiceTypeEnum] = None
    fallback_response: Optional[str] = None
    agent_temperature: Optional[int] = Field(None, ge=0, le=100)
    agent_max_tokens: Optional[int] = Field(None, gt=0)
    tts_settings_json: Optional[TtsSettingsJsonSchema] = Field(
        default=None,
        description=(
            "Optional TTS tuning. Example uses normal speed/volume (1.0). "
            "Omit entirely to use the same defaults at call time."
        ),
        json_schema_extra={"example": {"speed": 1.0, "volume": 1.0}},
    )
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

    @field_validator("llm_model")
    @classmethod
    def _strip_llm_model(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("llmModel must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _byo_key_rules(self) -> "AgentUpdate":
        if self.tts_model is not None:
            is_byo = self.tts_model.provider == TtsProviderEnum.elevenlabs_byo
            if is_byo and self.eleven_labs_api_key is not None and not self.eleven_labs_api_key.strip():
                raise ValueError("elevenLabsApiKey must be a non-empty string")
            if not is_byo and self.eleven_labs_api_key is not None:
                raise ValueError(
                    "elevenLabsApiKey is only allowed when ttsModel.provider is 'elevenlabs_byo'"
                )
        return self


class AgentOut(BaseModel):
    """API agent projection — ticket fields only; never exposes BYO key or catalog UUIDs."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    name: str
    llm_model: Optional[str] = Field(default=None, serialization_alias="llmModel")
    tts_model: Optional[TtsModelSchema] = Field(default=None, serialization_alias="ttsModel")
    stt_model: Optional[SttModelSchema] = Field(default=None, serialization_alias="sttModel")
    status: AgentStatusEnum
    created_at: datetime = Field(..., serialization_alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, serialization_alias="updatedAt")

    tenant_id: Optional[uuid.UUID] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    voice_type: Optional[str] = None
    is_inbound_agent: Optional[bool] = None
    is_follow_up_agent: Optional[bool] = None


def agent_to_out(agent: Agent) -> AgentOut:
    """Map ORM row to response; omit ``encrypted_elevenlabs_api_key`` and catalog UUIDs."""
    tts_model: Optional[TtsModelSchema] = None
    if agent.tts_provider_slug and agent.tts_voice_external_id and agent.tts_language:
        try:
            tts_model = TtsModelSchema(
                provider=tts_slug_to_api_provider(agent.tts_provider_slug),
                voice_id=agent.tts_voice_external_id,
                language=LanguageEnum(agent.tts_language),
                tts_voice_id=agent.tts_voice_id,
            )
        except ValueError:
            tts_model = None

    stt_model: Optional[SttModelSchema] = None
    if agent.stt_provider_slug and agent.stt_model_external_id and agent.stt_language_code:
        try:
            stt_model = SttModelSchema(
                provider=SttProviderEnum(agent.stt_provider_slug),
                model_id=agent.stt_model_external_id,
                language_code=agent.stt_language_code,
            )
        except ValueError:
            stt_model = None

    try:
        status = AgentStatusEnum(agent.status or "pending")
    except ValueError:
        status = AgentStatusEnum.pending

    return AgentOut(
        id=agent.id,
        name=agent.name,
        llm_model=agent.llm_model,
        tts_model=tts_model,
        stt_model=stt_model,
        status=status,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        tenant_id=agent.tenant_id,
        system_prompt=agent.system_prompt,
        language=agent.language,
        voice_type=agent.voice_type,
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
