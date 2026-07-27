from datetime import datetime
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field


class TTSProviderOut(BaseModel):
    id: uuid.UUID
    slug: str
    display_name: str
    is_active: bool
    supports_streaming: bool
    supports_ssml: bool
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class TTSVoiceOut(BaseModel):
    id: uuid.UUID
    provider_id: uuid.UUID
    external_voice_id: str
    display_name: str
    language_code: str | None = None
    gender: str | None = None
    accent: str | None = None
    description: str | None = None
    preview_audio_url: str | None = None
    sample_rate_hz: int | None = None
    is_active: bool
    metadata_json: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class TTSProviderListOut(BaseModel):
    providers: list[TTSProviderOut]


class TTSVoiceListOut(BaseModel):
    voices: list[TTSVoiceOut]



class TTSProviderCreate(BaseModel):
    slug: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    is_active: bool = True
    supports_streaming: bool = False
    supports_ssml: bool = True


class TTSVoiceCreate(BaseModel):
    provider_id: uuid.UUID
    external_voice_id: str = Field(..., min_length=1, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=255)
    language_code: str | None = Field(None, max_length=20)
    gender: str | None = Field(None, max_length=32)
    accent: str | None = Field(None, max_length=64)
    description: str | None = None
    preview_audio_url: str | None = Field(None, max_length=1000)
    sample_rate_hz: int | None = None
    is_active: bool = True
    metadata_json: dict[str, Any] | None = None
