from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.models.tts_provider import TTSProvider
from app.services.elevenlabs_service import elevenlabs_service


class BaseTTSProviderAdapter(ABC):
    @abstractmethod
    def list_voices(self) -> list[dict[str, Any]]:
        """Return raw provider voice payload list."""

    @abstractmethod
    def normalize_voice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize provider payload into local catalog shape."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ) -> bytes:
        """Synthesize speech and return telephony-ready bytes."""

    def stream_synthesize(self, *args, **kwargs):
        raise NotImplementedError("Streaming is not implemented for this provider")


class ElevenLabsAdapter(BaseTTSProviderAdapter):
    def list_voices(self) -> list[dict[str, Any]]:
        payload = elevenlabs_service.get_available_voices() or {}
        return payload.get("voices", [])

    def normalize_voice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        labels = payload.get("labels") or {}
        return {
            "external_voice_id": payload.get("voice_id"),
            "display_name": payload.get("name") or "Unnamed Voice",
            "language_code": labels.get("language"),
            "gender": labels.get("gender"),
            "accent": labels.get("accent"),
            "description": payload.get("description"),
            "preview_audio_url": payload.get("preview_url"),
            "sample_rate_hz": None,
            "metadata_json": payload,
        }

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        model_id = cfg.pop("model", "eleven_flash_v2_5")
        output_format = cfg.pop("output_format", "ulaw_8000")
        optimize_streaming_latency = int(cfg.pop("optimize_streaming_latency", 4))
        return elevenlabs_service.text_to_speech(
            text=text,
            voice_id=voice_external_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=cfg if cfg else None,
            optimize_streaming_latency=optimize_streaming_latency,
        )

    def stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ):
        cfg = dict(settings_json or {})
        model_id = cfg.pop("model", "eleven_flash_v2_5")
        output_format = cfg.pop("output_format", "ulaw_8000")
        optimize_streaming_latency = int(cfg.pop("optimize_streaming_latency", 4))
        return elevenlabs_service.stream_text_to_speech(
            text=text,
            voice_id=voice_external_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=cfg if cfg else None,
            optimize_streaming_latency=optimize_streaming_latency,
        )


def get_tts_adapter(provider_slug: str) -> BaseTTSProviderAdapter:
    slug = (provider_slug or "").strip().lower()
    if slug == "elevenlabs":
        return ElevenLabsAdapter()
    raise ValueError(f"Unsupported TTS provider: {provider_slug}")


def get_tts_adapter_for_provider(provider: TTSProvider) -> BaseTTSProviderAdapter:
    if not provider:
        raise ValueError("TTS provider is required.")
    return get_tts_adapter(provider.slug)
