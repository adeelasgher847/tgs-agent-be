from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.models.tts_provider import TTSProvider
from app.services.elevenlabs_service import elevenlabs_service
from app.services.google_tts_service import google_tts_service


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

    def _pop_elevenlabs_api_key(self, cfg: dict[str, Any]) -> Optional[str]:
        key = cfg.pop("elevenlabs_api_key", None) or cfg.pop("xi_api_key", None)
        return str(key).strip() if key else None

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        api_key_override = self._pop_elevenlabs_api_key(cfg)
        model_id = cfg.pop("model", "eleven_flash_v2_5")
        output_format = cfg.pop("output_format", "ulaw_8000")
        optimize_streaming_latency = int(cfg.pop("optimize_streaming_latency", 4))
        language_code = cfg.pop("language_code", None)
        previous_text = cfg.pop("previous_text", None)
        next_text = cfg.pop("next_text", None)
        previous_request_ids = cfg.pop("previous_request_ids", None)
        next_request_ids = cfg.pop("next_request_ids", None)
        apply_text_normalization = cfg.pop("apply_text_normalization", None)
        apply_language_text_normalization = cfg.pop("apply_language_text_normalization", None)
        cfg.pop("eleven_background", None)
        cfg.pop("eleven_background_level", None)
        return elevenlabs_service.text_to_speech(
            text=text,
            voice_id=voice_external_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=cfg if cfg else None,
            language_code=language_code,
            previous_text=previous_text,
            next_text=next_text,
            previous_request_ids=previous_request_ids,
            next_request_ids=next_request_ids,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
            optimize_streaming_latency=optimize_streaming_latency,
            api_key_override=api_key_override,
        )

    def stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ):
        cfg = dict(settings_json or {})
        api_key_override = self._pop_elevenlabs_api_key(cfg)
        model_id = cfg.pop("model", "eleven_flash_v2_5")
        output_format = cfg.pop("output_format", "ulaw_8000")
        optimize_streaming_latency = int(cfg.pop("optimize_streaming_latency", 4))
        language_code = cfg.pop("language_code", None)
        previous_text = cfg.pop("previous_text", None)
        next_text = cfg.pop("next_text", None)
        previous_request_ids = cfg.pop("previous_request_ids", None)
        next_request_ids = cfg.pop("next_request_ids", None)
        apply_text_normalization = cfg.pop("apply_text_normalization", None)
        apply_language_text_normalization = cfg.pop("apply_language_text_normalization", None)
        cfg.pop("eleven_background", None)
        cfg.pop("eleven_background_level", None)
        return elevenlabs_service.stream_text_to_speech(
            text=text,
            voice_id=voice_external_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=cfg if cfg else None,
            language_code=language_code,
            previous_text=previous_text,
            next_text=next_text,
            previous_request_ids=previous_request_ids,
            next_request_ids=next_request_ids,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
            optimize_streaming_latency=optimize_streaming_latency,
            api_key_override=api_key_override,
        )

    async def async_stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ):
        """
        True async streaming via httpx — no event-loop blocking.
        Used by _prefetch_tts_audio for the ElevenLabs hot path.
        """
        cfg = dict(settings_json or {})
        api_key_override = self._pop_elevenlabs_api_key(cfg)
        model_id = cfg.pop("model", "eleven_flash_v2_5")
        output_format = cfg.pop("output_format", "ulaw_8000")
        optimize_streaming_latency = int(cfg.pop("optimize_streaming_latency", 4))
        language_code = cfg.pop("language_code", None)
        previous_text = cfg.pop("previous_text", None)
        next_text = cfg.pop("next_text", None)
        previous_request_ids = cfg.pop("previous_request_ids", None)
        next_request_ids = cfg.pop("next_request_ids", None)
        apply_text_normalization = cfg.pop("apply_text_normalization", None)
        apply_language_text_normalization = cfg.pop("apply_language_text_normalization", None)
        cfg.pop("eleven_background", None)
        cfg.pop("eleven_background_level", None)
        async for chunk in elevenlabs_service.async_stream_text_to_speech(
            text=text,
            voice_id=voice_external_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=cfg if cfg else None,
            language_code=language_code,
            previous_text=previous_text,
            next_text=next_text,
            previous_request_ids=previous_request_ids,
            next_request_ids=next_request_ids,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
            optimize_streaming_latency=optimize_streaming_latency,
            api_key_override=api_key_override,
        ):
            yield chunk


class GoogleTTSAdapter(BaseTTSProviderAdapter):
    def list_voices(self) -> list[dict[str, Any]]:
        return google_tts_service.list_supported_voices()

    def normalize_voice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "external_voice_id": payload.get("voice_name"),
            "display_name": payload.get("display_name") or payload.get("voice_name") or "Google Voice",
            "language_code": payload.get("language_code"),
            "gender": payload.get("gender"),
            "accent": payload.get("accent"),
            "description": "Google Cloud Text-to-Speech voice",
            "preview_audio_url": None,
            "sample_rate_hz": 8000,
            "metadata_json": payload,
        }

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        language = cfg.pop("language", "en")
        voice_type = cfg.pop("voice_type", "female")
        speaking_rate = float(cfg.pop("speaking_rate", 1.0))
        pitch = float(cfg.pop("pitch", 0.0))
        output_format = cfg.pop("output_format", "mulaw")
        use_chirp3_hd = bool(cfg.pop("use_chirp3_hd", True))
        return google_tts_service.text_to_speech(
            text=text,
            language=language,
            voice_type=voice_type,
            speaking_rate=speaking_rate,
            pitch=pitch,
            output_format=output_format,
            use_chirp3_hd=use_chirp3_hd,
            voice_name_override=voice_external_id,
        )

    def stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: Optional[dict[str, Any]] = None,
    ):
        cfg = dict(settings_json or {})
        language = cfg.pop("language", "en")
        voice_type = cfg.pop("voice_type", "female")
        speaking_rate = float(cfg.pop("speaking_rate", 1.0))
        output_format = cfg.pop("output_format", "mulaw")
        use_chirp3_hd = bool(cfg.pop("use_chirp3_hd", True))
        sample_rate_hz = int(cfg.pop("sample_rate_hz", 8000))
        return google_tts_service.stream_text_to_speech(
            text=text,
            language=language,
            voice_type=voice_type,
            speaking_rate=speaking_rate,
            output_format=output_format,
            use_chirp3_hd=use_chirp3_hd,
            sample_rate_hz=sample_rate_hz,
            voice_name_override=voice_external_id,
        )


def get_tts_adapter(provider_slug: str) -> BaseTTSProviderAdapter:
    slug = (provider_slug or "").strip().lower()
    if slug == "elevenlabs":
        return ElevenLabsAdapter()
    if slug == "google":
        return GoogleTTSAdapter()
    raise ValueError(f"Unsupported TTS provider: {provider_slug}")


def get_tts_adapter_for_provider(provider: TTSProvider) -> BaseTTSProviderAdapter:
    if not provider:
        raise ValueError("TTS provider is required.")
    return get_tts_adapter(provider.slug)
