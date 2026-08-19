from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Coroutine

from app.core.config import settings
from app.core.secret_manager import get_hume_api_key, get_rime_api_key
from app.models.tts_provider import TTSProvider
from app.services.elevenlabs_service import elevenlabs_service
from app.services.google_tts_service import google_tts_service


class _RimeSyncEventLoopBridge:
    """Runs Rime's async HTTP calls on a single, persistent background event
    loop so synchronous callers (RimeTTSAdapter.synthesize(), invoked from a
    worker thread via run_in_executor()) can drive them without spinning up
    and tearing down a fresh event loop on every call.

    Why this exists: rime_tts_service caches an httpx.AsyncClient for
    connection pooling, and an httpx.AsyncClient's connection pool is bound
    to whichever event loop was running when its sockets were opened. Doing
    `asyncio.run(coro)` per call — one fresh loop created and destroyed each
    time — meant a second call could reuse a pooled connection still bound
    to the *first* call's already-closed loop, raising
    "RuntimeError: Event loop is closed" deep inside httpx/httpcore/anyio's
    connection teardown.

    A dedicated loop, started once and reused for the process's lifetime,
    keeps the cached client bound to exactly one stable loop for its whole
    life — matching httpx's actual threading/loop contract. This is chosen
    over "don't cache the client" because Rime is on the hot TTS-latency
    path and repeated TLS handshakes per call would add real latency; a
    single background thread is a small, one-time cost.

    This bridge is used *only* for the sync-bridge path
    (RimeTTSAdapter.synthesize()). The live streaming path
    (async_stream_synthesize / stream_synthesize, called from
    app/voice/tts_stream_mixin.py) is already correctly awaited directly on
    the caller's own real running loop and must keep working exactly as
    before — it does not go through this bridge at all.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is not None:
            return loop
        with self._start_lock:
            if self._loop is not None:
                return self._loop

            ready = threading.Event()
            state: dict[str, asyncio.AbstractEventLoop] = {}

            def _run_forever() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                state["loop"] = loop
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=_run_forever,
                name="rime-tts-sync-bridge",
                daemon=True,
            )
            thread.start()
            ready.wait()
            self._loop = state["loop"]
            return self._loop

    def run(self, coro: "Coroutine[Any, Any, bytes]") -> bytes:
        """Schedule `coro` on the dedicated bridge loop and block the
        calling thread until it completes. Safe to call from any thread,
        regardless of whether that thread has its own running event loop."""
        loop = self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


_rime_sync_bridge = _RimeSyncEventLoopBridge()


class _HumeSyncEventLoopBridge:
    """Same rationale/pattern as `_RimeSyncEventLoopBridge` above, applied to
    Hume's `websockets`-based streaming client instead of Rime's cached
    httpx.AsyncClient — a websocket connection opened on one event loop
    cannot be safely driven from a different loop either, so
    HumeTTSAdapter.synthesize() (the sync-bridge path, invoked from a worker
    thread via run_in_executor()) needs its own dedicated, persistent
    background loop, kept entirely separate from Rime's bridge instance.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is not None:
            return loop
        with self._start_lock:
            if self._loop is not None:
                return self._loop

            ready = threading.Event()
            state: dict[str, asyncio.AbstractEventLoop] = {}

            def _run_forever() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                state["loop"] = loop
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=_run_forever,
                name="hume-tts-sync-bridge",
                daemon=True,
            )
            thread.start()
            ready.wait()
            self._loop = state["loop"]
            return self._loop

    def run(self, coro: "Coroutine[Any, Any, bytes]") -> bytes:
        """Schedule `coro` on the dedicated bridge loop and block the
        calling thread until it completes. Safe to call from any thread,
        regardless of whether that thread has its own running event loop."""
        loop = self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


_hume_sync_bridge = _HumeSyncEventLoopBridge()


class _XaiSyncEventLoopBridge:
    """Same rationale/pattern as `_RimeSyncEventLoopBridge`/`_HumeSyncEventLoopBridge`
    above, applied to xAI's `websockets`-based streaming client — a websocket
    connection opened on one event loop cannot be safely driven from a
    different loop, so XaiTTSAdapter.synthesize() (the sync-bridge path,
    invoked from a worker thread via run_in_executor()) needs its own
    dedicated, persistent background loop, kept entirely separate from
    Rime's/Hume's bridge instances.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is not None:
            return loop
        with self._start_lock:
            if self._loop is not None:
                return self._loop

            ready = threading.Event()
            state: dict[str, asyncio.AbstractEventLoop] = {}

            def _run_forever() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                state["loop"] = loop
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=_run_forever,
                name="xai-tts-sync-bridge",
                daemon=True,
            )
            thread.start()
            ready.wait()
            self._loop = state["loop"]
            return self._loop

    def run(self, coro: "Coroutine[Any, Any, bytes]") -> bytes:
        """Schedule `coro` on the dedicated bridge loop and block the
        calling thread until it completes. Safe to call from any thread,
        regardless of whether that thread has its own running event loop."""
        loop = self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


_xai_sync_bridge = _XaiSyncEventLoopBridge()


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
        settings_json: dict[str, Any] | None = None,
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

    def _pop_elevenlabs_api_key(self, cfg: dict[str, Any]) -> str | None:
        key = cfg.pop("elevenlabs_api_key", None) or cfg.pop("xi_api_key", None)
        return str(key).strip() if key else None

    @staticmethod
    def _strip_agent_runtime_keys(cfg: dict[str, Any]) -> None:
        """Remove keys that are owned by tts_stream_mixin (post-process / nesting),
        not the ElevenLabs API. Keeps `speed` so it lands in voice_settings."""
        cfg.pop("volume", None)        # applied as mulaw gain at playback
        cfg.pop("settings", None)      # nested form already merged upstream
        cfg.pop("background_enabled", None)
        cfg.pop("background_profile", None)
        cfg.pop("background_volume", None)

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
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
        self._strip_agent_runtime_keys(cfg)
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
        settings_json: dict[str, Any] | None = None,
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
        self._strip_agent_runtime_keys(cfg)
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
        settings_json: dict[str, Any] | None = None,
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
        self._strip_agent_runtime_keys(cfg)
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

    @staticmethod
    def _resolve_speaking_rate(cfg: dict[str, Any]) -> float:
        """User-facing `speed` (1.0 = normal) maps directly to Google
        `speaking_rate`. Explicit `speaking_rate` wins if both are set."""
        if "speaking_rate" in cfg:
            return float(cfg.pop("speaking_rate"))
        if "speed" in cfg:
            # Clamp to Google's [0.25, 2.0] range to avoid INVALID_ARGUMENT.
            return max(0.25, min(2.0, float(cfg.get("speed", 1.0))))
        return 1.0

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        language = cfg.pop("language", "en")
        voice_type = cfg.pop("voice_type", "female")
        speaking_rate = self._resolve_speaking_rate(cfg)
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
        settings_json: dict[str, Any] | None = None,
    ):
        cfg = dict(settings_json or {})
        language = cfg.pop("language", "en")
        voice_type = cfg.pop("voice_type", "female")
        speaking_rate = self._resolve_speaking_rate(cfg)
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


class RimeTTSAdapter(BaseTTSProviderAdapter):
    """
    Adapter for Rime Labs TTS (mistv2 model).

    Telephony output: mulaw 8 kHz — matches Twilio MULAW 8000 directly.
    Default voice: mistv2_Wildflower (configurable via voiceId per agent).

    settings_json keys (all optional):
        speed    (float, default 1.0) — mapped to Rime speedAlpha
        volume   (float, default 1.0) — post-processing gain (0.0–2.0)
        model_id (str,   default "mistv2")
    """

    _DEFAULT_VOICE = "mistv2_Wildflower"
    _DEFAULT_MODEL = "mistv2"
    # mist / mistv2: speedAlpha < 1.0 is FASTER, > 1.0 is SLOWER (per Rime docs).
    # mistv3 / arcana: convention matches user mental model (>1 = faster).
    _SPEED_INVERTED_MODELS = {"mist", "mistv2"}

    def __init__(self) -> None:
        # Fail at adapter construction — not on first mid-call synthesis request.
        self._api_key = get_rime_api_key()

    @classmethod
    def _user_speed_to_speed_alpha(cls, user_speed: float, model_id: str) -> float:
        """Map user-facing speed (1.0 = normal, >1 = faster) to Rime speedAlpha.

        For mist/mistv2 the API direction is inverted, so we send 1/user_speed.
        For all other models we pass user_speed through unchanged.
        """
        try:
            speed = float(user_speed)
        except (TypeError, ValueError):
            speed = 1.0
        if speed <= 0:
            speed = 1.0
        if (model_id or "").lower() in cls._SPEED_INVERTED_MODELS:
            return max(0.5, min(2.0, 1.0 / speed))
        return max(0.5, min(2.0, speed))

    def list_voices(self) -> list[dict[str, Any]]:
        # Rime does not expose a public voice catalogue endpoint; return a
        # static list of known mistv2 voices for the UI voice picker.
        return [
            {"voice_id": "mistv2_Wildflower", "name": "Wildflower (mistv2)"},
            {"voice_id": "mistv2_Meadow", "name": "Meadow (mistv2)"},
            {"voice_id": "mistv2_Brook", "name": "Brook (mistv2)"},
            {"voice_id": "mistv2_Cliff", "name": "Cliff (mistv2)"},
            {"voice_id": "mistv2_Stone", "name": "Stone (mistv2)"},
        ]

    def normalize_voice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "external_voice_id": payload.get("voice_id"),
            "display_name": payload.get("name") or "Rime Voice",
            "language_code": "en",
            "gender": None,
            "accent": None,
            "description": "Rime Labs mistv2 voice",
            "preview_audio_url": None,
            "sample_rate_hz": 8000,
            "metadata_json": payload,
        }

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        model_id = cfg.get("model_id", self._DEFAULT_MODEL)
        speed_alpha = self._user_speed_to_speed_alpha(cfg.get("speed", 1.0), model_id)
        speaker = voice_external_id or self._DEFAULT_VOICE
        from app.services.rime_tts_service import rime_tts_service

        coro = rime_tts_service.synthesize(
            text=text,
            speaker=speaker,
            model_id=model_id,
            speed_alpha=speed_alpha,
            sample_rate=8000,
            audio_format="mulaw",
        )

        # Drive the coroutine on a single persistent background event loop
        # (see _RimeSyncEventLoopBridge above) instead of asyncio.run()'s
        # per-call fresh-loop-then-teardown, which corrupted rime_tts_service's
        # cached httpx.AsyncClient across calls ("Event loop is closed").
        # Safe to call regardless of whether the calling thread already has
        # its own running loop — run_coroutine_threadsafe only needs the
        # target (bridge) loop to be running, not the caller's thread.
        return _rime_sync_bridge.run(coro)

    async def async_stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """True async streaming — yields raw mulaw bytes as they arrive."""
        cfg = dict(settings_json or {})
        model_id = cfg.get("model_id", self._DEFAULT_MODEL)
        speed_alpha = self._user_speed_to_speed_alpha(cfg.get("speed", 1.0), model_id)
        speaker = voice_external_id or self._DEFAULT_VOICE
        from app.services.rime_tts_service import rime_tts_service
        async for chunk in rime_tts_service.stream_text_to_speech(
            text=text,
            speaker=speaker,
            model_id=model_id,
            speed_alpha=speed_alpha,
            sample_rate=8000,
            audio_format="mulaw",
        ):
            yield chunk

    def stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ):
        # Sync streaming is not used for Rime; callers should use async_stream_synthesize.
        raise NotImplementedError(
            "Use async_stream_synthesize for Rime TTS streaming"
        )


class HumeTTSAdapter(BaseTTSProviderAdapter):
    """
    Adapter for Hume AI TTS (streaming WebSocket, `tts/stream/input`).

    Telephony output: mulaw 8 kHz — matches Twilio MULAW 8000 directly
    (downsampled incrementally from Hume's PCM16LE stream via
    PCMStreamDownsampler; see hume_tts_service.py for details, including the
    ASSUMED source-sample-rate caveat).

    Default voice: "Male English Actor" (a HUME_AI-provider preset voice
    from Hume's Voice Library), configurable via voice_id per agent.

    settings_json keys (all optional):
        speed              (float, default 1.0) — Hume's native speed param (0.5-2.0)
        hume_voice_provider (str, default "HUME_AI") — "HUME_AI" | "CUSTOM_VOICE"
        hume_description    (str) — optional free-text prosody/acting-instructions hint,
                             Hume's own mechanism (structurally distinct from the numeric
                             stability path build_voice_settings_overlay() drives for
                             ElevenLabs) — passed through as-is, not derived from
                             HumanizationDecision.
    """

    def __init__(self) -> None:
        from app.services.hume_tts_service import HUME_DEFAULT_VOICE

        self._DEFAULT_VOICE = HUME_DEFAULT_VOICE
        # Fail at adapter construction — not on first mid-call synthesis request.
        self._api_key = get_hume_api_key()

    def list_voices(self) -> list[dict[str, Any]]:
        # Admin voice-picker UI path only — latency doesn't matter here.
        import requests

        try:
            response = requests.get(
                "https://api.hume.ai/v0/tts/voices",
                headers={"X-Hume-Api-Key": self._api_key},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError("Failed to fetch Hume voices.") from exc
        if isinstance(payload, dict):
            return payload.get("voices_page") or payload.get("voices") or []
        if isinstance(payload, list):
            return payload
        return []

    def normalize_voice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "external_voice_id": payload.get("id") or payload.get("name"),
            "display_name": payload.get("name") or "Hume Voice",
            "language_code": None,
            "gender": None,
            "accent": None,
            "description": "Hume AI voice",
            "preview_audio_url": None,
            "sample_rate_hz": 8000,
            "metadata_json": payload,
        }

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        speed = float(cfg.get("speed", 1.0))
        voice_provider = cfg.get("hume_voice_provider", "HUME_AI")
        description = cfg.get("hume_description")
        speaker = voice_external_id or self._DEFAULT_VOICE
        from app.services.hume_tts_service import hume_tts_service

        coro = hume_tts_service.synthesize(
            text=text,
            voice_external_id=speaker,
            voice_provider=voice_provider,
            speed=speed,
            description=description,
        )

        # Drive the coroutine on a single persistent background event loop
        # (see _HumeSyncEventLoopBridge above) instead of asyncio.run()'s
        # per-call fresh-loop-then-teardown — mirrors the Rime sync-bridge
        # rationale exactly, using its own dedicated bridge instance.
        return _hume_sync_bridge.run(coro)

    async def async_stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """True async streaming — yields raw mulaw bytes as they arrive."""
        cfg = dict(settings_json or {})
        speed = float(cfg.get("speed", 1.0))
        voice_provider = cfg.get("hume_voice_provider", "HUME_AI")
        description = cfg.get("hume_description")
        speaker = voice_external_id or self._DEFAULT_VOICE
        from app.services.hume_tts_service import hume_tts_service
        async for chunk in hume_tts_service.stream_text_to_speech(
            text=text,
            voice_external_id=speaker,
            voice_provider=voice_provider,
            speed=speed,
            description=description,
        ):
            yield chunk

    def stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ):
        # Sync streaming is not used for Hume; callers should use async_stream_synthesize.
        raise NotImplementedError(
            "Use async_stream_synthesize for Hume TTS streaming"
        )


class XaiTTSAdapter(BaseTTSProviderAdapter):
    """
    Adapter for xAI Grok TTS (streaming WebSocket, `wss://api.x.ai/v1/tts`).

    Telephony output: mulaw 8 kHz — requested directly from xAI via
    `codec=mulaw&sample_rate=8000` query params, so unlike Hume there is no
    resampling step here (see xai_tts_service.py).

    Default voice: "eve" (xAI's documented default), configurable via
    voice_id per agent.

    settings_json keys (all optional):
        speed                        (float, default 1.0) — xAI's native speed param (0.7-1.5)
        language                     (str, default "auto") — BCP-47 or "auto"
        optimize_streaming_latency   (int, default 2) — 0=off, 1=moderate, 2=aggressive
    """

    def __init__(self) -> None:
        from app.services.xai_tts_service import XAI_DEFAULT_VOICE

        self._DEFAULT_VOICE = XAI_DEFAULT_VOICE
        # Fail at adapter construction — not on first mid-call synthesis request.
        key = (settings.XAI_API_KEY or "").strip()
        if not key:
            raise ValueError(
                "XAI_API_KEY is not set. Add it to your environment/.env to use xAI TTS."
            )
        self._api_key = key

    def list_voices(self) -> list[dict[str, Any]]:
        # xAI's GET /v1/tts/voices exists but there is no vendor SDK / prior
        # verified integration for it in this codebase — return a small
        # static list (matches Rime's approach) so the admin voice-picker UI
        # has something to show without depending on an unverified live call.
        return [
            {"voice_id": "eve", "name": "Eve"},
        ]

    def normalize_voice_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "external_voice_id": payload.get("voice_id") or payload.get("id") or payload.get("name"),
            "display_name": payload.get("name") or "xAI Voice",
            "language_code": None,
            "gender": None,
            "accent": None,
            "description": "xAI Grok voice",
            "preview_audio_url": None,
            "sample_rate_hz": 8000,
            "metadata_json": payload,
        }

    def synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> bytes:
        cfg = dict(settings_json or {})
        speed = float(cfg.get("speed", 1.0))
        language = cfg.get("language", "auto")
        optimize_streaming_latency = int(cfg.get("optimize_streaming_latency", 2))
        voice = voice_external_id or self._DEFAULT_VOICE
        from app.services.xai_tts_service import xai_tts_service

        coro = xai_tts_service.synthesize(
            text=text,
            voice=voice,
            language=language,
            speed=speed,
            optimize_streaming_latency=optimize_streaming_latency,
        )

        # Drive the coroutine on a single persistent background event loop
        # (see _XaiSyncEventLoopBridge above) instead of asyncio.run()'s
        # per-call fresh-loop-then-teardown — mirrors the Rime/Hume
        # sync-bridge rationale exactly, using its own dedicated bridge instance.
        return _xai_sync_bridge.run(coro)

    async def async_stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """True async streaming — yields raw mulaw bytes as they arrive."""
        cfg = dict(settings_json or {})
        speed = float(cfg.get("speed", 1.0))
        language = cfg.get("language", "auto")
        optimize_streaming_latency = int(cfg.get("optimize_streaming_latency", 2))
        voice = voice_external_id or self._DEFAULT_VOICE
        from app.services.xai_tts_service import xai_tts_service
        async for chunk in xai_tts_service.stream_text_to_speech(
            text=text,
            voice=voice,
            language=language,
            speed=speed,
            optimize_streaming_latency=optimize_streaming_latency,
        ):
            yield chunk

    def stream_synthesize(
        self,
        text: str,
        voice_external_id: str,
        settings_json: dict[str, Any] | None = None,
    ):
        # Sync streaming is not used for xAI; callers should use async_stream_synthesize.
        raise NotImplementedError(
            "Use async_stream_synthesize for xAI TTS streaming"
        )


def get_tts_adapter(provider_slug: str) -> BaseTTSProviderAdapter:
    slug = (provider_slug or "").strip().lower()
    if slug == "elevenlabs":
        return ElevenLabsAdapter()
    if slug == "google":
        return GoogleTTSAdapter()
    if slug == "rime":
        return RimeTTSAdapter()
    if slug == "hume":
        return HumeTTSAdapter()
    if slug == "xai":
        return XaiTTSAdapter()
    raise ValueError(f"Unsupported TTS provider: {provider_slug}")


def get_tts_adapter_for_provider(provider: TTSProvider) -> BaseTTSProviderAdapter:
    if not provider:
        raise ValueError("TTS provider is required.")
    return get_tts_adapter(provider.slug)
