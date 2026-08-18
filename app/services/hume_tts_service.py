"""
Hume AI TTS service — streaming WebSocket synthesis for real-time calling.

API: wss://api.hume.ai/v0/tts/stream/input?api_key={api_key}&format_type=pcm&instant_mode=true
Audio: server streams JSON messages containing base64-encoded raw PCM16LE
mono audio chunks. There is no explicit per-chunk sample rate in the
protocol; see `settings.HUME_TTS_SAMPLE_RATE_HZ` (app/core/config.py) for
the configured-but-ASSUMED source rate this service downsamples from — that
value is not confirmed by Hume's public docs as of this integration and
should be verified against a real account/API key before production
traffic depends on it.

Telephony output: mulaw 8 kHz (matches Twilio MULAW 8000), produced
incrementally via `PCMStreamDownsampler` as WebSocket chunks arrive.

This integration deliberately mirrors Rime's "one WebSocket request per
pipeline chunk" model (see rime_tts_service.py) — NOT a persistent,
cross-turn streaming-input session (that's an ElevenLabs Phase 6-3 only
feature, out of scope here).
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Any, AsyncIterator

from app.core.config import settings
from app.core.logger import logger
from app.core.secret_manager import get_hume_api_key
from app.utils.audio_utils import PCMStreamDownsampler

_HUME_WS_URL = "wss://api.hume.ai/v0/tts/stream/input"
_DEFAULT_VOICE = "Male English Actor"
_DEFAULT_VOICE_PROVIDER = "HUME_AI"

# Mirrors the connect-timeout budget used elsewhere on the hot TTS path
# (see app/services/elevenlabs_ws_session.py::_WS_CONNECT_TIMEOUT_S).
_WS_CONNECT_TIMEOUT_S: float = 5.0
# Idle read timeout per incoming message — guards against a stalled Hume
# connection hanging a chunk forever.
_WS_RECV_TIMEOUT_S: float = 15.0

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class HumeTtsServiceError(RuntimeError):
    """Connect/auth/send/receive failure talking to Hume's TTS WebSocket."""


def _looks_like_uuid(value: str) -> bool:
    try:
        return bool(_UUID_RE.match(value.strip()))
    except (TypeError, AttributeError):
        return False


_API_KEY_QS_RE = re.compile(r"(api_key=)[^&\s]+")


def _redact(text: str) -> str:
    """Strip the `api_key=...` query-string value out of any string before
    it's logged or included in a raised exception's message — defensive
    insurance against a future `websockets` version (or connection error
    path) that embeds the request URI in its own str()/repr(), which today's
    pinned websockets==16.0 does not for the exception types actually
    reachable here."""
    return _API_KEY_QS_RE.sub(r"\1<redacted>", text)


def _build_voice_payload(voice_external_id: str, voice_provider: str) -> dict[str, Any]:
    voice_id = (voice_external_id or "").strip() or _DEFAULT_VOICE
    if _looks_like_uuid(voice_id):
        return {"id": voice_id}
    return {"name": voice_id, "provider": voice_provider}


class HumeTtsService:
    """Thin async wrapper around the Hume TTS WebSocket streaming endpoint."""

    def __init__(self) -> None:
        # Resolve once at construction so a missing key fails before any live call.
        self._api_key = get_hume_api_key()

    def _build_url(self) -> str:
        return f"{_HUME_WS_URL}?api_key={self._api_key}&format_type=pcm&instant_mode=true"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_text_to_speech(
        self,
        text: str,
        voice_external_id: str = _DEFAULT_VOICE,
        voice_provider: str = _DEFAULT_VOICE_PROVIDER,
        speed: float = 1.0,
        description: str | None = None,
        src_sample_rate_hz: int | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Async generator yielding raw mulaw 8kHz audio byte chunks in
        real-time, decoded and downsampled incrementally as Hume's
        WebSocket sends PCM16LE audio messages.

        Opens one WebSocket per call (one-shot request/response, `close:
        true` sent in the single outbound message) — not a persistent
        cross-turn session.
        """
        if not text or not text.strip():
            return

        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements.txt
            raise HumeTtsServiceError(
                "the 'websockets' package is not installed"
            ) from exc

        src_rate = src_sample_rate_hz or settings.HUME_TTS_SAMPLE_RATE_HZ
        downsampler = PCMStreamDownsampler(src_rate, 8000)

        voice_payload = _build_voice_payload(voice_external_id, voice_provider)
        message: dict[str, Any] = {
            "text": text.strip(),
            "voice": voice_payload,
            "speed": float(speed),
            "close": True,
        }
        if description:
            message["description"] = description[:100]

        url = self._build_url()
        t0 = time.perf_counter()
        first_chunk = True

        try:
            ws = await asyncio.wait_for(
                websockets.connect(url), timeout=_WS_CONNECT_TIMEOUT_S
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HumeTtsServiceError(
                f"failed to open Hume TTS WebSocket: {_redact(str(exc))}"
            ) from exc

        try:
            try:
                await ws.send(json.dumps(message))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise HumeTtsServiceError(
                    f"failed to send to Hume TTS WebSocket: {_redact(str(exc))}"
                ) from exc

            while True:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=_WS_RECV_TIMEOUT_S)
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError as exc:
                    raise HumeTtsServiceError(
                        f"Hume TTS WebSocket idle timeout after {_WS_RECV_TIMEOUT_S}s"
                    ) from exc
                except websockets.ConnectionClosedOK as exc:
                    # Server closed cleanly after honoring `close: true` —
                    # expected end-of-stream for the one-shot-per-chunk model.
                    logger.debug("[Hume] connection closed normally: %s", exc)
                    break
                except Exception as exc:
                    # Abnormal close / protocol error — this is a real failure,
                    # not a benign end-of-stream. Surface it (mirrors Rime's
                    # HTTP error handling) so the caller's fallback path
                    # actually engages instead of a chunk silently going dark.
                    logger.error("[Hume] receive loop failed: %s", exc)
                    raise HumeTtsServiceError(
                        f"Hume TTS WebSocket receive failed: {_redact(str(exc))}"
                    ) from exc

                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = msg.get("type")
                if msg_type == "timestamp":
                    continue
                if msg_type == "error":
                    error_detail = msg.get("message") or msg.get("error") or msg
                    logger.error("[Hume] TTS error response: %s", error_detail)
                    raise HumeTtsServiceError(f"Hume TTS error: {error_detail}")
                if msg_type != "audio":
                    # Unrecognized message type — log it (visible in logs,
                    # unlike a fully silent break) and end the stream. Not
                    # raised, since Hume's message vocabulary beyond
                    # audio/timestamp/error isn't fully documented and an
                    # unknown-but-benign message type shouldn't hard-fail a call.
                    logger.warning(
                        "[Hume] unexpected message type %r — ending stream", msg_type
                    )
                    break

                audio_b64 = msg.get("audio")
                if audio_b64:
                    try:
                        pcm_bytes = base64.b64decode(audio_b64)
                    except Exception as exc:  # noqa: BLE001 - malformed payload must not kill the loop
                        logger.warning("[Hume] failed to decode audio chunk: %s", exc)
                        pcm_bytes = b""
                    if pcm_bytes:
                        mulaw_bytes = downsampler.feed(pcm_bytes)
                        if mulaw_bytes:
                            if first_chunk:
                                latency_ms = (time.perf_counter() - t0) * 1000
                                logger.info(
                                    "[Hume] first audio chunk latency: %.0f ms (voice=%s)",
                                    latency_ms, voice_external_id,
                                )
                                first_chunk = False
                            yield mulaw_bytes

                if bool(msg.get("is_last_chunk")):
                    break

            tail = downsampler.flush()
            if tail:
                yield tail
        finally:
            try:
                await ws.close()
            except Exception as exc:  # noqa: BLE001 - best-effort close, already tearing down
                logger.debug("[Hume] close() raised during teardown: %s", exc)

    async def synthesize(
        self,
        text: str,
        voice_external_id: str = _DEFAULT_VOICE,
        voice_provider: str = _DEFAULT_VOICE_PROVIDER,
        speed: float = 1.0,
        description: str | None = None,
    ) -> bytes:
        """Collect full audio into memory (used by batch/cache paths)."""
        chunks: list[bytes] = []
        async for chunk in self.stream_text_to_speech(
            text=text,
            voice_external_id=voice_external_id,
            voice_provider=voice_provider,
            speed=speed,
            description=description,
        ):
            chunks.append(chunk)
        return b"".join(chunks)


hume_tts_service = HumeTtsService()
