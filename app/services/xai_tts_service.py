"""
xAI Grok TTS service — streaming WebSocket synthesis for real-time calling.

API: wss://api.x.ai/v1/tts
Auth: Bearer token via the `Authorization` header on the WebSocket upgrade
request (matches how `app/services/xai_grok_stt_service.py` authenticates
xAI's STT WebSocket — xAI's TTS endpoint does not document query-string
auth). Query params configure the session: `language` (required, BCP-47 or
`auto`), `voice` (default `eve`), `codec` (`mp3`/`wav`/`pcm`/`mulaw`/`ulaw`/
`alaw`), `sample_rate`, `bit_rate` (mp3 only), `speed` (0.7-1.5),
`optimize_streaming_latency` (0=off, 1=moderate, 2=aggressive),
`text_normalization`, `with_timestamps`.

Telephony output: this service always requests `codec=mulaw&sample_rate=8000`
(matches Twilio MULAW 8000) directly from xAI — unlike Hume (PCM-only,
requires `PCMStreamDownsampler` to get to mulaw/8kHz), xAI natively emits
telephony-ready audio, so no resampling step exists in this file.

Client->server JSON messages: `text.delta` (`{"type":"text.delta","delta":
"..."}`, incremental text, <=15k chars), `text.done` (flush + synthesize the
accumulated text), `text.clear` (cancel the current utterance without closing
the connection — xAI's documented mechanism for a persistent, cross-turn
session). Server->client JSON messages: `audio.delta` (base64 audio chunk),
`audio.done` (utterance complete, connection may be reused), `audio.clear`
(ack of `text.clear`), `error` (connection stays open per xAI's docs).

Connection model: this integration deliberately mirrors Hume/Rime's "one
WebSocket per synthesis chunk" model (open -> text.delta(s) + text.done ->
drain audio.delta until audio.done -> close) — NOT a persistent, cross-turn
streaming session (that's ElevenLabs' Phase 6-3-only
`elevenlabs_ws_session.py` pattern, out of scope here, same as Hume's own
docstring scopes itself). This is consistent with how cancellation already
works elsewhere in this codebase: the pipeline layer cancels the consuming
async generator/task on barge-in, which propagates through `await ws.recv()`
and closes the socket in a `finally` block below — that alone is a valid
stop-synthesis mechanism per xAI's docs; nothing requires sending
`text.clear` before closing. `text.clear`/`audio.clear` are consequently
unused here but are xAI's documented mechanism for a future persistent-
session mode, should this integration ever grow one.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, AsyncIterator
from urllib.parse import urlencode

from app.core.config import settings
from app.core.logger import logger

_XAI_TTS_URL = "wss://api.x.ai/v1/tts"
# Public — imported by agent_runtime.py / tts_stream_mixin.py /
# livekit_browser_call_handler.py so the default voice string exists in
# exactly one place, mirroring HUME_DEFAULT_VOICE's rationale.
XAI_DEFAULT_VOICE = "eve"
_DEFAULT_VOICE = XAI_DEFAULT_VOICE
_DEFAULT_LANGUAGE = "auto"

# Mirrors the connect-timeout budget used elsewhere on the hot TTS path
# (see app/services/hume_tts_service.py / elevenlabs_ws_session.py).
_WS_CONNECT_TIMEOUT_S: float = 5.0
# Idle read timeout per incoming message — guards against a stalled xAI
# connection hanging a chunk forever.
_WS_RECV_TIMEOUT_S: float = 15.0

# xAI's documented ceiling on a single text.delta payload.
_MAX_TEXT_DELTA_CHARS = 15_000


class XaiTtsServiceError(RuntimeError):
    """Connect/auth/send/receive failure talking to xAI's TTS WebSocket."""


def _mask_key(key: str | None) -> str:
    """Safe, non-reversible preview for diagnostics — never logs the raw key."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return f"<masked len={len(key)}>"
    return f"{key[:4]}...{key[-2:]} (len={len(key)})"


class XaiTtsService:
    """Thin async wrapper around the xAI TTS WebSocket streaming endpoint."""

    def __init__(self) -> None:
        # Resolved lazily on first call, not here -- this class is
        # constructed as a module-level singleton at import time, and xAI
        # TTS is an optional provider most tenants never configure. Eagerly
        # reading/validating settings.XAI_API_KEY here would surface a
        # confusing failure mode tied to import order rather than actual use.
        pass

    def _get_api_key(self) -> str:
        key = (settings.XAI_API_KEY or "").strip()
        if not key:
            raise XaiTtsServiceError(
                "XAI_API_KEY is not set. Add it to your environment/.env to use xAI TTS."
            )
        return key

    def _build_url(
        self,
        *,
        language: str,
        voice: str,
        codec: str,
        sample_rate: int,
        speed: float,
        optimize_streaming_latency: int,
    ) -> str:
        params: dict[str, Any] = {
            "language": language,
            "voice": voice,
            "codec": codec,
            "sample_rate": sample_rate,
            "speed": speed,
            "optimize_streaming_latency": optimize_streaming_latency,
        }
        return f"{_XAI_TTS_URL}?{urlencode(params)}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_text_to_speech(
        self,
        text: str,
        voice: str = _DEFAULT_VOICE,
        language: str = _DEFAULT_LANGUAGE,
        speed: float = 1.0,
        codec: str = "mulaw",
        sample_rate: int = 8000,
        optimize_streaming_latency: int = 2,
    ) -> AsyncIterator[bytes]:
        """
        Async generator yielding raw audio byte chunks in real-time, decoded
        as xAI's WebSocket sends `audio.delta` messages. Requests
        `codec=mulaw&sample_rate=8000` by default — telephony-ready audio,
        no resampling needed downstream.

        Opens one WebSocket per call (one-shot text.delta + text.done ->
        drain audio.delta until audio.done -> close) — not a persistent
        cross-turn session; see module docstring.
        """
        if not text or not text.strip():
            return

        try:
            from websockets.asyncio.client import connect as websocket_connect
        except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements.txt
            raise XaiTtsServiceError(
                "the 'websockets' package is not installed"
            ) from exc

        api_key = self._get_api_key()
        stripped_text = text.strip()
        if len(stripped_text) > _MAX_TEXT_DELTA_CHARS:
            # Should not happen in practice — tts_stream_mixin.py chunks LLM
            # output into short per-turn segments well under this ceiling —
            # but truncating silently would clip audio mid-word with zero
            # trace in the logs if that upstream invariant ever breaks.
            logger.warning(
                "[xAI TTS] text.delta exceeds %d chars (got %d) — truncating",
                _MAX_TEXT_DELTA_CHARS, len(stripped_text),
            )
            stripped_text = stripped_text[:_MAX_TEXT_DELTA_CHARS]

        url = self._build_url(
            language=language or _DEFAULT_LANGUAGE,
            voice=voice or _DEFAULT_VOICE,
            codec=codec,
            sample_rate=sample_rate,
            speed=float(speed),
            optimize_streaming_latency=optimize_streaming_latency,
        )
        t0 = time.perf_counter()
        first_chunk = True

        try:
            ws = await asyncio.wait_for(
                websocket_connect(
                    url,
                    additional_headers={"Authorization": f"Bearer {api_key}"},
                ),
                timeout=_WS_CONNECT_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[xAI TTS] connect failed (api_key=%s): %s", _mask_key(api_key), exc
            )
            raise XaiTtsServiceError(
                f"failed to open xAI TTS WebSocket: {exc}"
            ) from exc

        try:
            try:
                await ws.send(json.dumps({"type": "text.delta", "delta": stripped_text}))
                await ws.send(json.dumps({"type": "text.done"}))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise XaiTtsServiceError(
                    f"failed to send to xAI TTS WebSocket: {exc}"
                ) from exc

            while True:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=_WS_RECV_TIMEOUT_S)
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError as exc:
                    raise XaiTtsServiceError(
                        f"xAI TTS WebSocket idle timeout after {_WS_RECV_TIMEOUT_S}s"
                    ) from exc
                except Exception as exc:
                    # ConnectionClosed (normal or abnormal) and any other
                    # protocol-level failure land here. Unlike Hume, xAI's
                    # documented end-of-utterance signal is the in-band
                    # `audio.done` JSON message, not a socket close — so a
                    # closed-before-audio.done connection is treated as a
                    # real failure (mirrors Hume/Rime's abnormal-close
                    # handling) so the caller's fallback path engages.
                    logger.error("[xAI TTS] receive loop failed: %s", exc)
                    raise XaiTtsServiceError(
                        f"xAI TTS WebSocket receive failed: {exc}"
                    ) from exc

                # Defensive: don't assume every server message is JSON text
                # (lesson from the Hume binary-frame production incident) —
                # xAI's docs only document JSON envelopes for this endpoint,
                # but a stray binary frame must not crash the receive loop.
                if isinstance(raw_msg, (bytes, bytearray)):
                    logger.warning(
                        "[xAI TTS] unexpected binary WebSocket frame (%d bytes) — skipped",
                        len(raw_msg),
                    )
                    continue

                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                    continue

                msg_type = msg.get("type")
                if msg_type == "audio.delta":
                    audio_b64 = msg.get("delta")
                    if audio_b64:
                        try:
                            audio_bytes = base64.b64decode(audio_b64)
                        except Exception as exc:  # noqa: BLE001 - malformed payload must not kill the loop
                            logger.warning("[xAI TTS] failed to decode audio chunk: %s", exc)
                            audio_bytes = b""
                        if audio_bytes:
                            if first_chunk:
                                latency_ms = (time.perf_counter() - t0) * 1000
                                logger.info(
                                    "[xAI TTS] first audio chunk latency: %.0f ms (voice=%s)",
                                    latency_ms, voice,
                                )
                                first_chunk = False
                            yield audio_bytes
                    continue
                if msg_type == "audio.done":
                    break
                if msg_type == "audio.clear":
                    # Ack of a text.clear we never send in this one-shot
                    # model; benign if it ever arrives.
                    continue
                if msg_type == "error":
                    error_detail = msg.get("message") or msg
                    logger.error("[xAI TTS] error response: %s", error_detail)
                    raise XaiTtsServiceError(f"xAI TTS error: {error_detail}")
                # Unrecognized message type — log and keep draining; xAI's
                # message vocabulary beyond the documented set isn't assumed
                # exhaustive, and an unknown-but-benign message shouldn't
                # hard-fail a call.
                logger.debug("[xAI TTS] unrecognized message type %r — ignored", msg_type)
        finally:
            try:
                await ws.close()
            except Exception as exc:  # noqa: BLE001 - best-effort close, already tearing down
                logger.debug("[xAI TTS] close() raised during teardown: %s", exc)

    async def synthesize(
        self,
        text: str,
        voice: str = _DEFAULT_VOICE,
        language: str = _DEFAULT_LANGUAGE,
        speed: float = 1.0,
        codec: str = "mulaw",
        sample_rate: int = 8000,
        optimize_streaming_latency: int = 2,
    ) -> bytes:
        """Collect full audio into memory (used by batch/cache paths)."""
        chunks: list[bytes] = []
        async for chunk in self.stream_text_to_speech(
            text=text,
            voice=voice,
            language=language,
            speed=speed,
            codec=codec,
            sample_rate=sample_rate,
            optimize_streaming_latency=optimize_streaming_latency,
        ):
            chunks.append(chunk)
        return b"".join(chunks)


xai_tts_service = XaiTtsService()
