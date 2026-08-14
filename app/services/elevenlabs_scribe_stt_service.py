"""
ElevenLabs Scribe v2 Realtime Speech-to-Text: streaming (Twilio MULAW / LiveKit
LINEAR16) provider. Provider implementation used by SttPipeline, mirroring
deepgram_stt_service.py's duck-typed session contract (push_audio/finish/
start/get_result).

Reuses settings.ELEVENLABS_API_KEY (same key already used for ElevenLabs TTS)
rather than a separate credential -- same vendor account for both.

Like speechmatics-rt, ElevenLabs' realtime SDK (elevenlabs.realtime.ScribeRealtime)
is asyncio-native, so the session runs directly on the caller's event loop with
no thread/queue.Queue bridge needed.

Turn detection uses Scribe's own server-side VAD/commit strategy
(commit_strategy="vad") exclusively -- no separate VAD is implemented here.
Verified against the raw AsyncAPI spec (not just SDK docstrings, which are
stale here): the wire-protocol field for transcript text is "text", not
"transcript", and error payloads use "error", not "reason". A commit (VAD-
triggered or manual) directly delivers the finalized segment via a single
committed_transcript event -- unlike Deepgram/Speechmatics there is no
separate accumulate-then-flush step, since ElevenLabs collapses "finalize"
and "notify" into one event.

The SDK has no flush-on-close primitive (unlike Speechmatics' stop_session()),
so any audio pushed since the last commit is force-committed before close()
to avoid silently dropping a caller's trailing utterance at call teardown.
"""
from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict

from elevenlabs.realtime import (
    AudioFormat,
    CommitStrategy,
    RealtimeConnection,
    RealtimeEvents,
    ScribeRealtime,
)

from app.core.config import settings
from app.core.logger import logger

_SAMPLE_RATE_TO_PCM_FORMAT = {
    8000: AudioFormat.PCM_8000,
    16000: AudioFormat.PCM_16000,
    22050: AudioFormat.PCM_22050,
    24000: AudioFormat.PCM_24000,
    44100: AudioFormat.PCM_44100,
    48000: AudioFormat.PCM_48000,
}

# Documented bounds for connect()'s VAD params. Values outside these ranges
# are rejected server-side as `invalid_request` -- but the vendored SDK's
# RealtimeEvents enum doesn't include that message type, so the SDK's
# _start_message_handler silently drops it (unmatched message_type -> no
# handler fires, {"done": True} never pushed). A misconfigured env var or
# catalog metadata_json value would otherwise produce a call with total STT
# dead-air and no error surfaced anywhere. Clamp here so an out-of-range
# value can never reach the server in the first place.
_VAD_SILENCE_THRESHOLD_SECS_RANGE = (0.3, 3.0)
_VAD_THRESHOLD_RANGE = (0.1, 0.9)
_MIN_SPEECH_DURATION_MS_RANGE = (50, 2000)
_MIN_SILENCE_DURATION_MS_RANGE = (50, 2000)


def _clamp(value: float, lo: float, hi: float, *, label: str) -> float:
    if value < lo or value > hi:
        logger.warning(
            "[ElevenLabs Scribe STT] %s=%s out of range [%s, %s] — clamping",
            label, value, lo, hi,
        )
        return max(lo, min(hi, value))
    return value


def _mask_key(key: str | None) -> str:
    """Safe, non-reversible preview for diagnostics — never logs the raw key.
    e.g. "sk_ab...wx (len=51)". Callers must never log `key` itself."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return f"<masked len={len(key)}>"
    return f"{key[:4]}...{key[-2:]} (len={len(key)})"


class ElevenLabsScribeSTTService:
    """Service for ElevenLabs Scribe v2 Realtime streaming STT."""

    def __init__(self) -> None:
        key = (settings.ELEVENLABS_API_KEY or "").strip()
        self._api_key = key or None
        if key:
            logger.info(
                "ElevenLabs Scribe STT configured (reusing ELEVENLABS_API_KEY, %s) — "
                "same key must have Speech-to-Text/Scribe access enabled on the "
                "ElevenLabs account, not just Text-to-Speech (ElevenLabs API keys "
                "can be scoped to specific products in their dashboard)",
                _mask_key(key),
            )
        else:
            logger.warning("ELEVENLABS_API_KEY not set — Scribe STT will fail until configured")

    class StreamingSTTSession:
        """
        Long-lived ElevenLabs Scribe v2 Realtime transcription over WebSocket.
        Exposes a stable session interface (push_audio, finish, start, get_result).
        """

        def __init__(
            self,
            *,
            api_key: str | None,
            language_code: str | None,
            encoding: str,
            sample_rate: int,
            model: str | None,
            vad_silence_threshold_secs: float,
            vad_threshold: float,
            min_speech_duration_ms: int,
            min_silence_duration_ms: int,
        ) -> None:
            self._api_key = api_key
            self._language_code = language_code or settings.ELEVENLABS_SCRIBE_LANGUAGE or "en"
            self._encoding = encoding.upper() if encoding else "MULAW"
            self._sample_rate = sample_rate or 8000
            self._model = model or settings.ELEVENLABS_SCRIBE_MODEL or "scribe_v2_realtime"
            self._vad_silence_threshold_secs = vad_silence_threshold_secs
            self._vad_threshold = vad_threshold
            self._min_speech_duration_ms = min_speech_duration_ms
            self._min_silence_duration_ms = min_silence_duration_ms

            self._connection: RealtimeConnection | None = None
            self._audio_q: "asyncio.Queue[bytes | None]" = asyncio.Queue()
            self._results_q: "asyncio.Queue[dict]" = asyncio.Queue()
            self._closed = False
            self._task_started = False
            self._sender_task: asyncio.Task | None = None
            # Guards against pushing {"done": True} twice (e.g. _on_error
            # pushes it, then cancels _sender_task, whose finally block would
            # otherwise push a second, orphaned one).
            self._done_pushed = False

            # Tracks whether audio has been sent since the last commit, so
            # close() knows whether a forced commit is needed to avoid
            # dropping a not-yet-committed trailing utterance.
            self._audio_since_commit = False

        def push_audio(self, audio_chunk: bytes) -> None:
            if self._closed:
                return
            self._audio_q.put_nowait(audio_chunk)

        def finish(self) -> None:
            if not self._closed:
                self._closed = True
                self._audio_q.put_nowait(None)

        async def start(self) -> None:
            if self._task_started:
                return
            self._task_started = True

            if not self._api_key:
                await self._results_q.put(
                    {"error": "ElevenLabs client not initialized", "transcript": "", "confidence": 0.0, "is_final": True}
                )
                self._push_done()
                return

            audio_format = (
                AudioFormat.ULAW_8000
                if self._encoding == "MULAW"
                else _SAMPLE_RATE_TO_PCM_FORMAT.get(self._sample_rate, AudioFormat.PCM_16000)
            )

            try:
                scribe = ScribeRealtime(api_key=self._api_key)
                options: Dict[str, Any] = {
                    "model_id": self._model,
                    "audio_format": audio_format,
                    "sample_rate": self._sample_rate,
                    "commit_strategy": CommitStrategy.VAD,
                    "vad_silence_threshold_secs": self._vad_silence_threshold_secs,
                    "vad_threshold": self._vad_threshold,
                    "min_speech_duration_ms": self._min_speech_duration_ms,
                    "min_silence_duration_ms": self._min_silence_duration_ms,
                    "language_code": self._language_code,
                }

                logger.info(
                    "[ElevenLabs Scribe STT] session_start model=%s language=%s sample_rate=%s "
                    "encoding=%s vad_silence_threshold_secs=%s vad_threshold=%s api_key=%s",
                    self._model, self._language_code, self._sample_rate,
                    self._encoding, self._vad_silence_threshold_secs, self._vad_threshold,
                    _mask_key(self._api_key),
                )
                self._connection = await scribe.connect(options)
                # No `await` between connect() returning and handlers being
                # registered, so the SDK's internal message-handler task
                # (which needs an event-loop tick to run) cannot process any
                # message before these are in place -- no missed-event race.
                self._register_handlers(self._connection)
            except Exception as exc:  # noqa: BLE001
                logger.error("[ElevenLabs Scribe STT] session start error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._push_done()
                return

            self._sender_task = asyncio.create_task(self._sender_loop())

        def _register_handlers(self, connection: RealtimeConnection) -> None:
            connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial)
            connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed)
            connection.on(RealtimeEvents.ERROR, self._on_error)

        def _on_partial(self, data: Dict[str, Any]) -> None:
            # Wire field is "text" -- verified against the raw AsyncAPI spec.
            # The SDK's own docstring examples show "transcript", which is stale.
            transcript = (data.get("text") or "").strip()
            if not transcript:
                return
            self._audio_since_commit = True
            self._results_q.put_nowait({"transcript": transcript, "confidence": 0.0, "is_final": False})

        def _on_committed(self, data: Dict[str, Any]) -> None:
            transcript = (data.get("text") or "").strip()
            self._audio_since_commit = False
            if not transcript:
                # A VAD commit on pure silence/noise -- don't trigger a spurious LLM turn.
                return
            self._results_q.put_nowait({"transcript": transcript, "confidence": 0.0, "is_final": True})

        def _on_error(self, data: Dict[str, Any]) -> None:
            reason = data.get("error") or data.get("message_type") or "unknown"
            logger.error("[ElevenLabs Scribe STT] server error: %s", reason)
            self._results_q.put_nowait(
                {"error": str(reason), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
            )
            self._push_done()
            self._closed = True
            if self._sender_task and not self._sender_task.done():
                self._sender_task.cancel()

        def _push_done(self) -> None:
            if self._done_pushed:
                return
            self._done_pushed = True
            self._results_q.put_nowait({"done": True})

        async def _sender_loop(self) -> None:
            session_end_reason = "unknown"
            try:
                while True:
                    chunk = await self._audio_q.get()
                    if chunk is None:
                        session_end_reason = "client_finish"
                        break
                    if not chunk:
                        continue
                    try:
                        chunk_b64 = base64.b64encode(chunk).decode("ascii")
                        await self._connection.send({"audio_base_64": chunk_b64})
                        self._audio_since_commit = True
                    except Exception as exc:  # noqa: BLE001
                        session_end_reason = "send_audio_failed"
                        logger.error("[ElevenLabs Scribe STT] send failed: %s", exc, exc_info=True)
                        await self._results_q.put(
                            {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                        )
                        return

                # ScribeRealtime has no equivalent of Speechmatics' stop_session()
                # flush-and-wait. Force a manual commit for any trailing audio
                # that hasn't yet crossed the VAD silence threshold (e.g. the
                # caller's last utterance right before hangup) so it isn't
                # silently dropped. commit() is documented as being "only
                # needed" under CommitStrategy.MANUAL, but nothing in the wire
                # protocol ties the per-chunk `commit` flag to the session's
                # configured commit_strategy, so this is expected to also
                # finalize a VAD-strategy session's pending segment.
                if self._audio_since_commit and self._connection is not None:
                    try:
                        await asyncio.wait_for(self._connection.commit(), timeout=3.0)
                        await asyncio.sleep(0.5)  # let the committed_transcript arrive
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[ElevenLabs Scribe STT] final forced commit failed: %s", exc)

                if self._connection is not None:
                    try:
                        await asyncio.wait_for(self._connection.close(), timeout=5.0)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[ElevenLabs Scribe STT] close failed/timed out: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                session_end_reason = "stream_exception"
                logger.error("[ElevenLabs Scribe STT] sender loop error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                )
            finally:
                logger.info("[ElevenLabs Scribe STT] session_end reason=%s", session_end_reason)
                self._push_done()

        async def get_result(self) -> Dict[str, Any]:
            return await self._results_q.get()

    def create_streaming_session(
        self,
        language_code: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        model: str | None = None,
        api_config: Dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> "ElevenLabsScribeSTTService.StreamingSTTSession":
        if not self._api_key:
            raise RuntimeError("ElevenLabs client not initialized — set ELEVENLABS_API_KEY")

        cfg = api_config or {}
        vad_silence_threshold_secs = _clamp(
            float(cfg.get("vad_silence_threshold_secs", settings.ELEVENLABS_SCRIBE_VAD_SILENCE_THRESHOLD_SECS)),
            *_VAD_SILENCE_THRESHOLD_SECS_RANGE,
            label="vad_silence_threshold_secs",
        )
        vad_threshold = _clamp(
            float(cfg.get("vad_threshold", settings.ELEVENLABS_SCRIBE_VAD_THRESHOLD)),
            *_VAD_THRESHOLD_RANGE,
            label="vad_threshold",
        )
        min_speech_duration_ms = _clamp(
            int(cfg.get("min_speech_duration_ms", settings.ELEVENLABS_SCRIBE_MIN_SPEECH_DURATION_MS)),
            *_MIN_SPEECH_DURATION_MS_RANGE,
            label="min_speech_duration_ms",
        )
        min_silence_duration_ms = _clamp(
            int(cfg.get("min_silence_duration_ms", settings.ELEVENLABS_SCRIBE_MIN_SILENCE_DURATION_MS)),
            *_MIN_SILENCE_DURATION_MS_RANGE,
            label="min_silence_duration_ms",
        )
        return ElevenLabsScribeSTTService.StreamingSTTSession(
            api_key=self._api_key,
            language_code=language_code,
            encoding=encoding or "MULAW",
            sample_rate=sample_rate or settings.STT_SAMPLE_RATE or 8000,
            model=model or cfg.get("model") or settings.ELEVENLABS_SCRIBE_MODEL,
            vad_silence_threshold_secs=vad_silence_threshold_secs,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=int(min_speech_duration_ms),
            min_silence_duration_ms=int(min_silence_duration_ms),
        )


elevenlabs_scribe_stt_service = ElevenLabsScribeSTTService()
