"""
AssemblyAI Universal-Streaming Speech-to-Text: streaming (Twilio MULAW /
LiveKit LINEAR16) provider. Provider implementation used by SttPipeline,
mirroring the duck-typed session contract (push_audio/finish/start/
get_result) used by every other adapter.

Endpoint: wss://streaming.assemblyai.com/v3/ws -- verified against the raw
(non-summarized) official docs. Auth is the raw API key in the `Authorization`
header -- no `Bearer` prefix (unlike xAI). No vendor SDK is used here; like
xAI, this adapter talks the raw `websockets` library directly (already
pinned in requirements.txt) rather than adding the official `assemblyai`
PyPI package as a second, parallel mechanism.

Audio is sent as raw binary WebSocket frames (no base64/JSON wrapping),
matching xAI/Speechmatics. Config is via URL query parameters only -- no
setup message required (`speech_model`, `encoding`, `sample_rate`, plus
AssemblyAI's native turn-detection tunables: `min_turn_silence`,
`max_turn_silence`, `end_of_turn_confidence_threshold`, `vad_threshold`,
`format_turns`). `speech_model` is a genuine, selectable API parameter --
unlike xAI's `model`, which has no server-side meaning at all. AssemblyAI
offers `universal-3-5-pro`, `universal-streaming-english`, and
`universal-streaming-multilingual`; only `universal-streaming-multilingual`
is seeded in this repo's sttmodel catalog (product decision), though any of
the three can still be passed through `api_config`/`model` directly.

CRITICAL contrast with xAI: this API's `Turn` message is *two-state*, not
three-state. `end_of_turn: false` -> interim (`is_final=False`).
`end_of_turn: true` -> that Turn's `transcript` field IS the finalized text
for the turn (`is_final=True`) -- there is no separate "chunk-final but not
a turn boundary" state to withhold, unlike xAI's speech_final gate. Do not
build one.

There is also no documented in-band JSON `error` message type for this API
(unlike xAI, whose `error` event is explicitly non-fatal/recoverable).
Consequently a WebSocket close or exception while receiving is always
treated as fatal here: if it happens before the server's `Termination`
message has been observed, a `{"error": ..., "recoverable": False}` result
is pushed followed by `{"done": True}`.

Graceful shutdown is simpler than xAI's: `finish()` sends
`{"type": "Terminate"}` and the docs state this itself finalizes any
currently-open turn server-side (the resulting final `Turn` message, if any,
arrives on the same receive loop before `Termination` does) -- so unlike
xAI's `finalize` -> `audio.done` two-step, or Speechmatics/ElevenLabs' forced
flush-before-close, there is no separate client-side flush action needed
before sending `Terminate`. The session still waits (bounded timeout,
mirroring xAI's 5s pattern) for the server's `Termination` message before
closing the socket, so a pending trailing final is not dropped.

`ForceEndpoint` (a separate manual-endpoint tool documented for this API) is
intentionally not implemented -- it isn't needed for shutdown.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from urllib.parse import urlencode

from websockets.asyncio.client import connect as websocket_connect

from app.core.config import settings
from app.core.logger import logger

_ASSEMBLYAI_STT_URL = "wss://streaming.assemblyai.com/v3/ws"

# Documented bounds. Values outside these ranges may be rejected or behave
# unpredictably server-side -- clamp so a misconfigured env var/catalog
# metadata_json value can't silently degrade a live call (lesson learned
# from the Speechmatics/ElevenLabs/xAI adapters' equivalent review findings).
_MIN_TURN_SILENCE_MS_RANGE = (50, 10000)
# max_turn_silence has no separately-documented range from min_turn_silence
# in the source docs; mirrored to the same bound as a conservative default.
_MAX_TURN_SILENCE_MS_RANGE = (50, 10000)
_END_OF_TURN_CONFIDENCE_THRESHOLD_RANGE = (0.0, 1.0)
_VAD_THRESHOLD_RANGE = (0.0, 1.0)


def _clamp(value: float, lo: float, hi: float, *, label: str) -> float:
    if value < lo or value > hi:
        logger.warning(
            "[AssemblyAI STT] %s=%s out of range [%s, %s] — clamping",
            label, value, lo, hi,
        )
        return max(lo, min(hi, value))
    return value


def _mask_key(key: str | None) -> str:
    """Safe, non-reversible preview for diagnostics — never logs the raw key."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return f"<masked len={len(key)}>"
    return f"{key[:4]}{'*' * min(8, len(key) - 4)} (len={len(key)})"


class AssemblyAiSTTService:
    """Service for AssemblyAI Universal-Streaming realtime STT."""

    def __init__(self) -> None:
        key = (settings.ASSEMBLYAI_API_KEY or "").strip()
        self._api_key = key or None
        if key:
            logger.info("AssemblyAI STT configured (%s)", _mask_key(key))
        else:
            logger.warning("ASSEMBLYAI_API_KEY not set — AssemblyAI STT will fail until configured")

    class StreamingSTTSession:
        """
        Long-lived AssemblyAI Universal-Streaming realtime transcription over
        WebSocket. Exposes a stable session interface (push_audio, finish,
        start, get_result).
        """

        def __init__(
            self,
            *,
            api_key: str | None,
            language_code: str | None,
            encoding: str,
            sample_rate: int,
            speech_model: str | None,
            min_turn_silence_ms: int,
            max_turn_silence_ms: int,
            end_of_turn_confidence_threshold: float,
            vad_threshold: float,
            format_turns: bool,
        ) -> None:
            self._api_key = api_key
            # AssemblyAI's v3 streaming API has no `language` query param --
            # language is baked into the `speech_model` choice (english vs
            # multilingual variants). Kept for interface parity/logging only,
            # same rationale as xAI's unused `model` kwarg.
            self._language_code = language_code or settings.ASSEMBLYAI_STT_LANGUAGE or "en"
            self._encoding = encoding.upper() if encoding else "MULAW"
            self._sample_rate = sample_rate or 8000
            self._speech_model = speech_model or settings.ASSEMBLYAI_STT_SPEECH_MODEL or "universal-3-5-pro"
            self._min_turn_silence_ms = min_turn_silence_ms
            self._max_turn_silence_ms = max_turn_silence_ms
            self._end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
            self._vad_threshold = vad_threshold
            self._format_turns = format_turns

            self._ws = None
            self._audio_q: "asyncio.Queue[bytes | None]" = asyncio.Queue()
            self._results_q: "asyncio.Queue[dict]" = asyncio.Queue()
            self._closed = False
            self._task_started = False
            self._sender_task: asyncio.Task | None = None
            self._receiver_task: asyncio.Task | None = None
            self._done_pushed = False

            # Set when the server sends Begin -- audio must not be sent before this.
            self._ready_evt = asyncio.Event()
            # Set when the server sends Termination -- graceful shutdown waits
            # on this after sending Terminate. Also unblocked (from
            # _receiver_loop's finally) if the socket dies before that.
            self._termination_evt = asyncio.Event()
            self._termination_received = False
            # Last unrecognized (non Begin/Turn/Termination) server payload,
            # e.g. an undocumented validation-error message sent just before
            # a close -- surfaced in the fatal error text if the socket dies.
            self._last_server_message: str = ""

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
                    {"error": "AssemblyAI client not initialized", "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._push_done()
                self._closed = True  # no sender/receiver task will ever drain push_audio()
                return

            aai_encoding = "pcm_mulaw" if self._encoding == "MULAW" else "pcm_s16le"
            params: Dict[str, Any] = {
                "speech_model": self._speech_model,
                "encoding": aai_encoding,
                "sample_rate": self._sample_rate,
                "min_turn_silence": self._min_turn_silence_ms,
                "max_turn_silence": self._max_turn_silence_ms,
                "end_of_turn_confidence_threshold": self._end_of_turn_confidence_threshold,
                "vad_threshold": self._vad_threshold,
                "format_turns": "true" if self._format_turns else "false",
            }
            url = f"{_ASSEMBLYAI_STT_URL}?{urlencode(params)}"

            try:
                logger.info(
                    "[AssemblyAI STT] session_start speech_model=%s sample_rate=%s encoding=%s "
                    "min_turn_silence_ms=%s max_turn_silence_ms=%s end_of_turn_confidence_threshold=%s "
                    "vad_threshold=%s format_turns=%s api_key=%s",
                    self._speech_model, self._sample_rate, self._encoding,
                    self._min_turn_silence_ms, self._max_turn_silence_ms,
                    self._end_of_turn_confidence_threshold, self._vad_threshold,
                    self._format_turns, _mask_key(self._api_key),
                )
                self._ws = await websocket_connect(
                    url,
                    additional_headers={"Authorization": self._api_key},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[AssemblyAI STT] session start error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._push_done()
                self._closed = True  # no sender/receiver task will ever drain push_audio()
                return

            self._receiver_task = asyncio.create_task(self._receiver_loop())
            self._sender_task = asyncio.create_task(self._sender_loop())

        async def _receiver_loop(self) -> None:
            close_detail = ""
            try:
                async for raw in self._ws:
                    try:
                        data = json.loads(raw)
                    except (TypeError, ValueError):
                        logger.warning("[AssemblyAI STT] non-JSON message received: %r", raw)
                        continue
                    self._handle_message(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                close_detail = self._describe_close_exception(exc)
                logger.error(
                    "[AssemblyAI STT] receiver loop error: %s (close_detail=%s)",
                    exc, close_detail, exc_info=True,
                )
            finally:
                if not self._termination_received:
                    # No documented in-band recoverable error type for this
                    # API -- a close/exception here before Termination was
                    # ever observed is always fatal, unlike xAI's `error`.
                    reason_bits = [b for b in (close_detail, self._last_server_message) if b]
                    reason = " | ".join(reason_bits) if reason_bits else "no further detail from server"
                    logger.warning(
                        "[AssemblyAI STT] connection ended before Termination was received — "
                        "treating as fatal (%s)", reason,
                    )
                    self._results_q.put_nowait(
                        {
                            "error": f"AssemblyAI connection closed before Termination ({reason})",
                            "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False,
                        }
                    )
                    self._push_done()
                    self._closed = True
                    if self._sender_task and not self._sender_task.done():
                        self._sender_task.cancel()
                # Unblock a shutdown that's waiting on Termination if the
                # socket just died instead of gracefully closing.
                self._termination_evt.set()

        @staticmethod
        def _describe_close_exception(exc: Exception) -> str:
            """Pull the actual WebSocket close code/reason out of a
            ConnectionClosed[Error|OK] exception -- str(exc) alone often just
            says e.g. 'received 3007 ... See Error message for details',
            which is the server pointing at a JSON payload we need to have
            captured separately (see _handle_message's unrecognized-type
            branch), not information present in the exception itself."""
            rcvd = getattr(exc, "rcvd", None)
            if rcvd is not None:
                return f"close_code={rcvd.code} close_reason={rcvd.reason!r}"
            return ""

        def _handle_message(self, data: Dict[str, Any]) -> None:
            msg_type = data.get("type")
            if msg_type == "Begin":
                logger.debug(
                    "[AssemblyAI STT] Begin id=%s expires_at=%s",
                    data.get("id"), data.get("expires_at"),
                )
                self._ready_evt.set()
            elif msg_type == "Turn":
                self._handle_turn(data)
            elif msg_type == "Termination":
                logger.info(
                    "[AssemblyAI STT] Termination audio_duration_seconds=%s session_duration_seconds=%s",
                    data.get("audio_duration_seconds"), data.get("session_duration_seconds"),
                )
                self._termination_received = True
                self._termination_evt.set()
            else:
                # Catch-all: AssemblyAI's documented message set is
                # Begin/Turn/Termination/Heartbeat, but an undocumented or
                # future message type (e.g. a config-validation error payload
                # sent just before a close) must never be silently dropped --
                # stash it so a subsequent fatal close can report it.
                logger.warning("[AssemblyAI STT] unrecognized message type=%r payload=%s", msg_type, data)
                self._last_server_message = f"server_message type={msg_type!r} payload={data}"

        def _handle_turn(self, data: Dict[str, Any]) -> None:
            transcript = (data.get("transcript") or "").strip()
            if not transcript:
                # Empty/whitespace transcript must never reach the pipeline,
                # regardless of end_of_turn.
                return
            end_of_turn = bool(data.get("end_of_turn"))
            confidence = float(data.get("confidence") or 0.0)
            # Two-state only: end_of_turn=false -> interim, end_of_turn=true
            # -> this Turn's transcript IS the finalized text for the turn.
            # No third "chunk-final but not a boundary" state exists here.
            self._results_q.put_nowait(
                {"transcript": transcript, "confidence": confidence, "is_final": end_of_turn}
            )

        def _push_done(self) -> None:
            if self._done_pushed:
                return
            self._done_pushed = True
            self._results_q.put_nowait({"done": True})

        async def _sender_loop(self) -> None:
            session_end_reason = "unknown"
            try:
                try:
                    await asyncio.wait_for(self._ready_evt.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    session_end_reason = "ready_timeout"
                    await self._results_q.put(
                        {
                            "error": "AssemblyAI did not signal ready (Begin) in time",
                            "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False,
                        }
                    )
                    return

                while True:
                    chunk = await self._audio_q.get()
                    if chunk is None:
                        session_end_reason = "client_finish"
                        break
                    if not chunk:
                        continue
                    try:
                        await self._ws.send(chunk)  # raw binary, no base64/JSON wrapping
                    except Exception as exc:  # noqa: BLE001
                        session_end_reason = "send_audio_failed"
                        logger.error("[AssemblyAI STT] send failed: %s", exc, exc_info=True)
                        await self._results_q.put(
                            {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                        )
                        return

                # Graceful shutdown: Terminate is documented to itself
                # finalize any currently-open turn server-side, so unlike
                # xAI/Speechmatics/ElevenLabs there is no separate client-side
                # flush step before it -- just send it and wait for the
                # server's Termination acknowledgement.
                try:
                    await self._ws.send(json.dumps({"type": "Terminate"}))
                    await asyncio.wait_for(self._termination_evt.wait(), timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[AssemblyAI STT] graceful Terminate wait failed/timed out: %s", exc)

                try:
                    await self._ws.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[AssemblyAI STT] close failed: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                session_end_reason = "stream_exception"
                logger.error("[AssemblyAI STT] sender loop error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._closed = True
            finally:
                self._closed = True
                logger.info("[AssemblyAI STT] session_end reason=%s", session_end_reason)
                self._push_done()
                if self._receiver_task and not self._receiver_task.done():
                    self._receiver_task.cancel()

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
    ) -> "AssemblyAiSTTService.StreamingSTTSession":
        """`model` maps to AssemblyAI's `speech_model` -- unlike xAI's unused
        `model` kwarg, this IS a real, selectable catalog value here."""
        if not self._api_key:
            raise RuntimeError("AssemblyAI client not initialized — set ASSEMBLYAI_API_KEY")

        cfg = api_config or {}
        min_turn_silence_ms = _clamp(
            float(cfg.get("min_turn_silence_ms", settings.ASSEMBLYAI_STT_MIN_TURN_SILENCE_MS)),
            *_MIN_TURN_SILENCE_MS_RANGE,
            label="min_turn_silence_ms",
        )
        max_turn_silence_ms = _clamp(
            float(cfg.get("max_turn_silence_ms", settings.ASSEMBLYAI_STT_MAX_TURN_SILENCE_MS)),
            *_MAX_TURN_SILENCE_MS_RANGE,
            label="max_turn_silence_ms",
        )
        end_of_turn_confidence_threshold = _clamp(
            float(cfg.get("end_of_turn_confidence_threshold", settings.ASSEMBLYAI_STT_END_OF_TURN_CONFIDENCE_THRESHOLD)),
            *_END_OF_TURN_CONFIDENCE_THRESHOLD_RANGE,
            label="end_of_turn_confidence_threshold",
        )
        vad_threshold = _clamp(
            float(cfg.get("vad_threshold", settings.ASSEMBLYAI_STT_VAD_THRESHOLD)),
            *_VAD_THRESHOLD_RANGE,
            label="vad_threshold",
        )
        format_turns = bool(cfg.get("format_turns", settings.ASSEMBLYAI_STT_FORMAT_TURNS))

        return AssemblyAiSTTService.StreamingSTTSession(
            api_key=self._api_key,
            language_code=language_code,
            encoding=encoding or "MULAW",
            sample_rate=sample_rate or settings.STT_SAMPLE_RATE or 8000,
            speech_model=model or cfg.get("speech_model") or settings.ASSEMBLYAI_STT_SPEECH_MODEL,
            min_turn_silence_ms=int(min_turn_silence_ms),
            max_turn_silence_ms=int(max_turn_silence_ms),
            end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
            vad_threshold=vad_threshold,
            format_turns=format_turns,
        )


assemblyai_stt_service = AssemblyAiSTTService()
