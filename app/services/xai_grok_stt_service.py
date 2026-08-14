"""
xAI Grok Speech-to-Text: streaming (Twilio MULAW / LiveKit LINEAR16) provider.
Provider implementation used by SttPipeline, mirroring the duck-typed session
contract (push_audio/finish/start/get_result) used by every other adapter.

Endpoint: wss://api.x.ai/v1/stt -- a standalone, STT-only realtime WebSocket,
verified against the raw (non-summarized) official docs markdown, distinct
from xAI's separate full speech-to-speech agent API (wss://api.x.ai/v1/realtime,
not used here). No vendor SDK exists for this endpoint; the official docs'
own example uses the raw `websockets` library directly (already pinned in
requirements.txt), so this adapter does the same -- no new dependency.

Audio is sent as raw binary WebSocket frames (no base64), matching Speechmatics
and unlike ElevenLabs. Config is via URL query parameters only -- no setup
message required, matching ElevenLabs and unlike Speechmatics.

Turn detection uses xAI's native server-side mechanisms exclusively -- no
local/application VAD:
  - `endpointing`: silence (ms) before a turn-boundary check fires.
  - Smart Turn (`smart_turn` threshold + `smart_turn_timeout` safety net): an
    ML model that judges whether a silence pause is a real turn end or just a
    mid-sentence pause, demoting false positives to "chunk-final" instead of
    firing the real turn boundary.

CRITICAL and easy to get wrong: xAI's `transcript.partial` event carries a
*three-state* dual flag, not a simple boolean --
  (is_final=false, speech_final=false) -> interim, still changing
  (is_final=true,  speech_final=false) -> "chunk-final": text locked for this
      ~3s window, but NOT a turn boundary (e.g. Smart Turn demoted a
      mid-sentence pause) -- must NOT be surfaced as a pipeline final, or
      every ~3s of continuous speech would spuriously trigger an LLM turn.
  (is_final=true,  speech_final=true)  -> the real turn boundary -- this is
      the only case that maps to our pipeline's is_final=True.

There is no flush-on-close primitive gap here (unlike ElevenLabs): xAI's
`{"type": "audio.done"}` -> `transcript.done` is a documented graceful-drain
sequence. Before sending it, any audio pushed since the last speech_final is
force-finalized via `{"type": "finalize"}` (xAI's PTT primitive) so a
caller's trailing utterance isn't silently dropped at call teardown.

Unlike every other provider here, xAI's `error` event does NOT close the
connection (per the official docs) -- it's treated as recoverable/non-fatal
here rather than tearing the session down, matching that documented behavior.

xAI's STT API has no `model` parameter at all (verified absent from both the
REST and streaming query-param tables in the official docs) -- there is
nothing to select. `model` is still accepted here for interface parity with
SttPipeline's provider dispatch, but it is never sent to xAI.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from urllib.parse import urlencode

from websockets.asyncio.client import connect as websocket_connect

from app.core.config import settings
from app.core.logger import logger

_XAI_STT_URL = "wss://api.x.ai/v1/stt"

# Documented bounds. Values outside these ranges may be rejected or behave
# unpredictably server-side -- clamp so a misconfigured env var/catalog
# metadata_json value can't silently degrade a live call (lesson learned
# from the Speechmatics/ElevenLabs adapters' equivalent review findings).
_ENDPOINTING_MS_RANGE = (0, 5000)
_SMART_TURN_THRESHOLD_RANGE = (0.0, 1.0)
_SMART_TURN_TIMEOUT_MS_RANGE = (1, 5000)


def _clamp(value: float, lo: float, hi: float, *, label: str) -> float:
    if value < lo or value > hi:
        logger.warning(
            "[xAI Grok STT] %s=%s out of range [%s, %s] — clamping",
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
    return f"{key[:4]}...{key[-2:]} (len={len(key)})"


class XaiGrokSTTService:
    """Service for xAI Grok realtime streaming STT."""

    def __init__(self) -> None:
        key = (settings.XAI_API_KEY or "").strip()
        self._api_key = key or None
        if key:
            logger.info("xAI Grok STT configured (%s)", _mask_key(key))
        else:
            logger.warning("XAI_API_KEY not set — xAI Grok STT will fail until configured")

    class StreamingSTTSession:
        """
        Long-lived xAI Grok realtime transcription over WebSocket.
        Exposes a stable session interface (push_audio, finish, start, get_result).
        """

        def __init__(
            self,
            *,
            api_key: str | None,
            language_code: str | None,
            encoding: str,
            sample_rate: int,
            endpointing_ms: int,
            smart_turn_threshold: float,
            smart_turn_timeout_ms: int,
        ) -> None:
            self._api_key = api_key
            self._language_code = language_code or settings.XAI_STT_LANGUAGE or "en"
            self._encoding = encoding.upper() if encoding else "MULAW"
            self._sample_rate = sample_rate or 8000
            self._endpointing_ms = endpointing_ms
            self._smart_turn_threshold = smart_turn_threshold
            self._smart_turn_timeout_ms = smart_turn_timeout_ms

            self._ws = None
            self._audio_q: "asyncio.Queue[bytes | None]" = asyncio.Queue()
            self._results_q: "asyncio.Queue[dict]" = asyncio.Queue()
            self._closed = False
            self._task_started = False
            self._sender_task: asyncio.Task | None = None
            self._receiver_task: asyncio.Task | None = None
            self._done_pushed = False

            # Set when the server sends transcript.created -- audio must not
            # be sent before this.
            self._ready_evt = asyncio.Event()
            # Set when the server sends transcript.done -- graceful shutdown
            # waits on this after sending audio.done.
            self._transcript_done_evt = asyncio.Event()

            # Tracks whether speech has occurred since the last speech_final,
            # so finish() knows whether a forced `finalize` is needed before
            # gracefully ending via audio.done.
            self._audio_since_final = False

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
                    {"error": "xAI client not initialized", "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._push_done()
                self._closed = True  # no sender/receiver task will ever drain push_audio()
                return

            xai_encoding = "mulaw" if self._encoding == "MULAW" else "pcm"
            params: Dict[str, Any] = {
                "sample_rate": self._sample_rate,
                "encoding": xai_encoding,
                "interim_results": "true",
                "endpointing": self._endpointing_ms,
                "language": self._language_code,
                "smart_turn": self._smart_turn_threshold,
                "smart_turn_timeout": self._smart_turn_timeout_ms,
            }
            url = f"{_XAI_STT_URL}?{urlencode(params)}"

            try:
                logger.info(
                    "[xAI Grok STT] session_start language=%s sample_rate=%s encoding=%s "
                    "endpointing_ms=%s smart_turn=%s smart_turn_timeout_ms=%s api_key=%s",
                    self._language_code, self._sample_rate, self._encoding,
                    self._endpointing_ms, self._smart_turn_threshold, self._smart_turn_timeout_ms,
                    _mask_key(self._api_key),
                )
                self._ws = await websocket_connect(
                    url,
                    additional_headers={"Authorization": f"Bearer {self._api_key}"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[xAI Grok STT] session start error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._push_done()
                self._closed = True  # no sender/receiver task will ever drain push_audio()
                return

            self._receiver_task = asyncio.create_task(self._receiver_loop())
            self._sender_task = asyncio.create_task(self._sender_loop())

        async def _receiver_loop(self) -> None:
            try:
                async for raw in self._ws:
                    try:
                        data = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    self._handle_message(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("[xAI Grok STT] receiver loop error: %s", exc, exc_info=True)
            finally:
                # Unblock a shutdown that's waiting on transcript.done if the
                # socket just died instead of gracefully closing.
                self._transcript_done_evt.set()

        def _handle_message(self, data: Dict[str, Any]) -> None:
            msg_type = data.get("type")
            if msg_type == "transcript.created":
                self._ready_evt.set()
            elif msg_type == "transcript.partial":
                self._handle_partial(data)
            elif msg_type == "transcript.done":
                self._transcript_done_evt.set()
            elif msg_type == "error":
                # Per xAI's docs, the connection stays open after an error --
                # surface as recoverable, don't tear the session down.
                reason = data.get("message", "unknown")
                logger.warning("[xAI Grok STT] server error (non-fatal): %s", reason)
                self._results_q.put_nowait(
                    {"error": str(reason), "transcript": "", "confidence": 0.0, "is_final": False, "recoverable": True}
                )

        def _handle_partial(self, data: Dict[str, Any]) -> None:
            transcript = (data.get("text") or "").strip()
            if not transcript:
                return

            is_final = bool(data.get("is_final"))
            speech_final = bool(data.get("speech_final"))

            if not is_final:
                # (false, false) -- interim, still changing.
                self._audio_since_final = True
                self._results_q.put_nowait({"transcript": transcript, "confidence": 0.0, "is_final": False})
                return

            if not speech_final:
                # (true, false) -- chunk-final: text locked for this window
                # but NOT a turn boundary. Withheld from the pipeline --
                # surfacing this as is_final=True would spuriously trigger an
                # LLM turn every ~3s of continuous speech.
                self._audio_since_final = True
                return

            # (true, true) -- utterance-final: the real turn boundary.
            self._audio_since_final = False
            self._results_q.put_nowait({"transcript": transcript, "confidence": 0.0, "is_final": True})

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
                            "error": "xAI STT did not signal ready (transcript.created) in time",
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
                        await self._ws.send(chunk)  # raw binary, no base64
                        self._audio_since_final = True
                    except Exception as exc:  # noqa: BLE001
                        session_end_reason = "send_audio_failed"
                        logger.error("[xAI Grok STT] send failed: %s", exc, exc_info=True)
                        await self._results_q.put(
                            {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                        )
                        return

                # Graceful shutdown: force any pending speech to finalize
                # immediately (PTT-style), then signal end-of-audio and wait
                # for the server's documented flush-and-close acknowledgement.
                if self._audio_since_final:
                    try:
                        await self._ws.send(json.dumps({"type": "finalize"}))
                        await asyncio.sleep(0.5)  # let the resulting speech_final partial arrive
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[xAI Grok STT] finalize-on-close failed: %s", exc)

                try:
                    await self._ws.send(json.dumps({"type": "audio.done"}))
                    await asyncio.wait_for(self._transcript_done_evt.wait(), timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[xAI Grok STT] graceful audio.done wait failed/timed out: %s", exc)

                try:
                    await self._ws.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[xAI Grok STT] close failed: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                session_end_reason = "stream_exception"
                logger.error("[xAI Grok STT] sender loop error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                )
            finally:
                logger.info("[xAI Grok STT] session_end reason=%s", session_end_reason)
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
    ) -> "XaiGrokSTTService.StreamingSTTSession":
        """`model` is accepted for interface parity with the other STT
        adapters' create_streaming_session() signature (SttPipeline always
        passes model=self._model_id) but is intentionally never forwarded to
        xAI -- the API has no model parameter."""
        if not self._api_key:
            raise RuntimeError("xAI client not initialized — set XAI_API_KEY")

        cfg = api_config or {}
        endpointing_ms = _clamp(
            float(cfg.get("endpointing_ms", settings.XAI_STT_ENDPOINTING_MS)),
            *_ENDPOINTING_MS_RANGE,
            label="endpointing_ms",
        )
        smart_turn_threshold = _clamp(
            float(cfg.get("smart_turn_threshold", settings.XAI_STT_SMART_TURN_THRESHOLD)),
            *_SMART_TURN_THRESHOLD_RANGE,
            label="smart_turn_threshold",
        )
        smart_turn_timeout_ms = _clamp(
            float(cfg.get("smart_turn_timeout_ms", settings.XAI_STT_SMART_TURN_TIMEOUT_MS)),
            *_SMART_TURN_TIMEOUT_MS_RANGE,
            label="smart_turn_timeout_ms",
        )
        return XaiGrokSTTService.StreamingSTTSession(
            api_key=self._api_key,
            language_code=language_code,
            encoding=encoding or "MULAW",
            sample_rate=sample_rate or settings.STT_SAMPLE_RATE or 8000,
            endpointing_ms=int(endpointing_ms),
            smart_turn_threshold=smart_turn_threshold,
            smart_turn_timeout_ms=int(smart_turn_timeout_ms),
        )


xai_grok_stt_service = XaiGrokSTTService()
