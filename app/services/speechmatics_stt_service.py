"""
Speechmatics Speech-to-Text: streaming (Twilio MULAW / LiveKit LINEAR16) provider.
Provider implementation used by SttPipeline, mirroring deepgram_stt_service.py's
duck-typed session contract (push_audio/finish/start/get_result).

Unlike Deepgram/Google (sync SDKs bridged into asyncio via a background thread),
speechmatics-rt's AsyncClient is asyncio-native, so the session runs directly on
the caller's event loop with no thread/queue.Queue bridge needed.

Turn detection uses Speechmatics' own server-side EndOfUtterance signal (driven
by transcription_config.conversation_config.end_of_utterance_silence_trigger) —
no separate VAD is implemented here. AddTranscript segments are accumulated but
withheld from the pipeline as interim-only; only EndOfUtterance promotes the
accumulated text to a pipeline-final (is_final=True), matching how Deepgram's
speech_final gates turn-end for Nova.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    ConversationConfig,
    Model,
    ServerMessageType,
    StaticKeyAuth,
    TranscriptionConfig,
)

from app.core.config import settings
from app.core.logger import logger


def _avg_confidence(msg: Dict[str, Any]) -> float:
    results = msg.get("results") or []
    confidences = []
    for item in results:
        alts = item.get("alternatives") or []
        if alts:
            confidences.append(float(alts[0].get("confidence") or 0.0))
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


class SpeechmaticsSTTService:
    """Service for Speechmatics realtime streaming STT."""

    def __init__(self) -> None:
        key = (settings.SPEECHMATICS_API_KEY or "").strip()
        self._api_key = key or None
        if key:
            logger.info("Speechmatics STT configured")
        else:
            logger.warning("SPEECHMATICS_API_KEY not set — STT will fail until configured")

    class StreamingSTTSession:
        """
        Long-lived Speechmatics realtime transcription over WebSocket.
        Exposes a stable session interface (push_audio, finish, start, get_result).
        """

        def __init__(
            self,
            *,
            api_key: str | None,
            url: str | None,
            language_code: str | None,
            encoding: str,
            sample_rate: int,
            model: str | None,
            end_of_utterance_silence_trigger: float,
            max_delay: float,
        ) -> None:
            self._api_key = api_key
            self._url = url
            self._language_code = language_code or settings.SPEECHMATICS_STT_LANGUAGE or "en"
            self._encoding = encoding.upper() if encoding else "MULAW"
            self._sample_rate = sample_rate or 8000
            self._model = model or settings.SPEECHMATICS_STT_MODEL or "enhanced"
            self._eot_silence_trigger = end_of_utterance_silence_trigger
            self._max_delay = max_delay

            self._client: AsyncClient | None = None
            self._audio_q: "asyncio.Queue[bytes | None]" = asyncio.Queue()
            self._results_q: "asyncio.Queue[dict]" = asyncio.Queue()
            self._closed = False
            self._task_started = False
            self._sender_task: asyncio.Task | None = None

            # AddTranscript segments accumulate here; only surfaced to the
            # pipeline (is_final=True) when the native EndOfUtterance event
            # fires -- this is what makes Speechmatics' own VAD the sole
            # turn-boundary signal, matching Deepgram Nova's speech_final gate.
            self._pending_transcript = ""
            self._pending_confidence = 0.0

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
                    {"error": "Speechmatics client not initialized", "transcript": "", "confidence": 0.0, "is_final": True}
                )
                await self._results_q.put({"done": True})
                return

            sm_encoding = AudioEncoding.MULAW if self._encoding == "MULAW" else AudioEncoding.PCM_S16LE

            try:
                self._client = AsyncClient(
                    auth=StaticKeyAuth(self._api_key),
                    url=self._url or None,
                )
                self._register_handlers(self._client)

                logger.info(
                    "[Speechmatics STT] session_start model=%s language=%s sample_rate=%s "
                    "encoding=%s eot_silence_trigger=%s max_delay=%s",
                    self._model, self._language_code, self._sample_rate,
                    self._encoding, self._eot_silence_trigger, self._max_delay,
                )
                await self._client.start_session(
                    transcription_config=TranscriptionConfig(
                        language=self._language_code,
                        model=Model(self._model),
                        enable_partials=True,
                        max_delay=self._max_delay,
                        max_delay_mode="flexible",
                        conversation_config=ConversationConfig(
                            end_of_utterance_silence_trigger=self._eot_silence_trigger,
                        ),
                    ),
                    audio_format=AudioFormat(
                        encoding=sm_encoding,
                        sample_rate=self._sample_rate,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[Speechmatics STT] session start error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                await self._results_q.put({"done": True})
                return

            self._sender_task = asyncio.create_task(self._sender_loop())

        def _register_handlers(self, client: AsyncClient) -> None:
            @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def _on_partial(msg: Dict[str, Any]) -> None:
                transcript = ((msg.get("metadata") or {}).get("transcript") or "").strip()
                if not transcript:
                    return
                self._results_q.put_nowait(
                    {"transcript": transcript, "confidence": _avg_confidence(msg), "is_final": False}
                )

            @client.on(ServerMessageType.ADD_TRANSCRIPT)
            def _on_transcript(msg: Dict[str, Any]) -> None:
                transcript = ((msg.get("metadata") or {}).get("transcript") or "").strip()
                if not transcript:
                    return
                # Withheld from the pipeline until EndOfUtterance -- AddTranscript
                # only means "this segment's wording is settled", not "turn over".
                self._pending_transcript = (
                    f"{self._pending_transcript} {transcript}".strip()
                    if self._pending_transcript
                    else transcript
                )
                self._pending_confidence = _avg_confidence(msg)

            @client.on(ServerMessageType.END_OF_UTTERANCE)
            def _on_end_of_utterance(_msg: Dict[str, Any]) -> None:
                if not self._pending_transcript:
                    return
                self._results_q.put_nowait(
                    {
                        "transcript": self._pending_transcript,
                        "confidence": self._pending_confidence,
                        "is_final": True,
                    }
                )
                self._pending_transcript = ""
                self._pending_confidence = 0.0

            @client.on(ServerMessageType.ERROR)
            def _on_error(msg: Dict[str, Any]) -> None:
                reason = msg.get("reason", "unknown")
                logger.error("[Speechmatics STT] server error: %s", reason)
                self._results_q.put_nowait(
                    {"error": str(reason), "transcript": "", "confidence": 0.0, "is_final": True, "recoverable": False}
                )
                self._results_q.put_nowait({"done": True})
                self._closed = True
                if self._sender_task and not self._sender_task.done():
                    self._sender_task.cancel()

            @client.on(ServerMessageType.WARNING)
            def _on_warning(msg: Dict[str, Any]) -> None:
                logger.warning("[Speechmatics STT] server warning: %s", msg.get("reason", "unknown"))

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
                        await self._client.send_audio(chunk)
                    except Exception as exc:  # noqa: BLE001
                        session_end_reason = "send_audio_failed"
                        logger.error("[Speechmatics STT] send_audio failed: %s", exc, exc_info=True)
                        await self._results_q.put(
                            {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                        )
                        return

                try:
                    await asyncio.wait_for(self._client.stop_session(), timeout=10.0)
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    logger.warning("[Speechmatics STT] graceful stop_session failed/timed out: %s", exc)
                    await self._client.close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                session_end_reason = "stream_exception"
                logger.error("[Speechmatics STT] sender loop error: %s", exc, exc_info=True)
                await self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                )
            finally:
                # Flush a still-pending AddTranscript segment that never got a
                # trailing EndOfUtterance (e.g. the caller's last utterance
                # before hangup, finalized right as the call tears down) --
                # otherwise it's silently dropped. Mirrors Deepgram's
                # UtteranceEnd/pending-prefix fallback in deepgram_stt_service.py.
                if self._pending_transcript:
                    await self._results_q.put(
                        {
                            "transcript": self._pending_transcript,
                            "confidence": self._pending_confidence,
                            "is_final": True,
                        }
                    )
                    self._pending_transcript = ""
                    self._pending_confidence = 0.0
                logger.info("[Speechmatics STT] session_end reason=%s", session_end_reason)
                await self._results_q.put({"done": True})

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
    ) -> "SpeechmaticsSTTService.StreamingSTTSession":
        if not self._api_key:
            raise RuntimeError("Speechmatics client not initialized — set SPEECHMATICS_API_KEY")

        cfg = api_config or {}
        eot_silence_trigger = float(
            cfg.get("end_of_utterance_silence_trigger", settings.SPEECHMATICS_EOT_SILENCE_TRIGGER_SEC)
        )
        max_delay = float(cfg.get("max_delay", settings.SPEECHMATICS_MAX_DELAY_SEC))

        return SpeechmaticsSTTService.StreamingSTTSession(
            api_key=self._api_key,
            url=settings.SPEECHMATICS_RT_URL or None,
            language_code=language_code,
            encoding=encoding or "MULAW",
            sample_rate=sample_rate or settings.STT_SAMPLE_RATE or 8000,
            model=model or cfg.get("model") or settings.SPEECHMATICS_STT_MODEL,
            end_of_utterance_silence_trigger=eot_silence_trigger,
            max_delay=max_delay,
        )


speechmatics_stt_service = SpeechmaticsSTTService()
