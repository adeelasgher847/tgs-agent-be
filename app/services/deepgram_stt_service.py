"""
Deepgram Speech-to-Text: streaming (Twilio MULAW) and batch (recordings).
Provider implementation used by SttPipeline and webhook transcription paths.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any, Dict, Optional

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from app.core.config import settings
from app.core.logger import logger


class DeepgramSTTService:
    """Service for Deepgram streaming + prerecorded STT."""

    def __init__(self) -> None:
        key = (settings.DEEPGRAM_API_KEY or "").strip()
        if key:
            self._client: Optional[DeepgramClient] = DeepgramClient(api_key=key)
            logger.info("Deepgram STT client initialized")
        else:
            self._client = None
            logger.warning("DEEPGRAM_API_KEY not set — STT will fail until configured")

    class StreamingSTTSession:
        """
        Long-lived Deepgram live transcription over WebSocket.
        Exposes a stable session interface (push_audio, finish, start, get_result).
        """

        def __init__(
            self,
            *,
            client: DeepgramClient,
            language_code: Optional[str],
            encoding: str,
            sample_rate: int,
            interim_results: bool,
            single_utterance: bool,
        ) -> None:
            self._client = client
            self._language_code = language_code or settings.DEEPGRAM_STT_LANGUAGE or "en"
            self._encoding = encoding.upper() if encoding else "MULAW"
            self._sample_rate = sample_rate or 8000
            self._interim_results = interim_results
            self._single_utterance = single_utterance  # unused — stream stays open like Google

            self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
            self._results_q: "queue.Queue[dict]" = queue.Queue()
            self._closed = False
            self._task_started = False
            self._thread: Optional[threading.Thread] = None
            self._session_started_monotonic: Optional[float] = None
            self._first_interim_logged = False
            self._first_final_logged = False
            self._session_end_reason = "unknown"

        def push_audio(self, audio_chunk: bytes) -> None:
            if self._closed:
                return
            self._audio_q.put(audio_chunk)

        def finish(self) -> None:
            if not self._closed:
                self._closed = True
                self._audio_q.put(None)

        async def start(self) -> None:
            if self._task_started:
                return
            self._task_started = True
            self._session_started_monotonic = time.perf_counter()
            logger.info(
                "[Deepgram STT] session_start model=%s language=%s sample_rate=%s encoding=%s",
                settings.DEEPGRAM_STT_MODEL or "nova-3",
                self._language_code,
                self._sample_rate,
                self._encoding,
            )
            self._thread = threading.Thread(target=self._run_blocking_stream, daemon=True)
            self._thread.start()

        def _run_blocking_stream(self) -> None:
            if not self._client:
                self._results_q.put(
                    {"error": "Deepgram client not initialized", "transcript": "", "confidence": 0.0, "is_final": True}
                )
                self._results_q.put({"done": True})
                return

            dg_encoding = "mulaw" if self._encoding == "MULAW" else "linear16"

            def on_message(message: Any) -> None:
                try:
                    if not isinstance(message, ListenV1Results):
                        return
                    if not message.channel or not message.channel.alternatives:
                        return
                    alt = message.channel.alternatives[0]
                    transcript = (alt.transcript or "").strip()
                    confidence = float(getattr(alt, "confidence", 0.0) or 0.0)
                    speech_final = bool(message.speech_final)

                    # Turn-taking: speech_final mirrors Google's "user stopped" final.
                    if speech_final:
                        if not transcript:
                            return
                        if (
                            self._session_started_monotonic is not None
                            and not self._first_final_logged
                        ):
                            final_latency_ms = int(
                                (time.perf_counter() - self._session_started_monotonic) * 1000
                            )
                            logger.info(
                                "[Deepgram STT] final_latency_ms=%s",
                                final_latency_ms,
                            )
                            self._first_final_logged = True
                        self._results_q.put(
                            {"transcript": transcript, "confidence": confidence, "is_final": True}
                        )
                        return

                    if not transcript:
                        return
                    if (
                        self._session_started_monotonic is not None
                        and not self._first_interim_logged
                    ):
                        first_interim_latency_ms = int(
                            (time.perf_counter() - self._session_started_monotonic) * 1000
                        )
                        logger.info(
                            "[Deepgram STT] first_interim_latency_ms=%s",
                            first_interim_latency_ms,
                        )
                        self._first_interim_logged = True
                    self._results_q.put(
                        {"transcript": transcript, "confidence": confidence, "is_final": False}
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("[Deepgram STT] on_message error: %s", exc, exc_info=True)

            def on_error(error: Any) -> None:
                self._session_end_reason = "websocket_error"
                logger.error("[Deepgram STT] websocket error: %s", error, exc_info=True)
                self._results_q.put(
                    {"error": str(error), "transcript": "", "confidence": 0.0, "is_final": True}
                )

            def sender_loop(conn: Any) -> None:
                while True:
                    chunk = self._audio_q.get()
                    if chunk is None:
                        self._session_end_reason = "client_finish"
                        try:
                            conn.send_close_stream()
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("[Deepgram STT] send_close_stream: %s", exc)
                        break
                    if chunk:
                        try:
                            conn.send_media(chunk)
                        except Exception as exc:  # noqa: BLE001
                            self._session_end_reason = "send_media_failed"
                            logger.error("[Deepgram STT] send_media failed: %s", exc, exc_info=True)
                            self._results_q.put(
                                {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                            )
                            break

            try:
                endpointing = int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 300) or 300)
                # SDK urlencodes Python bools as True/False; Deepgram requires "true"/"false"
                # or handshake returns HTTP 400 + dg-error: Invalid query string.
                interim_q: str = "true" if self._interim_results else "false"
                with self._client.listen.v1.connect(
                    model=settings.DEEPGRAM_STT_MODEL or "nova-3",
                    encoding=dg_encoding,
                    sample_rate=self._sample_rate,
                    channels=1,
                    language=self._language_code,
                    interim_results=interim_q,
                    smart_format="true",
                    endpointing=endpointing,
                    punctuate="true",
                ) as connection:
                    connection.on(EventType.MESSAGE, on_message)
                    connection.on(EventType.ERROR, on_error)

                    sender = threading.Thread(target=sender_loop, args=(connection,), daemon=True)
                    sender.start()
                    connection.start_listening()
                    if self._session_end_reason == "unknown":
                        self._session_end_reason = "normal_close"
                    sender.join(timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                self._session_end_reason = "stream_exception"
                logger.error("[Deepgram STT] streaming session error: %s", exc, exc_info=True)
                self._results_q.put(
                    {"error": str(exc), "transcript": "", "confidence": 0.0, "is_final": True}
                )
            finally:
                logger.info("[Deepgram STT] session_end reason=%s", self._session_end_reason)
                self._results_q.put({"done": True})

        async def get_result(self) -> Dict[str, Any]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._results_q.get)

    def create_streaming_session(
        self,
        language_code: Optional[str] = None,
        encoding: Optional[str] = None,
        sample_rate: Optional[int] = None,
        interim_results: bool = True,
        single_utterance: bool = False,
    ) -> "DeepgramSTTService.StreamingSTTSession":
        if not self._client:
            raise RuntimeError("Deepgram client not initialized — set DEEPGRAM_API_KEY")
        return DeepgramSTTService.StreamingSTTSession(
            client=self._client,
            language_code=language_code,
            encoding=encoding or "MULAW",
            sample_rate=sample_rate or settings.STT_SAMPLE_RATE or settings.GOOGLE_STT_SAMPLE_RATE or 8000,
            interim_results=interim_results,
            single_utterance=single_utterance,
        )

    def _transcribe_sync(
        self,
        audio_content: bytes,
        language_code: Optional[str],
        encoding: Optional[str],
        sample_rate: Optional[int],
    ) -> Dict[str, Any]:
        if not self._client:
            return {"error": "Deepgram client not initialized", "transcript": "", "confidence": 0.0}

        lang = language_code or settings.DEEPGRAM_STT_LANGUAGE or "en"
        model = settings.DEEPGRAM_STT_MODEL or "nova-3"
        enc: Optional[str] = None
        rate: int = 8000

        if encoding is None or sample_rate is None:
            if audio_content[:4] == b"RIFF":
                enc = "linear16"
                rate = int.from_bytes(audio_content[24:28], byteorder="little")
                logger.info("Deepgram prerecorded: WAV LINEAR16, %s Hz", rate)
            else:
                enc = "mulaw"
                rate = sample_rate or settings.STT_SAMPLE_RATE or settings.GOOGLE_STT_SAMPLE_RATE or 8000
                logger.info("Deepgram prerecorded: MULAW, %s Hz", rate)
        else:
            enc = "linear16" if encoding.upper() == "LINEAR16" else "mulaw"
            rate = sample_rate or 8000

        try:
            response = self._client.listen.v1.media.transcribe_file(
                request=audio_content,
                model=model,
                language=lang,
                smart_format="true",
                punctuate="true",
                encoding=enc,
                sample_rate=rate,
            )
            # ListenV1AcceptedResponse (async callback) has no results
            if not hasattr(response, "results") or response.results is None:
                return {"transcript": "", "confidence": 0.0, "is_final": True}
            channels = getattr(response.results, "channels", None) or []
            if not channels:
                return {"transcript": "", "confidence": 0.0, "is_final": True}
            ch0 = channels[0]
            alts = getattr(ch0, "alternatives", None) or []
            if not alts:
                return {"transcript": "", "confidence": 0.0, "is_final": True}
            alt0 = alts[0]
            transcript_text = (getattr(alt0, "transcript", "") or "").strip()
            conf = float(getattr(alt0, "confidence", 0.9) or 0.0)
            logger.info("Deepgram prerecorded transcript: '%s' (%.2f)", transcript_text[:200], conf)
            return {"transcript": transcript_text, "confidence": conf, "is_final": True}
        except Exception as exc:  # noqa: BLE001
            logger.error("Deepgram prerecorded error: %s", exc, exc_info=True)
            return {"error": str(exc), "transcript": "", "confidence": 0.0}

    async def transcribe_audio_chunk(
        self,
        audio_content: bytes,
        language_code: Optional[str] = None,
        encoding: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch / prerecorded transcription (replaces Google transcribe_audio_chunk_streaming)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(audio_content, language_code, encoding, sample_rate),
        )


deepgram_stt_service = DeepgramSTTService()
