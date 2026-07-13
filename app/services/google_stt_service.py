"""
Google Cloud Speech-to-Text v1p1beta1 streaming service.

Authentication: Application Default Credentials (ADC) or GOOGLE_APPLICATION_CREDENTIALS
(JSON content / file path). Workload Identity in GKE uses ADC with no env set.

Catalog display model "chirp-3" maps to API model "phone_call" via metadata_json.google_model.

Stream auto-restarts at the Google-imposed 5-minute limit with buffered audio replay.
Recoverable errors (quota, transient network) trigger bounded automatic restarts.
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logger import logger

_STREAM_RESTART_SECONDS = 290  # restart before Google's ~305s hard limit
_AUDIO_BUFFER_MAX_BYTES = 1_024 * 1_024
_MAX_ERROR_RESTARTS = 3
_ERROR_RESTART_WINDOW_SEC = 60.0
_ERROR_BACKOFF_SEC = 0.75


class GoogleSttService:
    """Service for Google Cloud STT v1p1beta1 bidirectional streaming."""

    def __init__(self) -> None:
        self._credentials_initialized = False
        self._auth_mode_logged = False

    def _initialize_credentials(self) -> None:
        """Resolve credentials for SpeechClient (ADC or service account file)."""
        if self._credentials_initialized:
            return
        import json
        import tempfile

        creds_env = (settings.GOOGLE_APPLICATION_CREDENTIALS or "").strip()
        auth_mode = "ADC"
        if creds_env:
            try:
                json.loads(creds_env)
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json"
                ) as f:
                    f.write(creds_env)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                auth_mode = "service_account_json_env"
            except (json.JSONDecodeError, ValueError):
                if os.path.exists(creds_env):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_env
                    auth_mode = "service_account_file"
        if not self._auth_mode_logged:
            logger.info("Google STT auth: %s", auth_mode)
            self._auth_mode_logged = True
        self._credentials_initialized = True

    def create_streaming_session(
        self,
        language_code: str = "en-AU",
        sample_rate_hz: int = 16000,
        encoding: str = "LINEAR16",
        interim_results: bool = True,
        api_config: Optional[Dict[str, Any]] = None,
        silence_threshold_ms: int = 1500,
    ) -> "GoogleSttService.StreamingSTTSession":
        self._initialize_credentials()
        return GoogleSttService.StreamingSTTSession(
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
            encoding=encoding,
            interim_results=interim_results,
            api_config=api_config or {},
            silence_threshold_ms=silence_threshold_ms,
        )

    class StreamingSTTSession:
        """Long-lived Google STT streaming session (Deepgram-compatible interface)."""

        def __init__(
            self,
            language_code: str,
            sample_rate_hz: int,
            encoding: str,
            interim_results: bool,
            api_config: Dict[str, Any],
            silence_threshold_ms: int,
        ) -> None:
            self._language_code = language_code
            self._sample_rate_hz = sample_rate_hz
            self._encoding = encoding.upper()
            self._interim_results = interim_results
            self._api_config = api_config
            self._silence_threshold_ms = silence_threshold_ms

            self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
            self._results_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
            self._audio_finished = False
            self._closed = False
            self._task_started = False
            self._thread: Optional[threading.Thread] = None

            self._stream_started_at: float = 0.0
            self._restart_buffer: List[bytes] = []
            self._restart_buffer_bytes: int = 0

            self._error_restart_count: int = 0
            self._error_restart_window_start: float = 0.0
            self._speech_end_mono: float = 0.0

        def push_audio(self, audio_chunk: bytes) -> None:
            if self._audio_finished or self._closed:
                return
            self._restart_buffer.append(audio_chunk)
            self._restart_buffer_bytes += len(audio_chunk)
            while self._restart_buffer_bytes > _AUDIO_BUFFER_MAX_BYTES and self._restart_buffer:
                removed = self._restart_buffer.pop(0)
                self._restart_buffer_bytes -= len(removed)
            self._audio_q.put(audio_chunk)

        def finish(self) -> None:
            if not self._audio_finished:
                self._speech_end_mono = time.monotonic()
                self._audio_finished = True
                self._audio_q.put(None)

        async def start(self) -> None:
            if self._task_started:
                return
            self._task_started = True
            self._thread = threading.Thread(
                target=self._run_blocking_stream, daemon=True
            )
            self._thread.start()

        async def get_result(self) -> Dict[str, Any]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._results_q.get)

        def _make_recognition_config(self):
            from google.cloud import speech_v1p1beta1 as speech

            enc_map = {
                "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
                "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
                "FLAC": speech.RecognitionConfig.AudioEncoding.FLAC,
                "OGG_OPUS": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            }
            encoding_enum = enc_map.get(
                self._encoding, speech.RecognitionConfig.AudioEncoding.LINEAR16
            )

            # Display catalog id "chirp-3" → API model "phone_call" (metadata_json only).
            google_model = self._api_config.get("google_model", "phone_call")

            kwargs: Dict[str, Any] = {
                "encoding": encoding_enum,
                "sample_rate_hertz": self._sample_rate_hz,
                "language_code": self._language_code,
                "model": google_model,
                "enable_automatic_punctuation": True,
            }
            if self._api_config.get("use_enhanced") is True:
                kwargs["use_enhanced"] = True

            return speech.RecognitionConfig(**kwargs)

        def _record_recoverable_error(self, message: str) -> bool:
            """Emit error result; return True if another restart is allowed."""
            now = time.monotonic()
            if (
                self._error_restart_window_start == 0.0
                or (now - self._error_restart_window_start) > _ERROR_RESTART_WINDOW_SEC
            ):
                self._error_restart_window_start = now
                self._error_restart_count = 0

            self._error_restart_count += 1
            self._results_q.put(
                {
                    "error": message,
                    "recoverable": self._error_restart_count <= _MAX_ERROR_RESTARTS,
                    "transcript": "",
                    "confidence": 0.0,
                    "is_final": False,
                }
            )
            if self._error_restart_count > _MAX_ERROR_RESTARTS:
                logger.error(
                    "[Google STT] max error restarts (%s) exceeded — stopping",
                    _MAX_ERROR_RESTARTS,
                )
                return False
            logger.warning(
                "[Google STT] recoverable error (restart %s/%s): %s",
                self._error_restart_count,
                _MAX_ERROR_RESTARTS,
                message,
            )
            time.sleep(_ERROR_BACKOFF_SEC)
            return True

        def _run_single_stream(
            self,
            replay_chunks: Optional[List[bytes]] = None,
        ) -> bool:
            """Run one stream. Returns True when a restart (time or recoverable error) is needed."""
            from google.cloud import speech_v1p1beta1 as speech
            from google.api_core.exceptions import (
                OutOfRange,
                PermissionDenied,
                ResourceExhausted,
                ServiceUnavailable,
                Unauthenticated,
            )

            client = speech.SpeechClient()
            config = self._make_recognition_config()
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=self._interim_results,
            )

            self._stream_started_at = time.monotonic()
            needs_restart = False
            session_end_reason_ref: dict = {"v": "normal_close"}

            def _audio_generator():
                nonlocal needs_restart
                if replay_chunks:
                    for chunk in replay_chunks:
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)

                while True:
                    if time.monotonic() - self._stream_started_at >= _STREAM_RESTART_SECONDS:
                        needs_restart = True
                        session_end_reason_ref["v"] = "time_limit"
                        return

                    try:
                        chunk = self._audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if chunk is None:
                        session_end_reason_ref["v"] = "client_finish"
                        return

                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            try:
                responses = client.streaming_recognize(streaming_config, _audio_generator())
                first_latency_logged = False
                for response in responses:
                    for result in response.results:
                        alts = result.alternatives
                        if not alts:
                            continue
                        alt = alts[0]
                        transcript = (alt.transcript or "").strip()
                        confidence = float(getattr(alt, "confidence", 0.0) or 0.0)
                        is_final = result.is_final

                        if not transcript:
                            continue

                        speech_end_to_final_ms: Optional[int] = None
                        if is_final and self._audio_finished and self._speech_end_mono > 0:
                            speech_end_to_final_ms = int(
                                (time.monotonic() - self._speech_end_mono) * 1000
                            )
                            logger.info(
                                "[Metrics] stt_speech_end_to_final=%s ms",
                                speech_end_to_final_ms,
                            )
                            if not first_latency_logged:
                                elapsed_ms = int(
                                    (time.monotonic() - self._stream_started_at) * 1000
                                )
                                logger.info(
                                    "[Google STT] first_final_latency_ms=%s lang=%s model=%s",
                                    elapsed_ms,
                                    self._language_code,
                                    self._api_config.get("google_model", "phone_call"),
                                )
                                first_latency_logged = True

                        payload: Dict[str, Any] = {
                            "transcript": transcript,
                            "confidence": confidence,
                            "is_final": is_final,
                        }
                        if speech_end_to_final_ms is not None:
                            payload["stt_speech_end_to_final_ms"] = speech_end_to_final_ms
                        self._results_q.put(payload)

            except OutOfRange as exc:
                logger.info("[Google STT] stream time limit (OutOfRange): %s", exc)
                needs_restart = True
                session_end_reason_ref["v"] = "out_of_range"
            except (ServiceUnavailable, ResourceExhausted) as exc:
                if self._record_recoverable_error(str(exc)):
                    needs_restart = True
                    session_end_reason_ref["v"] = "recoverable_error"
                else:
                    return False
            except (PermissionDenied, Unauthenticated) as exc:
                logger.error("[Google STT] auth error (non-recoverable): %s", exc)
                self._results_q.put(
                    {
                        "error": str(exc),
                        "recoverable": False,
                        "transcript": "",
                        "confidence": 0.0,
                        "is_final": False,
                    }
                )
                return False
            except Exception as exc:
                msg = str(exc)
                if self._record_recoverable_error(msg):
                    needs_restart = True
                    session_end_reason_ref["v"] = "recoverable_error"
                else:
                    return False

            logger.info(
                "[Google STT] stream ended reason=%s restart=%s",
                session_end_reason_ref["v"],
                needs_restart,
            )
            return needs_restart

        def _run_blocking_stream(self) -> None:
            logger.info(
                "[Google STT] session_start lang=%s sample_rate=%s encoding=%s api_model=%s",
                self._language_code,
                self._sample_rate_hz,
                self._encoding,
                self._api_config.get("google_model", "phone_call"),
            )

            replay_chunks: Optional[List[bytes]] = None
            restart_count = 0

            while not self._closed:
                try:
                    needs_restart = self._run_single_stream(replay_chunks=replay_chunks)
                except Exception as exc:
                    logger.error(
                        "[Google STT] unexpected stream error: %s", exc, exc_info=True
                    )
                    if not self._record_recoverable_error(str(exc)):
                        break
                    needs_restart = True

                if not needs_restart:
                    break

                if self._audio_finished:
                    logger.info(
                        "[Google STT] skip restart after client finish (reason=time_limit_or_error)"
                    )
                    break

                restart_count += 1
                replay_chunks = list(self._restart_buffer)
                logger.info(
                    "[Google STT] restarting stream #%d — replay %d chunks",
                    restart_count,
                    len(replay_chunks),
                )
                time.sleep(0.05)

            self._closed = True
            logger.info("[Google STT] session_end restart_count=%d", restart_count)
            self._results_q.put({"done": True})


google_stt_service = GoogleSttService()
