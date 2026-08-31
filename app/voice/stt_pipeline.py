"""
SttPipeline — provider-agnostic streaming STT wrapper.

Supports Deepgram (existing Twilio MULAW path), Google STT (LiveKit LINEAR16
path), Speechmatics (Twilio MULAW path, native VAD/end-of-utterance),
ElevenLabs Scribe v2 Realtime (Twilio MULAW path, native VAD/commit strategy),
xAI Grok STT (Twilio MULAW path, native Smart Turn end-of-turn detection),
and AssemblyAI Universal-Streaming (Twilio MULAW / LiveKit LINEAR16 path,
native two-state end_of_turn detection).
Provider is selected at construction time via provider_slug.

Public interface is unchanged for existing callers (VoiceOrchestrator):
  feed_audio_chunk(bytes)
  finish_session()
  aclose()
  recreate_with_endpointing(ms)  — Deepgram-only; no-op for Google

New: emit() pushes typed SttEvent objects to SttEventBus; the legacy
on_interim/on_final callbacks are still called so VoiceOrchestrator wiring
requires zero changes.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Awaitable, Callable, TYPE_CHECKING

from app.core.config import settings
from app.core.logger import logger
from app.voice.stt_events import (
    SttEventBus,
    SttInterimEvent,
    SttFinalEvent,
    SttErrorEvent,
    SttSpeechStartedEvent,
)
from app.voice.turn_signals import is_utterance_likely_incomplete

if TYPE_CHECKING:
    from app.core.agent_runtime import ResolvedSttRuntime


InterimCallback = Callable[[str, float], Awaitable[None]]
FinalCallback = Callable[[str, float], Awaitable[None]]
SpeechStartedCallback = Callable[[], Awaitable[None]]


class SttPipeline:
    """
    Provider-agnostic STT pipeline. Manages session lifecycle,
    emits typed events, and calls legacy interim/final callbacks.

    Providers:
      "deepgram"     — DeepgramSTTService (MULAW 8kHz, Twilio path)
      "google"       — GoogleSttService (LINEAR16 16kHz, LiveKit path)
      "speechmatics" — SpeechmaticsSTTService (MULAW 8kHz, Twilio path; native
                       EndOfUtterance drives turn-end, no app-side endpointing)
      "elevenlabs"   — ElevenLabsScribeSTTService (MULAW 8kHz, Twilio path;
                       native VAD/commit_strategy drives turn-end)
      "xai"          — XaiGrokSTTService (MULAW 8kHz, Twilio path; native
                       Smart Turn / endpointing drives turn-end)
      "assemblyai"   — AssemblyAiSTTService (MULAW 8kHz Twilio / LINEAR16
                       16kHz LiveKit path; native end_of_turn drives turn-end)
    """

    def __init__(
        self,
        language_code: str | None,
        on_interim: InterimCallback,
        on_final: FinalCallback,
        call_session_id: str | None = None,
        agent_id: str | None = None,
        endpointing_ms: int | None = None,
        provider_slug: str = "deepgram",
        sample_rate_hz: int = 8000,
        encoding: str = "MULAW",
        silence_threshold_ms: int = 1500,
        api_config: dict | None = None,
        event_bus: SttEventBus | None = None,
        model_id: str | None = None,
        incomplete_final_grace_ms: int = 0,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> None:
        self._language_code = language_code
        self._on_interim = on_interim
        self._on_final = on_final
        # Optional -- only Deepgram Nova-3 (StreamingSTTSession) ever emits
        # this. None for every other provider/path; guarded with a plain
        # `if self._on_speech_started:` check at the call site below.
        self._on_speech_started = on_speech_started
        self._call_session_id = call_session_id
        self._agent_id = agent_id
        self._endpointing_ms: int | None = endpointing_ms
        self._provider_slug = provider_slug.lower()
        self._sample_rate_hz = sample_rate_hz
        self._encoding = encoding.upper()
        self._silence_threshold_ms = silence_threshold_ms
        self._api_config = api_config or {}
        self._event_bus = event_bus or SttEventBus()
        self._model_id = model_id
        # See is_utterance_likely_incomplete / _maybe_extend_incomplete_final.
        # 0 (default) disables this entirely -- opt-in per caller, not global,
        # so providers/transports with their own native turn detection
        # (Speechmatics, ElevenLabs Scribe, xAI Grok, AssemblyAI, Flux) or
        # the LiveKit browser path are unaffected unless wired in explicitly.
        self._incomplete_final_grace_ms = max(0, int(incomplete_final_grace_ms or 0))
        # Result pulled ahead during a grace wait that turned out to belong
        # to the NEXT utterance rather than a continuation of the current
        # one -- stashed here so _reader_loop's next iteration processes it
        # instead of silently dropping it.
        self._pending_result: dict | None = None

        self._stt_session = None
        self._reader_task: asyncio.Task | None = None
        self._start_task: asyncio.Task | None = None
        # Set by aclose() — once the pipeline has been deliberately shut down,
        # a trailing/late-arriving audio chunk (e.g. an ffmpeg buffer flush
        # racing the call's own shutdown sequence) must not lazily reopen a
        # brand-new STT session that can never receive further audio and
        # will just time out and error minutes later. Reset by
        # recreate_with_endpointing(), which intentionally closes then
        # expects the next feed_audio_chunk() to reopen a fresh session.
        self._closed = False

        # Normalized-final dedup — catches re-endpoints within window
        self._last_final_norm_key: str = ""
        self._last_final_norm_mono: float = 0.0
        self._final_norm_dedup_sec: float = float(
            getattr(settings, "VOICE_STT_FINAL_NORMALIZED_DEDUP_SEC", 6.0) or 6.0
        )

        # Silence detection state
        self._last_audio_mono: float = time.monotonic()

    @classmethod
    def from_runtime_config(
        cls,
        resolved: "ResolvedSttRuntime",
        on_interim: InterimCallback,
        on_final: FinalCallback,
        call_session_id: str | None = None,
        agent_id: str | None = None,
        endpointing_ms: int | None = None,
        event_bus: SttEventBus | None = None,
        incomplete_final_grace_ms: int = 0,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> "SttPipeline":
        """Factory: build SttPipeline from a ResolvedSttRuntime."""
        return cls(
            language_code=resolved.language_code,
            on_interim=on_interim,
            on_final=on_final,
            call_session_id=call_session_id,
            agent_id=agent_id,
            endpointing_ms=endpointing_ms,
            provider_slug=resolved.provider_slug,
            sample_rate_hz=resolved.sample_rate_hz,
            encoding=resolved.encoding,
            silence_threshold_ms=resolved.silence_threshold_ms,
            api_config=resolved.api_config,
            event_bus=event_bus,
            model_id=resolved.model_id,
            incomplete_final_grace_ms=incomplete_final_grace_ms,
            on_speech_started=on_speech_started,
        )

    @property
    def event_bus(self) -> SttEventBus:
        return self._event_bus

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_final_key(transcript: str) -> str:
        t = (transcript or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        return t

    def _effective_endpointing_ms(self) -> int:
        if self._endpointing_ms is not None:
            return int(self._endpointing_ms)
        return int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 900) or 900)

    def _is_silence(self) -> bool:
        elapsed_ms = (time.monotonic() - self._last_audio_mono) * 1000
        return elapsed_ms >= self._silence_threshold_ms

    async def _maybe_extend_incomplete_final(
        self, transcript: str, confidence: float
    ) -> tuple[str, float]:
        """
        If `transcript` (a fresh speech_final) looks mid-sentence (see
        `is_utterance_likely_incomplete`), wait up to
        `self._incomplete_final_grace_ms` for a continuation before treating
        it as the caller's finished turn.

        Only STT-session results are consumed here (no audio feed involved),
        via the same `get_result()` queue `_reader_loop` itself drains, so
        nothing is lost while this waits -- Deepgram keeps transcribing
        in the background and queues further results normally. A result
        that turns out to belong to an unrelated NEXT utterance (not a
        continuation of this one) is stashed on `self._pending_result` for
        `_reader_loop`'s next iteration rather than dropped.

        Bounded to at most 2 extra rounds so a string of short incomplete-
        looking fragments can't stack up unbounded latency.
        """
        if not is_utterance_likely_incomplete(transcript):
            return transcript, confidence

        sess = self._stt_session
        if sess is None:
            return transcript, confidence

        merged_transcript = transcript
        merged_confidence = confidence
        deadline = time.monotonic() + (self._incomplete_final_grace_ms / 1000.0)

        for _ in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                result = await asyncio.wait_for(sess.get_result(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                raise

            if not result or result.get("done") or result.get("error"):
                self._pending_result = result
                break

            next_transcript = (result.get("transcript") or "").strip()
            if not next_transcript:
                continue

            if next_transcript.startswith(merged_transcript) or merged_transcript.startswith(
                next_transcript
            ):
                # Genuine continuation of the same utterance -- adopt the
                # longer/updated version and re-check completeness.
                if len(next_transcript) >= len(merged_transcript):
                    merged_transcript = next_transcript
                    merged_confidence = float(result.get("confidence") or merged_confidence)
                if bool(result.get("is_final")) and not is_utterance_likely_incomplete(
                    merged_transcript
                ):
                    break
                continue

            # Unrelated result (new utterance, interim of a different
            # sentence) -- not a continuation. Preserve it for the next
            # _reader_loop iteration instead of dropping it.
            self._pending_result = result
            break

        return merged_transcript, merged_confidence

    # ── Session creation ───────────────────────────────────────────────────

    async def _ensure_session(self) -> None:
        if self._stt_session is not None:
            return

        if self._provider_slug == "google":
            await self._ensure_google_session()
        elif self._provider_slug == "speechmatics":
            await self._ensure_speechmatics_session()
        elif self._provider_slug == "elevenlabs":
            await self._ensure_elevenlabs_session()
        elif self._provider_slug == "xai":
            await self._ensure_xai_session()
        elif self._provider_slug == "assemblyai":
            await self._ensure_assemblyai_session()
        else:
            await self._ensure_deepgram_session()

    async def _ensure_deepgram_session(self) -> None:
        from app.services.deepgram_stt_service import deepgram_stt_service

        self._stt_session = deepgram_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding=self._encoding,
            sample_rate=self._sample_rate_hz,
            interim_results=True,
            single_utterance=False,
            endpointing_ms=self._endpointing_ms,
            model=self._model_id,
            api_config=self._api_config,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        asyncio.create_task(self._stt_session.start())

    async def _ensure_google_session(self) -> None:
        from app.services.google_stt_service import google_stt_service

        self._stt_session = google_stt_service.create_streaming_session(
            language_code=self._language_code or "en-AU",
            sample_rate_hz=self._sample_rate_hz,
            encoding=self._encoding,
            interim_results=True,
            api_config=self._api_config,
            silence_threshold_ms=self._silence_threshold_ms,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        asyncio.create_task(self._stt_session.start())

    async def _ensure_speechmatics_session(self) -> None:
        from app.services.speechmatics_stt_service import speechmatics_stt_service

        self._stt_session = speechmatics_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding=self._encoding,
            sample_rate=self._sample_rate_hz,
            model=self._model_id,
            api_config=self._api_config,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        asyncio.create_task(self._stt_session.start())

    async def _ensure_elevenlabs_session(self) -> None:
        from app.services.elevenlabs_scribe_stt_service import elevenlabs_scribe_stt_service

        self._stt_session = elevenlabs_scribe_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding=self._encoding,
            sample_rate=self._sample_rate_hz,
            model=self._model_id,
            api_config=self._api_config,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        asyncio.create_task(self._stt_session.start())

    async def _ensure_xai_session(self) -> None:
        from app.services.xai_grok_stt_service import xai_grok_stt_service

        self._stt_session = xai_grok_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding=self._encoding,
            sample_rate=self._sample_rate_hz,
            model=self._model_id,
            api_config=self._api_config,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        asyncio.create_task(self._stt_session.start())

    async def _ensure_assemblyai_session(self) -> None:
        from app.services.assemblyai_stt_service import assemblyai_stt_service

        self._stt_session = assemblyai_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding=self._encoding,
            sample_rate=self._sample_rate_hz,
            model=self._model_id,
            api_config=self._api_config,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        # Tracked so a start() failure (e.g. WebSocket connect error) isn't a
        # silently-dropped "Future exception was never retrieved" warning;
        # start() also pushes errors onto results_q, which _reader_loop drains.
        self._start_task = asyncio.create_task(self._stt_session.start())

    # ── Reader loop (provider-agnostic) ───────────────────────────────────

    async def _reader_loop(self) -> None:
        while True:
            sess = self._stt_session
            if sess is None:
                break
            try:
                if self._pending_result is not None:
                    result = self._pending_result
                    self._pending_result = None
                else:
                    result = await sess.get_result()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("[STT] reader loop error: %s", e, exc_info=True)
                if self._stt_session is None:
                    break
                await self._event_bus.emit(SttErrorEvent(message=str(e), recoverable=True))
                continue

            if not result:
                continue
            if result.get("done"):
                break
            if result.get("speech_started"):
                # Pure VAD onset (Deepgram Nova-3 `vad_events`) -- no
                # transcript/confidence, so this is deliberately NOT run
                # through _maybe_extend_incomplete_final/dedup below. Errors
                # in the optional callback must never take down the reader
                # loop (mirrors the try/except around on_interim/on_final).
                try:
                    await self._event_bus.emit(SttSpeechStartedEvent())
                    if self._on_speech_started:
                        await self._on_speech_started()
                except Exception as cb_err:
                    logger.error(
                        "[STT] speech_started callback error: %s", cb_err, exc_info=True
                    )
                continue
            if result.get("error"):
                err_msg = result.get("error", "unknown")
                recoverable = bool(result.get("recoverable", True))
                logger.warning(
                    "[STT] session error payload: %s recoverable=%s (call_session_id=%s)",
                    err_msg,
                    recoverable,
                    self._call_session_id,
                )
                await self._event_bus.emit(
                    SttErrorEvent(message=str(err_msg), recoverable=recoverable)
                )
                continue

            transcript = (result.get("transcript") or "").strip()
            if not transcript:
                continue

            is_final = bool(result.get("is_final"))
            confidence = float(result.get("confidence") or 0.0)

            try:
                if is_final and self._incomplete_final_grace_ms > 0:
                    transcript, confidence = await self._maybe_extend_incomplete_final(
                        transcript, confidence
                    )

                if is_final:
                    norm_key = self._normalize_final_key(transcript)
                    now_mono = time.monotonic()
                    if (
                        norm_key
                        and norm_key == self._last_final_norm_key
                        and (now_mono - self._last_final_norm_mono) < self._final_norm_dedup_sec
                    ):
                        logger.debug("[STT] skipping normalized duplicate final")
                        continue
                    if norm_key:
                        self._last_final_norm_key = norm_key
                        self._last_final_norm_mono = now_mono

                    is_silence = self._is_silence()
                    acoustic_speech_end_mono = result.get("acoustic_speech_end_mono")
                    speech_end_audio_sec = result.get("speech_end_audio_sec")
                    logger.debug(
                        "[STT] final transcript received: %r confidence=%.2f "
                        "(call_session_id=%s, speech_end_audio_sec=%s)",
                        transcript[:80], confidence, self._call_session_id, speech_end_audio_sec,
                    )
                    await self._event_bus.emit(
                        SttFinalEvent(
                            transcript=transcript,
                            confidence=confidence,
                            is_silence=is_silence,
                            acoustic_speech_end_mono=acoustic_speech_end_mono,
                            speech_end_audio_sec=speech_end_audio_sec,
                        )
                    )
                    try:
                        await self._on_final(transcript, confidence, acoustic_speech_end_mono)
                    except TypeError:
                        await self._on_final(transcript, confidence)
                else:
                    logger.debug(
                        "[STT] partial transcript received: %r confidence=%.2f "
                        "(call_session_id=%s)",
                        transcript[:80], confidence, self._call_session_id,
                    )
                    await self._event_bus.emit(
                        SttInterimEvent(transcript=transcript, confidence=confidence)
                    )
                    await self._on_interim(transcript, confidence)
            except Exception as cb_err:
                logger.error("[STT] callback error: %s", cb_err, exc_info=True)

    # ── Public interface ───────────────────────────────────────────────────

    async def feed_audio_chunk(self, audio_data: bytes) -> None:
        """Feed raw audio bytes (MULAW or LINEAR16) into the streaming session."""
        if not audio_data:
            return
        if self._closed:
            logger.debug(
                "[STT] send skipped — pipeline already closed, not reopening "
                "(call_session_id=%s)", self._call_session_id,
            )
            return
        self._last_audio_mono = time.monotonic()
        await self._ensure_session()
        if self._stt_session:
            logger.debug(
                "[STT] send: bytes=%s provider=%s (call_session_id=%s)",
                len(audio_data), self._provider_slug, self._call_session_id,
            )
            self._stt_session.push_audio(audio_data)
        else:
            logger.debug(
                "[STT] send skipped — no session available (provider=%s, call_session_id=%s)",
                self._provider_slug, self._call_session_id,
            )

    async def recreate_with_endpointing(self, endpointing_ms: int) -> None:
        """Reopen Deepgram session with a new endpointing value (email collection).
        No-op for Google STT (uses silence_threshold_ms instead) and for Deepgram
        Flux models, which use native turn detection (eot_threshold/eot_timeout_ms)
        instead of app-side endpointing.
        """
        if self._provider_slug != "deepgram":
            logger.debug("[STT] recreate_with_endpointing is Deepgram-only; skipping")
            return
        if (self._model_id or "").startswith("flux-"):
            logger.debug(
                "[STT] recreate_with_endpointing is not applicable to Flux (native turn detection); skipping"
            )
            return
        want = int(endpointing_ms)
        if want == self._effective_endpointing_ms() and self._stt_session is not None:
            return
        await self.aclose()
        self._closed = False  # aclose() marks closed; this call reopens deliberately
        self._endpointing_ms = want
        self._stt_session = None
        self._reader_task = None
        logger.info(
            "[STT] recreated Deepgram session with endpointing_ms=%s (call_session_id=%s)",
            want,
            self._call_session_id,
        )

    def finish_session(self) -> None:
        """Signal the underlying STT session to finish gracefully."""
        try:
            if self._stt_session:
                self._stt_session.finish()
        except Exception as exc:
            logger.debug("[STT] finish_session failed: %s", exc)

    async def aclose(self) -> None:
        """Graceful shutdown: signal finish then wait up to 5s for reader."""
        self._closed = True
        self.finish_session()
        if self._reader_task and not self._reader_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._reader_task), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[STT] reader_loop did not finish within 5s — cancelling")
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):  # noqa: S110 - expected from cancelling the reader task above
                    pass
            except asyncio.CancelledError:
                pass
        self._stt_session = None
        self._reader_task = None
