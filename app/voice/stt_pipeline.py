"""
SttPipeline — provider-agnostic streaming STT wrapper.

Supports Deepgram (existing Twilio MULAW path) and Google STT (LiveKit LINEAR16
path). Provider is selected at construction time via provider_slug.

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
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

from app.core.config import settings
from app.core.logger import logger
from app.voice.stt_events import SttEventBus, SttInterimEvent, SttFinalEvent, SttErrorEvent

if TYPE_CHECKING:
    from app.core.agent_runtime import ResolvedSttRuntime


InterimCallback = Callable[[str, float], Awaitable[None]]
FinalCallback = Callable[[str, float], Awaitable[None]]


class SttPipeline:
    """
    Provider-agnostic STT pipeline. Manages session lifecycle,
    emits typed events, and calls legacy interim/final callbacks.

    Providers:
      "deepgram"  — DeepgramSTTService (MULAW 8kHz, Twilio path)
      "google"    — GoogleSttService (LINEAR16 16kHz, LiveKit path)
    """

    def __init__(
        self,
        language_code: Optional[str],
        on_interim: InterimCallback,
        on_final: FinalCallback,
        call_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        endpointing_ms: Optional[int] = None,
        provider_slug: str = "deepgram",
        sample_rate_hz: int = 8000,
        encoding: str = "MULAW",
        silence_threshold_ms: int = 1500,
        api_config: Optional[dict] = None,
        event_bus: Optional[SttEventBus] = None,
    ) -> None:
        self._language_code = language_code
        self._on_interim = on_interim
        self._on_final = on_final
        self._call_session_id = call_session_id
        self._agent_id = agent_id
        self._endpointing_ms: Optional[int] = endpointing_ms
        self._provider_slug = provider_slug.lower()
        self._sample_rate_hz = sample_rate_hz
        self._encoding = encoding.upper()
        self._silence_threshold_ms = silence_threshold_ms
        self._api_config = api_config or {}
        self._event_bus = event_bus or SttEventBus()

        self._stt_session = None
        self._reader_task: Optional[asyncio.Task] = None

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
        call_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        endpointing_ms: Optional[int] = None,
        event_bus: Optional[SttEventBus] = None,
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

    # ── Session creation ───────────────────────────────────────────────────

    async def _ensure_session(self) -> None:
        if self._stt_session is not None:
            return

        if self._provider_slug == "google":
            await self._ensure_google_session()
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

    # ── Reader loop (provider-agnostic) ───────────────────────────────────

    async def _reader_loop(self) -> None:
        while True:
            sess = self._stt_session
            if sess is None:
                break
            try:
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
                    await self._event_bus.emit(
                        SttFinalEvent(
                            transcript=transcript,
                            confidence=confidence,
                            is_silence=is_silence,
                        )
                    )
                    await self._on_final(transcript, confidence)
                else:
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
        self._last_audio_mono = time.monotonic()
        await self._ensure_session()
        if self._stt_session:
            self._stt_session.push_audio(audio_data)

    async def recreate_with_endpointing(self, endpointing_ms: int) -> None:
        """Reopen Deepgram session with a new endpointing value (email collection).
        No-op for Google STT (uses silence_threshold_ms instead).
        """
        if self._provider_slug != "deepgram":
            logger.debug("[STT] recreate_with_endpointing is Deepgram-only; skipping")
            return
        want = int(endpointing_ms)
        if want == self._effective_endpointing_ms() and self._stt_session is not None:
            return
        await self.aclose()
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
        except Exception:
            pass

    async def aclose(self) -> None:
        """Graceful shutdown: signal finish then wait up to 5s for reader."""
        self.finish_session()
        if self._reader_task and not self._reader_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._reader_task), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[STT] reader_loop did not finish within 5s — cancelling")
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            except asyncio.CancelledError:
                pass
        self._stt_session = None
        self._reader_task = None
