"""
VoiceOrchestrator — coordination layer for the full STT → LLM → TTS pipeline.

Replaces the scattered pipeline wiring that was spread across
BidirectionalStreamHandler.__init__ and handle_media_message.

Responsibilities:
  • Owns SttPipeline lifecycle (lazy creation, email-endpointing upgrade, teardown)
  • Owns TtsPipeline lifecycle (creation, barge-in cancellation, teardown)
  • Handles user-pickup detection (RMS threshold over a short frame window)
  • Manages the STT grace period (skip Twilio system-message/ringing frames)
  • Routes STT callbacks to the handler's interim/final processors
  • Provides a single clean interface to the rest of the handler:
      orchestrator.on_audio_chunk(audio_bytes)   ← called from handle_media_message
      orchestrator.set_stream_sid(sid)            ← called from handle_start_message
      orchestrator.shutdown()                     ← called from _full_shutdown

Nothing from the business layer (LLM, booking, transcripts, RAG) is duplicated here;
those stay on BidirectionalStreamHandler so all the context is in one place.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional, Set

from app.core.config import settings
from app.core.logger import logger
from app.utils.audio_utils import ulaw_to_linear_sample
from app.voice.stt_pipeline import SttPipeline
from app.voice.tts_pipeline import TtsPipeline

if TYPE_CHECKING:
    # Avoid circular import — handler is referenced only for type hints and callbacks.
    pass


def _resolve_initial_endpointing_ms() -> int:
    """
    Map VOICE_STT_ENDPOINTING_MODE to an initial Deepgram endpointing (ms) value
    before any email-collection upgrade.
    """
    mode = (
        getattr(settings, "VOICE_STT_ENDPOINTING_MODE", "normal") or "normal"
    ).strip().lower()
    base = int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 200) or 200)
    ext = int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED", 300) or 300)
    if mode == "extended":
        return max(base, ext)
    if mode == "aggressive":
        # Snappier finals without extreme fragmentation (telephony-safe clamp).
        aggressive = max(80, int(base * 0.55))
        return min(aggressive, 400)
    return base


class VoiceOrchestrator:
    """
    Voice pipeline orchestration layer.

    Turn lifecycle (per user utterance):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Twilio MULAW frames                                                    │
    │    → on_audio_chunk()         [pickup detection + grace period gate]    │
    │      → SttPipeline            [Deepgram streaming STT]                  │
    │        → _on_interim()        [barge-in check; optional early LLM]     │
    │        → _on_final()          [full LLM trigger]                        │
    │          → handler._process_transcript()                                │
    │            → handler.generate_and_stream_response()                    │
    │              → TtsPipeline    [queue-based parallel TTS worker]         │
    │                → handler._stream_tts_chunk()   [Twilio frame streaming] │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, handler) -> None:
        """
        Args:
            handler: BidirectionalStreamHandler instance.  We hold a reference
                     so we can call back into its LLM / TTS / transcript methods
                     without duplicating their business logic here.
        """
        self._h = handler  # BidirectionalStreamHandler

        # ── STT state ────────────────────────────────────────────────────────
        self._stt_pipeline: Optional[SttPipeline] = None
        self._stt_active: bool = True
        # Deferred endpointing: stored here when the pipeline hasn't been created
        # yet but the email-collection hook fires early (race at call start).
        self._stt_deferred_endpointing_ms: Optional[int] = None
        self._email_stt_endpointing_upgraded: bool = False

        # ── User-pickup detection ─────────────────────────────────────────────
        # We use a short RMS window to detect real pickup before forwarding audio.
        # This mirrors the VAPI approach: actual caller audio, not Twilio music/
        # system messages, is what signals "user picked up".
        self._user_picked_up: bool = False
        self._first_media_received: bool = False
        self._audio_level_samples: list[int] = []
        # Absolute time.time() until which we discard audio after pickup
        # (Twilio can still send system messages in the first moments).
        self._skip_audio_until: Optional[float] = None

        # Pull thresholds from the handler so they stay in one config place.
        self._min_audio_level_threshold: int = handler._min_audio_level_threshold
        self._audio_samples_needed: int = handler._audio_samples_needed
        self._audio_non_silent_needed: int = max(
            1,
            min(
                self._audio_samples_needed,
                int(
                    getattr(
                        handler,
                        "_audio_non_silent_needed",
                        self._audio_samples_needed,
                    )
                    or self._audio_samples_needed
                ),
            ),
        )

        # ── STT confidence / barge-in thresholds (from handler) ──────────────
        self._enable_interim_llm: bool = handler._enable_interim_llm
        self._min_interim_words: int = handler._min_interim_words
        self._min_interim_confidence: float = handler._min_interim_confidence
        self._min_interim_interval_sec: float = handler._min_interim_interval_sec
        self._barge_in_min_conf: float = handler._barge_in_min_conf
        self._barge_in_min_conf_1w: float = handler._barge_in_min_conf_1w

        # ── TTS pipeline ──────────────────────────────────────────────────────
        # Create TtsPipeline here so it's owned by the orchestrator.
        # We write it back onto the handler so all existing handler methods
        # that reference `self._tts_pipeline` keep working without changes.
        self._tts_pipeline = TtsPipeline(handler)
        handler._tts_pipeline = self._tts_pipeline
        handler._tts_worker_task = self._tts_pipeline._worker_task
        pipeline_session = getattr(handler, "_pipeline", None)
        if pipeline_session is not None:
            pipeline_session.bind_tts(self._tts_pipeline)

        # Final STT callbacks are scheduled as tasks so the Deepgram reader never blocks
        # on full LLM+TTS work (parallel with continued STT ingestion).
        self._pending_final_tasks: Set[asyncio.Task] = set()

        logger.info("[VoiceOrchestrator] Initialized — STT lazy, TTS pipeline ready")

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def tts_pipeline(self) -> TtsPipeline:
        return self._tts_pipeline

    def set_stream_sid(self, stream_sid: str) -> None:
        """Called by handle_start_message when Twilio provides the stream SID.
        We don't store it here (the handler owns it), but this is a hook for
        any future orchestrator-level setup that depends on the SID being known.
        """
        # stream_sid lives on handler.stream_sid — nothing to do here for now.
        pass

    def deactivate_stt(self) -> None:
        """Stop accepting audio.  Called as part of shutdown so Twilio silence
        frames arriving after call-end don't trigger a new Deepgram session."""
        self._stt_active = False

    async def on_audio_chunk(self, audio_data: bytes) -> None:
        """
        Main entry point for every MULAW audio frame from Twilio.

        Gate order:
          1. Pickup detection  — require enough non-silent RMS frames in a short window.
          2. Grace period      — skip a brief post-pickup window to avoid
                                 feeding Twilio system messages to STT.
          3. Active guard      — ignore frames after call teardown.
          4. STT feed          — lazily create SttPipeline and push the chunk.
        """
        if not audio_data:
            return

        try:
            h = self._h

            if not self._first_media_received:
                self._first_media_received = True

            # ── 1. Pickup detection ───────────────────────────────────────────
            if not self._user_picked_up:
                audio_level = self._rms_of_mulaw(audio_data)
                self._audio_level_samples.append(audio_level)

                # Bound the sample ring-buffer
                max_samples = self._audio_samples_needed * 2
                if len(self._audio_level_samples) > max_samples:
                    self._audio_level_samples.pop(0)

                if len(self._audio_level_samples) >= self._audio_samples_needed:
                    recent = self._audio_level_samples[-self._audio_samples_needed:]
                    non_silent = sum(
                        1 for lvl in recent if lvl > self._min_audio_level_threshold
                    )
                    if non_silent >= self._audio_non_silent_needed:
                        # Confirmed: real caller audio detected.
                        # Mark as picked-up on the orchestrator FIRST so we stop
                        # running the detection loop on subsequent frames.
                        self._user_picked_up = True
                        # _handle_user_pickup has its own idempotent guard
                        # (if self._user_picked_up: return) so call it BEFORE
                        # setting h._user_picked_up to avoid an early-return.
                        await h._handle_user_pickup()
                        # Sync flag back to handler for any handler-side checks.
                        h._user_picked_up = True
                        # Keep a short grace period so ringback artifacts are skipped
                        # while preserving near-real-time pickup responsiveness.
                        grace_sec = float(
                            getattr(settings, "VOICE_POST_PICKUP_STT_GRACE_SEC", 0.35)
                            or 0.35
                        )
                        grace_sec = max(0.0, min(1.5, grace_sec))
                        self._skip_audio_until = time.time() + grace_sec
                    else:
                        # Not enough non-silent samples yet — wait.
                        return
                else:
                    # Not enough samples collected yet — wait.
                    return

            # ── 2. Grace period ───────────────────────────────────────────────
            if self._skip_audio_until and time.time() < self._skip_audio_until:
                return

            # ── 3. Active guard ───────────────────────────────────────────────
            if not self._stt_active:
                return

            # ── 4. STT feed ───────────────────────────────────────────────────
            if self._stt_pipeline is None:
                language_code = (settings.DEEPGRAM_STT_LANGUAGE or "en").strip()
                deferred_ep = self._stt_deferred_endpointing_ms
                self._stt_deferred_endpointing_ms = None
                initial_endpointing = (
                    int(deferred_ep)
                    if deferred_ep is not None
                    else _resolve_initial_endpointing_ms()
                )
                self._stt_pipeline = SttPipeline(
                    language_code=language_code,
                    on_interim=self._on_interim,
                    on_final=self._on_final,
                    call_session_id=h.call_session_id,
                    agent_id=h.agent_id,
                    endpointing_ms=initial_endpointing,
                )
                h._stt_pipeline = self._stt_pipeline
                pipeline_session = getattr(h, "_pipeline", None)
                if pipeline_session is not None:
                    pipeline_session.bind_stt(self._stt_pipeline)
                logger.debug(
                    "[VoiceOrchestrator] SttPipeline created (endpointing_ms=%s)",
                    initial_endpointing,
                )

            await self._stt_pipeline.feed_audio_chunk(audio_data)

        except Exception as exc:
            logger.error("[VoiceOrchestrator] on_audio_chunk error: %s", exc, exc_info=True)

    # ── Email-collection STT endpointing upgrade ──────────────────────────────

    def schedule_stt_recreate_for_email(self, agent_text: str) -> None:
        """
        Async-safe hook: defer the STT endpointing upgrade to the next event-loop
        tick so we never call aclose() from inside the Deepgram reader stack.
        Mirrors the old handler._schedule_recreate_stt_for_email_collection logic.
        """
        text = (agent_text or "").strip()
        if not text:
            return

        async def _deferred() -> None:
            try:
                await asyncio.sleep(0)
                await self._maybe_upgrade_stt_for_email(text)
            except Exception as exc:
                logger.debug("[VoiceOrchestrator] email-STT hook (deferred): %s", exc)

        asyncio.create_task(_deferred())

    async def _maybe_upgrade_stt_for_email(self, agent_text: str) -> None:
        """Recreate the Deepgram session with longer endpointing after the agent
        asks for an email address so spelling pauses don't split finals."""
        import re as _re
        from app.routers.bidirectional_stream import _EMAIL_AGENT_PROMPT_FOR_EXTENDED_STT_RE

        if not getattr(settings, "VOICE_STT_ENDPOINTING_EMAIL_PROMPT_RECREATES_STT", True):
            return
        if self._email_stt_endpointing_upgraded:
            return
        if not _EMAIL_AGENT_PROMPT_FOR_EXTENDED_STT_RE.search(agent_text):
            return

        ext = int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED", 2200) or 2200)
        base = int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 900) or 900)
        if ext <= base:
            self._email_stt_endpointing_upgraded = True
            return

        try:
            if self._stt_pipeline is None:
                self._stt_deferred_endpointing_ms = ext
                self._email_stt_endpointing_upgraded = True
                logger.info(
                    "[VoiceOrchestrator] deferred extended endpointing_ms=%s (email prompt, no STT yet)",
                    ext,
                )
                return
            await self._stt_pipeline.recreate_with_endpointing(ext)
            self._email_stt_endpointing_upgraded = True
            logger.info(
                "[VoiceOrchestrator] upgraded STT endpointing_ms=%s (email-collection prompt)",
                ext,
            )
        except Exception as exc:
            logger.warning(
                "[VoiceOrchestrator] extended endpointing upgrade skipped: %s", exc, exc_info=True
            )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Gracefully shut down all owned pipelines (STT + TTS).
        Idempotent — safe to call multiple times.
        Called by BidirectionalStreamHandler._full_shutdown().
        """
        self._stt_active = False

        # Cancel any async final-transcript tasks still running
        pending = list(self._pending_final_tasks)
        for t in pending:
            if t and not t.done():
                t.cancel()
        for t in pending:
            if t:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._pending_final_tasks.clear()

        # Cancel any in-flight LLM task tracked by the handler
        t = getattr(self._h, "_llm_response_task", None)
        if t and not t.done():
            t.cancel()
        self._h._llm_response_task = None

        # Signal TTS pipeline to stop current playback
        try:
            cancel_event = self._tts_pipeline.cancel_event
            if not cancel_event.is_set():
                cancel_event.set()
        except Exception:
            pass

        # Shutdown TTS worker (drains queue, cancels worker task)
        try:
            await self._tts_pipeline.shutdown()
        except Exception:
            pass

        # Close Deepgram WebSocket (sends CloseStream, waits up to 5 s for reader)
        try:
            if self._stt_pipeline:
                await self._stt_pipeline.aclose()
        except Exception:
            pass

        self._stt_pipeline = None
        pipeline_session = getattr(self._h, "_pipeline", None)
        if pipeline_session is not None:
            pipeline_session.clear_pipelines()

        logger.info("[VoiceOrchestrator] Shutdown complete")

    # ── Private STT callbacks ─────────────────────────────────────────────────

    async def _on_interim(self, transcript: str, confidence: float) -> None:
        """
        Deepgram interim result callback.
        Routed to the handler's _maybe_process_interim so barge-in detection
        and optional early-LLM logic remain centralised on the handler.
        """
        try:
            await self._h._maybe_process_interim(transcript, confidence)
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] _on_interim callback error: %s", exc, exc_info=True
            )

    async def _on_final(self, transcript: str, confidence: float) -> None:
        """
        Deepgram final result callback.
        Routed to the handler's _process_transcript which handles dedup,
        goodbye detection, voicemail detection, and LLM trigger.
        """
        try:
            h = self._h

            async def _run_final() -> None:
                try:
                    await h._process_transcript(transcript, confidence)
                except asyncio.CancelledError:
                    raise
                except Exception as cb_exc:
                    logger.error(
                        "[VoiceOrchestrator] _process_transcript error: %s",
                        cb_exc,
                        exc_info=True,
                    )

            t = asyncio.create_task(_run_final())
            self._pending_final_tasks.add(t)
            t.add_done_callback(
                lambda done_t, s=self: s._pending_final_tasks.discard(done_t)
            )
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] _on_final callback error: %s", exc, exc_info=True
            )

    # ── Audio helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _rms_of_mulaw(mulaw_bytes: bytes) -> int:
        """
        Convert MULAW bytes to linear PCM and return the integer RMS level.
        Used for voice-activity / pickup detection.
        """
        if not mulaw_bytes:
            return 0
        linear = [ulaw_to_linear_sample(b) for b in mulaw_bytes]
        mean_sq = sum(s * s for s in linear) / len(linear)
        return int(mean_sq ** 0.5)
