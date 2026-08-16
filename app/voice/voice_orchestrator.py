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
from typing import TYPE_CHECKING, Any, Set

from app.core.config import settings
from app.core.llm_models import is_gemini_live_native_audio_model
from app.core.logger import logger
from app.utils.audio_utils import ulaw_to_linear_sample
from app.voice.gemini_live_audio_bridge import mulaw8k_to_pcm16_16k, pcm16_24k_to_mulaw8k
from app.voice.stt_pipeline import SttPipeline
from app.voice.tts_pipeline import TtsPipeline
from app.voice.stt_events import SttEventBus

if TYPE_CHECKING:
    from app.core.agent_runtime import ResolvedSttRuntime

# Text model used to keep a native-audio call alive when GeminiLiveSession.start()
# fails BEFORE the session ever became usable (pre-session failure — see
# _start_gemini_live_session / _fallback_to_legacy_pipeline). Native-audio model
# IDs (e.g. "gemini-live-2.5-flash-native-audio") have no meaning to
# vertex_gemini_service.py's text-generation methods, so the in-memory
# `agent.llm_model` is swapped to this allow-listed text model for the
# remainder of THIS call object only (never persisted to the DB).
_GEMINI_LIVE_FALLBACK_TEXT_MODEL = "gemini-2.5-flash"


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
        self._stt_pipeline: SttPipeline | None = None
        self._stt_active: bool = True
        self._stt_deferred_endpointing_ms: int | None = None
        self._email_stt_endpointing_upgraded: bool = False
        # Resolved STT config (set by caller before first audio arrives)
        self._resolved_stt: "ResolvedSttRuntime" | None = None
        self._stt_event_bus: SttEventBus = SttEventBus()
        # LiveKit audio subscriber task (Google STT path only)
        self._livekit_audio_task: asyncio.Task | None = None

        # ── User-pickup detection ─────────────────────────────────────────────
        # We use a short RMS window to detect real pickup before forwarding audio.
        # This mirrors the VAPI approach: actual caller audio, not Twilio music/
        # system messages, is what signals "user picked up".
        self._user_picked_up: bool = False
        self._first_media_received: bool = False
        self._audio_level_samples: list[int] = []
        # Absolute time.time() until which we discard audio after pickup
        # (Twilio can still send system messages in the first moments).
        self._skip_audio_until: float | None = None

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
        ps = getattr(handler, "_pipeline_session", None)
        if ps is not None:
            ps.tts_pipeline = self._tts_pipeline

        # Final STT callbacks are scheduled as tasks so the Deepgram reader never blocks
        # on full LLM+TTS work (parallel with continued STT ingestion).
        self._pending_final_tasks: Set[asyncio.Task] = set()

        # ── Gemini Live (native-audio speech-to-speech) state ─────────────────
        # Resolved once here (not per-chunk) — handler.agent is already loaded
        # by BidirectionalStreamHandler._load_session_data(), which always runs
        # synchronously in __init__ before VoiceOrchestrator is constructed.
        # When True, on_audio_chunk() never creates SttPipeline for this call;
        # audio instead flows to a lazily-created GeminiLiveSession (see
        # _feed_gemini_live_audio / _start_gemini_live_session). NOTE:
        # TtsPipeline above is still constructed unconditionally (see class
        # docstring / task write-up) — it is cheap (no I/O, no provider calls
        # in __init__) and is simply never fed for a native-audio call, so no
        # external TTS provider is ever invoked.
        _agent = getattr(handler, "agent", None)
        self._is_gemini_live: bool = is_gemini_live_native_audio_model(
            getattr(_agent, "llm_model", None) if _agent else None
        )
        self._gemini_live_session: Any = None
        self._gemini_live_setup_lock: asyncio.Lock = asyncio.Lock()
        self._gemini_live_setup_failed: bool = False
        # Barge-in cancel flag scoped to Gemini Live's own outbound audio send
        # loop — the equivalent of handler._tts_cancel for this path, since
        # there is no TtsPipeline task to cancel (see _on_gemini_live_interrupted).
        self._gemini_live_cancel: asyncio.Event = asyncio.Event()
        self._gemini_live_first_audio_marked: bool = False

        logger.info(
            "[VoiceOrchestrator] Initialized — STT lazy, TTS pipeline ready, gemini_live=%s",
            self._is_gemini_live,
        )

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

    def set_resolved_stt(self, resolved: "ResolvedSttRuntime") -> None:
        """Configure STT runtime before first audio arrives (called by handler setup)."""
        self._resolved_stt = resolved

    @property
    def stt_event_bus(self) -> SttEventBus:
        return self._stt_event_bus

    def deactivate_stt(self) -> None:
        """Stop accepting audio.  Called as part of shutdown so Twilio silence
        frames arriving after call-end don't trigger a new Deepgram session."""
        self._stt_active = False

    def _start_livekit_audio_subscriber(self, handler) -> None:
        """Start the LiveKit audio subscriber task for Google STT path."""
        call_session_id = getattr(handler, "call_session_id", None)
        if not call_session_id:
            logger.warning("[VoiceOrchestrator] No call_session_id for LiveKit subscriber")
            return

        room_name = f"room_{call_session_id}"
        from app.voice.livekit_audio_subscriber import LiveKitAudioSubscriber

        sample_rate = (
            self._resolved_stt.sample_rate_hz if self._resolved_stt else 16000
        )
        subscriber = LiveKitAudioSubscriber(
            room_name=room_name,
            stt_pipeline=self._stt_pipeline,
            output_sample_rate=sample_rate,
        )
        self._livekit_audio_task = asyncio.create_task(subscriber.run())
        # Store subscriber for cleanup in shutdown
        self._livekit_subscriber = subscriber
        logger.info(
            "[VoiceOrchestrator] LiveKit audio subscriber started room=%s", room_name
        )

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

            # ── 3b. Native-audio (Gemini Live) fork ─────────────────────────────
            # Bypasses SttPipeline/TtsPipeline entirely for this call: caller
            # audio goes straight to a GeminiLiveSession, and Gemini's own
            # native audio response is streamed straight back (see
            # _feed_gemini_live_audio / _on_gemini_live_audio_chunk). The
            # pickup-detection/grace-period gating above is audio-format
            # agnostic (RMS over MULAW) and stays unchanged for both paths.
            if self._is_gemini_live:
                await self._feed_gemini_live_audio(h, audio_data)
                return

            # ── 4. STT feed ───────────────────────────────────────────────────
            if self._stt_pipeline is None:
                deferred_ep = self._stt_deferred_endpointing_ms
                self._stt_deferred_endpointing_ms = None
                initial_endpointing = (
                    int(deferred_ep)
                    if deferred_ep is not None
                    else _resolve_initial_endpointing_ms()
                )

                if self._resolved_stt is not None:
                    self._stt_pipeline = SttPipeline.from_runtime_config(
                        resolved=self._resolved_stt,
                        on_interim=self._on_interim,
                        on_final=self._on_final,
                        call_session_id=h.call_session_id,
                        agent_id=h.agent_id,
                        endpointing_ms=initial_endpointing,
                        event_bus=self._stt_event_bus,
                    )
                    # Start LiveKit audio subscriber for Google STT path
                    if (
                        self._resolved_stt.provider_slug == "google"
                        and settings.LIVEKIT_ENABLED
                    ):
                        self._start_livekit_audio_subscriber(h)
                else:
                    # Fallback: legacy Deepgram path (no resolved STT config)
                    language_code = (settings.DEEPGRAM_STT_LANGUAGE or "en").strip()
                    self._stt_pipeline = SttPipeline(
                        language_code=language_code,
                        on_interim=self._on_interim,
                        on_final=self._on_final,
                        call_session_id=h.call_session_id,
                        agent_id=h.agent_id,
                        endpointing_ms=initial_endpointing,
                        event_bus=self._stt_event_bus,
                    )

                ps = getattr(h, "_pipeline_session", None)
                if ps is not None:
                    ps.stt_emitter = self._stt_pipeline
                provider = (
                    self._resolved_stt.provider_slug if self._resolved_stt else "deepgram"
                )
                logger.debug(
                    "[VoiceOrchestrator] SttPipeline created provider=%s endpointing_ms=%s",
                    provider,
                    initial_endpointing,
                )

            # Only feed Twilio audio to non-Google paths (Google uses LiveKit)
            provider_slug = (
                self._resolved_stt.provider_slug if self._resolved_stt else "deepgram"
            )
            if provider_slug != "google":
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
        except Exception:  # noqa: S110 - defensive; Event.set() during shutdown must not block teardown
            pass

        # Shutdown TTS worker (drains queue, cancels worker task)
        try:
            await self._tts_pipeline.shutdown()
        except Exception as exc:
            logger.debug("[VoiceOrchestrator] TTS pipeline shutdown failed: %s", exc)

        # Stop LiveKit audio subscriber (Google STT path)
        lk_subscriber = getattr(self, "_livekit_subscriber", None)
        if lk_subscriber:
            try:
                await lk_subscriber.stop()
            except Exception as exc:
                logger.debug("[VoiceOrchestrator] LiveKit subscriber stop failed: %s", exc)
        lk_task = self._livekit_audio_task
        if lk_task and not lk_task.done():
            lk_task.cancel()
            try:
                await lk_task
            except (asyncio.CancelledError, Exception):  # noqa: S110 - expected from cancelling the task above
                pass
        self._livekit_audio_task = None

        # Close STT session (Deepgram: sends CloseStream, waits up to 5s; Google: stops stream)
        try:
            if self._stt_pipeline:
                await self._stt_pipeline.aclose()
        except Exception as exc:
            logger.debug("[VoiceOrchestrator] STT pipeline close failed: %s", exc)

        # Close Gemini Live session (native-audio calls only — no-op otherwise)
        try:
            if self._gemini_live_session is not None:
                self._gemini_live_cancel.set()
                await self._gemini_live_session.close()
                self._gemini_live_session = None
        except Exception as exc:
            logger.debug("[VoiceOrchestrator] Gemini Live session close failed: %s", exc)

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
        STT final result callback.
        1. Log with PII redaction for audit trail.
        2. Route to handler._process_transcript (dedup, goodbye, LLM trigger).
        """
        try:
            from app.core.pii_redactor import redact_pii
            redacted = redact_pii(transcript)
            logger.info(
                "[STT final] transcript=%r confidence=%.2f call_session_id=%s",
                redacted,
                confidence,
                self._h.call_session_id,
            )
        except Exception as exc:
            logger.debug("[VoiceOrchestrator] Failed to log redacted final transcript: %s", exc)

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

    # ── Gemini Live (native-audio speech-to-speech) ────────────────────────────

    async def _feed_gemini_live_audio(self, h, audio_data: bytes) -> None:
        """
        Route one MULAW/8kHz caller-audio frame to the (lazily created)
        GeminiLiveSession for this call, converting to PCM16/16kHz first.
        Called instead of the SttPipeline feed for native-audio agents.
        """
        if self._gemini_live_setup_failed:
            # Pre-session failure already handled (see _fallback_to_legacy_pipeline) —
            # self._is_gemini_live has been flipped to False, so this branch
            # should no longer be reached on the NEXT chunk, but guard anyway
            # in case a chunk was already in flight when the flip happened.
            return

        session = self._gemini_live_session
        if session is None:
            async with self._gemini_live_setup_lock:
                # Re-check inside the lock — a concurrent chunk may have
                # already started (or finished) setup while we awaited it.
                if self._gemini_live_session is None and not self._gemini_live_setup_failed:
                    await self._start_gemini_live_session(h)
            session = self._gemini_live_session

        if session is None:
            return  # setup failed and fallback (or shutdown) already handled it

        try:
            pcm16_16k = mulaw8k_to_pcm16_16k(audio_data)
            if pcm16_16k:
                await session.send_audio(pcm16_16k)
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] GeminiLiveSession.send_audio failed: %s", exc, exc_info=True
            )

    async def _start_gemini_live_session(self, h) -> None:
        """
        Lazily open the GeminiLiveSession for this call (once). Builds the
        initial system_instruction via each transport's own prompt-assembly
        method (reused, not duplicated) since there is no user utterance yet
        at session-start. On pre-session failure, falls back to the legacy
        STT/LLM/TTS pipeline for the remainder of the call (see
        _fallback_to_legacy_pipeline) rather than leaving the call silent.

        Transport-aware: the two call transports have deliberately separate,
        unshared prompt-assembly implementations (see CLAUDE.md's "Voice
        pipeline" section) —
        `BidirectionalStreamHandler.build_system_prompt` (Twilio, this
        handler's own method — includes booking_memory_block,
        inbound_kb_docs_context_block, recruitment-screening block,
        A/B prompt testing override, current-date/time + user-signals block,
        etc.) vs `ConversationOrchestrator.build_system_prompt` (LiveKit
        browser demo calls — a separate, simpler implementation).
        Duck-typed via `hasattr` rather than `isinstance` to avoid a
        circular import (bidirectional_stream.py already imports
        VoiceOrchestrator at module level).
        """
        from app.services.gemini_live_service import GeminiLiveSession
        from app.services.vertex_gemini_service import VertexLlmError

        agent = getattr(h, "agent", None)
        model_name = getattr(agent, "llm_model", None) if agent else None

        try:
            if hasattr(h, "build_system_prompt"):
                # Twilio (BidirectionalStreamHandler) — its own real
                # prompt-assembly method, not ConversationOrchestrator's.
                system_instruction = await h.build_system_prompt(
                    user_text="", confidence=1.0
                )
            else:
                # LiveKit browser demo calls (LiveKitBrowserCallHandler) —
                # genuinely uses ConversationOrchestrator.
                from app.voice.conversation_orchestrator import ConversationOrchestrator

                conv = ConversationOrchestrator(h)
                system_instruction = await conv.build_system_prompt(
                    user_text="", confidence=1.0
                )
        except Exception as exc:
            logger.error(
                "[GeminiLive] build_system_prompt failed for session start; using minimal "
                "fallback instruction: %s", exc, exc_info=True,
            )
            agent_name = getattr(agent, "name", None) or "AI Assistant"
            system_instruction = f"You are {agent_name}, a helpful phone assistant."

        session = GeminiLiveSession(model_name)
        try:
            await session.start(
                system_instruction=system_instruction,
                on_audio_chunk=self._on_gemini_live_audio_chunk,
                on_interrupted=self._on_gemini_live_interrupted,
                on_input_transcript=self._on_gemini_live_input_transcript,
                on_output_transcript=self._on_gemini_live_output_transcript,
                # Phase 1: Calendly/tool-calling is explicitly deferred (see
                # integration plan §6) — no tools, no on_tool_call callback.
                on_tool_call=None,
                on_error=self._on_gemini_live_error,
                tools=None,
            )
        except VertexLlmError as exc:
            logger.error(
                "[GeminiLive] pre-session start failed model=%s call_session_id=%s: %s",
                model_name, getattr(h, "call_session_id", None), exc,
            )
            await self._fallback_to_legacy_pipeline(h, reason=str(exc))
            return
        except Exception as exc:
            logger.error(
                "[GeminiLive] unexpected pre-session start failure model=%s call_session_id=%s: %s",
                model_name, getattr(h, "call_session_id", None), exc, exc_info=True,
            )
            await self._fallback_to_legacy_pipeline(h, reason=str(exc))
            return

        self._gemini_live_session = session
        logger.info(
            "[GeminiLive] session started model=%s call_session_id=%s",
            model_name, getattr(h, "call_session_id", None),
        )

    async def _fallback_to_legacy_pipeline(self, h, reason: str) -> None:
        """
        Pre-session GeminiLiveSession.start() failure fallback (see plan §7):
        swap this call's in-memory `agent.llm_model` to a default text model
        (native-audio model IDs have no meaning to vertex_gemini_service.py's
        text methods — never persisted to the DB, only mutates the Agent ORM
        instance already loaded for THIS call) and flip routing back to the
        existing STT/LLM/TTS pipeline. SttPipeline is already lazily created
        by on_audio_chunk's normal step 4 — nothing further to construct
        here; the next audio chunk takes that path automatically.

        Never called for a mid-call drop (session started successfully, then
        errored later) — see _on_gemini_live_error, which ends the call
        gracefully instead per the plan's mid-call-failure guidance.
        """
        self._gemini_live_setup_failed = True
        self._is_gemini_live = False

        agent = getattr(h, "agent", None)
        if agent is not None:
            logger.warning(
                "[GeminiLive] falling back to legacy pipeline for call_session_id=%s: "
                "%s -> %s (reason: %s)",
                getattr(h, "call_session_id", None),
                getattr(agent, "llm_model", None),
                _GEMINI_LIVE_FALLBACK_TEXT_MODEL,
                reason,
            )
            agent.llm_model = _GEMINI_LIVE_FALLBACK_TEXT_MODEL
        else:
            logger.error(
                "[GeminiLive] pre-session failure with no agent loaded for call_session_id=%s "
                "(reason: %s) — cannot fall back to a text pipeline; ending call.",
                getattr(h, "call_session_id", None), reason,
            )
            asyncio.create_task(h._full_shutdown())

    async def _on_gemini_live_audio_chunk(self, pcm16_24k_bytes: bytes) -> None:
        """
        GeminiLiveSession.on_audio_chunk callback. Gates on
        self._gemini_live_cancel (this path's barge-in flag — see
        _on_gemini_live_interrupted) exactly like the existing TTS send
        loops gate on handler._tts_cancel.

        Transport-aware (duck-typed via hasattr, same pattern as
        _start_gemini_live_session's system_instruction branch):
          - Twilio (BidirectionalStreamHandler, has `_stream_live_audio_chunk`):
            convert Gemini's raw PCM16/24kHz output to MULAW/8kHz and stream
            it to the Twilio WebSocket via that low-level frame-send
            primitive.
          - LiveKit browser (LiveKitBrowserCallHandler, no
            `_stream_live_audio_chunk` method): publish Gemini's raw
            PCM16/24kHz output directly into the room — that handler's
            `_agent_publisher` is opened at 24kHz for a native-audio call,
            so no mu-law conversion or resampling happens on this path at
            all (see LiveKitBrowserCallHandler._publish_gemini_live_audio_chunk).
        """
        if self._gemini_live_cancel.is_set():
            return
        try:
            if not self._gemini_live_first_audio_marked:
                self._gemini_live_first_audio_marked = True
                _vm = getattr(self._h, "_voice_metrics", None)
                if _vm:
                    _vm.mark_live_first_audio()

            if hasattr(self._h, "_stream_live_audio_chunk"):
                mulaw_bytes = pcm16_24k_to_mulaw8k(pcm16_24k_bytes)
                if not mulaw_bytes:
                    return
                await self._h._stream_live_audio_chunk(mulaw_bytes)
            else:
                await self._h._publish_gemini_live_audio_chunk(
                    pcm16_24k_bytes, self._gemini_live_cancel
                )
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live audio chunk send failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_interrupted(self) -> None:
        """
        Barge-in for the Gemini Live path: stop our own outbound send loop
        (self._gemini_live_cancel) AND tell the transport to flush anything
        it has already buffered — mirrors the existing two-part TTS
        barge-in, scoped to this path's own minimal cancel flag rather than
        the TtsPipeline/_tts_cancel machinery (nothing to cancel there — no
        TtsPipeline task exists for this call).

        Transport-aware (duck-typed via hasattr, mirrors
        _on_gemini_live_audio_chunk's branch): Twilio flushes via
        `_send_twilio_clear_event`; the LiveKit browser handler has no such
        method and instead clears its own LiveKit AudioSource's internal
        playout queue directly (see
        LiveKitBrowserCallHandler._clear_gemini_live_playout_queue).
        """
        try:
            self._gemini_live_cancel.set()
            if hasattr(self._h, "_send_twilio_clear_event"):
                await self._h._send_twilio_clear_event()
            else:
                await self._h._clear_gemini_live_playout_queue()
            # Reset for the next turn's outbound audio (mirrors _tts_cancel.clear()
            # in ConversationOrchestrator.generate_and_stream_response).
            self._gemini_live_cancel.clear()
            self._gemini_live_first_audio_marked = False
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live interrupted handling failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_input_transcript(self, text: str, finished: bool) -> None:
        """Caller-side transcript from Gemini's own input_transcription. Only
        written to call_transcript on finished chunks — mirrors the existing
        STT-final (not interim) transcript-write precedent."""
        if not finished or not (text or "").strip():
            return
        try:
            await self._h._add_to_transcript("client", text.strip(), "speech")
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live input transcript write failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_output_transcript(self, text: str, finished: bool) -> None:
        """Agent-side transcript from Gemini's own output_transcription. Only
        written on finished chunks so call_transcript gets one clean line per
        agent turn, matching the existing agent_response convention."""
        if not finished or not (text or "").strip():
            return
        try:
            await self._h._add_to_transcript("agent", text.strip(), "agent_response")
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live output transcript write failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_error(self, err) -> None:
        """
        Mid-call GeminiLiveSession failure (session had already started —
        pre-session failures are handled by _start_gemini_live_session's own
        try/except, never reach here). Per the plan: do NOT attempt to
        reconstruct STT/TTS mid-call (too much state/latency risk) — end the
        call gracefully via the existing _full_shutdown() path.

        # TODO(gemini-live-fallback): a static pre-recorded "sorry, goodbye"
        # audio clip played before hangup would be a friendlier mid-call
        # failure UX than a silent drop. Deferred per the integration plan
        # (Phase 1 accepts a graceful-but-abrupt hangup here, clearly logged).
        """
        logger.error(
            "[GeminiLive] mid-call session error call_session_id=%s error_type=%s: %s",
            getattr(self._h, "call_session_id", None),
            getattr(err, "error_type", None),
            err,
        )
        asyncio.create_task(self._h._full_shutdown())

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
