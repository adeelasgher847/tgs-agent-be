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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Set

from app.core.config import settings
from app.core.llm_models import is_gemini_live_native_audio_model, is_openai_realtime_model
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

# Same idea as _GEMINI_LIVE_FALLBACK_TEXT_MODEL above, for OpenAI Realtime
# ("gpt-realtime"/"gpt-realtime-2") pre-session failures — see
# _start_openai_realtime_session / _fallback_to_legacy_pipeline_openai. Kept
# in the same provider family (OpenAI) rather than falling back cross-
# provider to a Gemini text model, since the tenant/agent is already
# configured for an OpenAI API key.
_OPENAI_REALTIME_FALLBACK_TEXT_MODEL = "gpt-4o-mini"


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
        # Per-call accumulation buffers for input/output transcript fragments
        # (see _on_gemini_live_input_transcript / _on_gemini_live_output_transcript).
        # The Live API's per-fragment `finished` flag is NOT a reliable
        # turn-boundary signal for the native-audio model family (may never
        # fire, or fire on an incremental fragment rather than the full
        # utterance) — fragments are buffered here and flushed to
        # call_transcript on the documented workaround signals instead (see
        # _flush_gemini_live_input_buffer / _flush_gemini_live_output_buffer).
        # Reset alongside self._gemini_live_session wherever that is
        # (re)created, so no leftover text can bleed into a later call/turn.
        self._gemini_live_input_buffer: list[str] = []
        self._gemini_live_output_buffer: list[str] = []
        # Coalescing guard for _refresh_gemini_live_kb_context: a chatty
        # back-and-forth call could otherwise fire several overlapping KB
        # retrievals + un-ordered session.send_text(...) calls in quick
        # succession. At most one refresh runs at a time per call; a finished
        # transcript arriving while one is still in flight is skipped rather
        # than queued, since it will be superseded by whatever the caller
        # says next anyway.
        self._gemini_live_kb_refresh_in_flight: bool = False

        # ── OpenAI Realtime (native-audio speech-to-speech) state ─────────────
        # Parallel to the Gemini Live block above, not a shared/refactored
        # implementation (per this task's "strictly additive, don't touch
        # Gemini-Live-specific code" constraint) — see
        # app/services/openai_realtime_service.py's module docstring for the
        # deliberate behavioral divergences (no per-turn receive loop, no
        # transcript-fragment buffering, mostly-server-driven barge-in).
        self._is_openai_realtime: bool = is_openai_realtime_model(
            getattr(_agent, "llm_model", None) if _agent else None
        )
        self._openai_realtime_session: Any = None
        self._openai_realtime_setup_lock: asyncio.Lock = asyncio.Lock()
        self._openai_realtime_setup_failed: bool = False
        self._openai_realtime_cancel: asyncio.Event = asyncio.Event()
        self._openai_realtime_first_audio_marked: bool = False
        # No input/output transcript buffers needed here — unlike Gemini
        # Live, OpenAI's transcription events each fire exactly once per
        # utterance with the full final text (see
        # openai_realtime_service.OpenAIRealtimeSession's on_input_transcript/
        # on_output_transcript contract).
        self._openai_realtime_kb_refresh_in_flight: bool = False

        logger.info(
            "[VoiceOrchestrator] Initialized — STT lazy, TTS pipeline ready, "
            "gemini_live=%s openai_realtime=%s",
            self._is_gemini_live, self._is_openai_realtime,
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

            # ── 3c. Native-audio (OpenAI Realtime) fork ─────────────────────────
            # Same idea as 3b above, for OpenAI's Realtime speech-to-speech
            # models. Twilio's own MULAW/8kHz wire format is sent straight
            # through with NO conversion (see _feed_openai_realtime_audio /
            # app.services.openai_realtime_service's module docstring) —
            # unlike Gemini Live, which forces a PCM16 resample round-trip.
            if self._is_openai_realtime:
                await self._feed_openai_realtime_audio(h, audio_data)
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

        # Close Gemini Live session (native-audio calls only — no-op otherwise).
        # close() cancels and awaits the receive-loop task before returning, so
        # closing FIRST guarantees no more fragments can be appended to the
        # buffers by a still-running receive loop; only then is it safe to
        # flush once. Flushing before close() left a race where an in-flight
        # receive-loop message (scheduled during the flush's own awaits) could
        # land in a buffer that was just cleared and never get written —
        # exactly the "call ending mid-buffer" case this flush exists to cover.
        try:
            if self._gemini_live_session is not None:
                self._gemini_live_cancel.set()
                await self._gemini_live_session.close()
                self._gemini_live_session = None
                await self._flush_gemini_live_input_buffer()
                await self._flush_gemini_live_output_buffer()
        except Exception as exc:
            logger.debug("[VoiceOrchestrator] Gemini Live session close failed: %s", exc)

        # Close OpenAI Realtime session (native-audio calls only — no-op
        # otherwise). No buffer-flush step needed here (unlike the Gemini
        # Live block above) — OpenAI's transcripts are written to
        # call_transcript synchronously as each complete utterance arrives,
        # never buffered.
        try:
            if self._openai_realtime_session is not None:
                self._openai_realtime_cancel.set()
                await self._openai_realtime_session.close()
                self._openai_realtime_session = None
        except Exception as exc:
            logger.debug("[VoiceOrchestrator] OpenAI Realtime session close failed: %s", exc)

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
        from app.services.gemini_live_service import GeminiLiveSession, resolve_voice_name
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

        # Calendly tool-calling (Phase 2): only meaningful on Twilio
        # (BookingMixin/_execute_calendly_tool_call/_calendly_enabled only
        # exist on BidirectionalStreamHandler — hasattr naturally gates the
        # browser transport out, which has no Calendly wiring at all).
        gemini_tools = None
        on_tool_call = None
        if hasattr(h, "_calendly_enabled") and h._calendly_enabled():
            from app.services.vertex_gemini_service import _build_calendly_tool

            gemini_tools = [_build_calendly_tool()]
            on_tool_call = self._on_gemini_live_tool_call

        # Per-agent native voice (Phase 2): reuses the agent's EXISTING
        # tts_voice_external_id / tts_language fields rather than adding new
        # schema — an agent switched to a native-audio model doesn't lose its
        # voice identity outright, though the ID space is different (Gemini's
        # 30 prebuilt voice names vs. ElevenLabs/Rime/Google voice IDs), so
        # resolve_voice_name() falls back to DEFAULT_VOICE_NAME for anything
        # that isn't a real Gemini Live voice name (the common case, since
        # most agents' configured voice ID predates being switched to a
        # native-audio model).
        voice_name = resolve_voice_name(getattr(agent, "tts_voice_external_id", None))
        language_code = (getattr(agent, "tts_language", None) or "").strip() or None

        session = GeminiLiveSession(model_name)
        try:
            await session.start(
                system_instruction=system_instruction,
                voice_name=voice_name,
                language_code=language_code,
                on_audio_chunk=self._on_gemini_live_audio_chunk,
                on_interrupted=self._on_gemini_live_interrupted,
                on_input_transcript=self._on_gemini_live_input_transcript,
                on_output_transcript=self._on_gemini_live_output_transcript,
                on_turn_complete=self._on_gemini_live_turn_complete,
                on_tool_call=on_tool_call,
                on_error=self._on_gemini_live_error,
                tools=gemini_tools,
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
        # Reset transcript buffers alongside the session itself so no
        # leftover fragments from a prior (re)start could bleed in — mirrors
        # __init__'s initial reset.
        self._gemini_live_input_buffer = []
        self._gemini_live_output_buffer = []
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
            # Barge-in cuts a turn off mid-flight, but whatever was
            # accumulated in either buffer up to this point is real speech
            # that actually happened (the caller's interruption, and/or the
            # agent's partial response before being cut off) — flush rather
            # than silently drop it.
            await self._flush_gemini_live_input_buffer()
            await self._flush_gemini_live_output_buffer()
            # Reset for the next turn's outbound audio (mirrors _tts_cancel.clear()
            # in ConversationOrchestrator.generate_and_stream_response).
            self._gemini_live_cancel.clear()
            self._gemini_live_first_audio_marked = False
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live interrupted handling failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_input_transcript(self, text: str, finished: bool) -> None:
        """
        Caller-side transcript fragment from Gemini's own input_transcription.

        The Live API's per-fragment ``finished`` flag is NOT a reliable
        turn-boundary signal for the native-audio model family — it may
        never fire at all, and even when it does, ``text`` is itself an
        incremental fragment (e.g. " Ho", " you") rather than the full
        utterance (confirmed against googleapis/js-genai#1429 and Google's
        own forum reports). So every fragment is unconditionally appended to
        a per-call buffer here; the accumulated buffer is written to
        call_transcript later by _flush_gemini_live_input_buffer(), triggered
        by the first output-transcript fragment of the next turn (implicit
        "caller finished speaking" signal) or by turn_complete/interrupted/
        shutdown as safety nets — see those call sites.
        """
        text = text or ""
        if text:
            self._gemini_live_input_buffer.append(text)

    async def _flush_gemini_live_input_buffer(self) -> None:
        """Write the accumulated input-transcript buffer to call_transcript
        and clear it. Also fires the mid-call RAG refresh off the same
        flushed text (same signal as before, just correctly assembled now
        instead of a single stray fragment)."""
        if not self._gemini_live_input_buffer:
            return
        transcript = "".join(self._gemini_live_input_buffer).strip()
        self._gemini_live_input_buffer = []
        if not transcript:
            return
        try:
            await self._h._add_to_transcript("client", transcript, "speech")
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live input transcript write failed: %s", exc, exc_info=True
            )

        # Mid-call RAG refresh (Phase 2, plan §6): the flushed input
        # transcript is the closest thing this session has to a per-turn
        # boundary (no STT-interim prefetch signal exists here).
        # Fire-and-forget — must never block the receive loop or delay the
        # caller's live audio response; tracked in the same set
        # _full_shutdown already drains so a call ending mid-refresh doesn't
        # leak the task. Coalesced via _gemini_live_kb_refresh_in_flight:
        # skip rather than queue a new refresh while one is still running, so
        # a chatty back-and-forth can't fire several overlapping retrievals +
        # unordered session.send_text(...) calls racing each other.
        if not self._gemini_live_kb_refresh_in_flight:
            t = asyncio.create_task(self._refresh_gemini_live_kb_context(transcript))
            self._pending_final_tasks.add(t)
            t.add_done_callback(lambda done_t, s=self: s._pending_final_tasks.discard(done_t))

    async def _refresh_gemini_live_kb_context(self, transcript: str) -> None:
        """
        Re-query the call flow's attached knowledge bases against the
        caller's latest finished utterance and, if anything comes back, feed
        it to the live session as a non-blocking text turn — so a native-
        audio call's RAG grounding can adapt as the topic shifts, instead of
        staying frozen at whatever was known at session start (see
        _start_gemini_live_session's system_instruction, built once with an
        empty user_text). Fails open on any error, same as every other RAG
        call site in this codebase — KB context is a grounding aid, never a
        call-blocking dependency.
        """
        session = self._gemini_live_session
        if session is None or not transcript:
            return
        flow = getattr(self._h, "call_flow", None)
        kb_ids = (flow.knowledge_base_ids or []) if flow else []
        if not kb_ids:
            return
        self._gemini_live_kb_refresh_in_flight = True
        try:
            from app.services.kb_retrieval_service import retrieve_kb_context_for_turn
            from app.utils.redis_client import get_redis

            kb_context, latency_ms = await retrieve_kb_context_for_turn(
                transcript=transcript, kb_ids=kb_ids, redis_client=get_redis()
            )
            if not kb_context:
                return
            # turn_complete=False verified live (2026-08-17, against
            # gemini-live-2.5-flash-native-audio): a context-only turn left
            # open like this is correctly picked up by the model's next real
            # turn without needing an explicit closing call — the caller's
            # own ongoing audio (a separate send_realtime_input channel,
            # VAD-driven turn-taking) is what actually closes/advances turns
            # in production, not this side-channel text injection.
            await session.send_text(
                "[KNOWLEDGE BASE CONTEXT UPDATE — authoritative, use if relevant to the "
                f"caller's most recent question]\n{kb_context}",
                turn_complete=False,
            )
            logger.debug(
                "[GeminiLive] kb_refresh latency_ms=%.1f chars=%d", latency_ms, len(kb_context)
            )
        except Exception as exc:
            logger.debug("[GeminiLive] mid-call KB refresh failed (non-fatal): %s", exc)
        finally:
            self._gemini_live_kb_refresh_in_flight = False

    async def _on_gemini_live_output_transcript(self, text: str, finished: bool) -> None:
        """
        Agent-side transcript fragment from Gemini's own output_transcription.
        Same unreliable-``finished``/fragment-not-full-utterance caveat as
        _on_gemini_live_input_transcript applies here — every fragment is
        buffered rather than gated on ``finished``.

        The first output fragment of a turn is also the documented signal
        that the caller has finished speaking (Gemini has started
        responding), so it flushes any pending input buffer before appending
        itself to the output buffer. The output buffer itself is flushed on
        turn_complete (see _on_gemini_live_turn_complete) so call_transcript
        gets one clean accumulated line per agent turn, matching the
        existing agent_response convention.
        """
        text = text or ""
        if not text:
            return
        if self._gemini_live_input_buffer:
            await self._flush_gemini_live_input_buffer()
        self._gemini_live_output_buffer.append(text)

    async def _flush_gemini_live_output_buffer(self) -> None:
        """Write the accumulated output-transcript buffer to call_transcript
        and clear it."""
        if not self._gemini_live_output_buffer:
            return
        transcript = "".join(self._gemini_live_output_buffer).strip()
        self._gemini_live_output_buffer = []
        if not transcript:
            return
        try:
            await self._h._add_to_transcript("agent", transcript, "agent_response")
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live output transcript write failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_turn_complete(self) -> None:
        """
        GeminiLiveSession.on_turn_complete callback — the only reliable
        turn-boundary signal for these models (see
        _on_gemini_live_input_transcript's docstring). Flushes the output
        buffer (the normal, expected flush point for an agent turn) and, as
        a safety net, the input buffer too — covers turns that never produce
        an output-transcript fragment at all (e.g. a tool-call-only turn),
        which would otherwise leave a caller utterance buffered forever.
        """
        try:
            await self._flush_gemini_live_input_buffer()
            await self._flush_gemini_live_output_buffer()
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] gemini live turn_complete flush failed: %s", exc, exc_info=True
            )

    async def _on_gemini_live_tool_call(self, function_calls) -> None:
        """
        Calendly tool-calling for a Gemini Live session (Phase 2). Only ever
        registered when the handler is Twilio's BidirectionalStreamHandler
        AND the call flow has Calendly enabled (see _start_gemini_live_session)
        — BookingMixin/_execute_calendly_tool_call don't exist on the browser
        handler, so this method is simply never reached on that transport.

        Unlike the existing non-live generate_with_tools() request/response
        tool-calling (one function_call, one response, done), the Live API's
        tool-calling is async-within-session: function_calls can arrive at
        any point mid-conversation, potentially more than one per message,
        and the caller audio stream keeps flowing while we resolve them —
        there is no "pause the turn" concept to lean on. Each call is
        resolved independently and its response sent back via
        send_tool_response(); _execute_calendly_tool_call already never
        raises (always returns a JSON-serializable dict, {"error": ...} on
        failure), so a single bad tool call can't crash the session or block
        the others.
        """
        from google.genai import types

        session = self._gemini_live_session
        if session is None:
            return

        function_responses = []
        for fc in function_calls:
            name = getattr(fc, "name", None) or ""
            args = dict(getattr(fc, "args", None) or {})
            call_id = getattr(fc, "id", None)
            try:
                result = await self._h._execute_calendly_tool_call(name, args)
            except Exception as exc:
                logger.error(
                    "[GeminiLive] Calendly tool call %s failed unexpectedly: %s",
                    name, exc, exc_info=True,
                )
                result = {"error": "internal error executing tool call"}
            from app.core.pii_redactor import redact_pii

            logger.info("[GeminiLive] tool_call name=%s args=%s", name, redact_pii(args))
            function_responses.append(
                types.FunctionResponse(id=call_id, name=name, response=result)
            )

        if function_responses:
            try:
                await session.send_tool_response(function_responses=function_responses)
            except Exception as exc:
                logger.error(
                    "[GeminiLive] send_tool_response failed: %s", exc, exc_info=True
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

    # ── OpenAI Realtime (native-audio speech-to-speech) ─────────────────────────
    # Parallel to the "Gemini Live" section above — see
    # app/services/openai_realtime_service.py's module docstring for the
    # deliberate behavioral divergences from Gemini Live (no per-turn
    # receive loop, no transcript-fragment buffering, mostly-server-driven
    # barge-in via turn_detection.interrupt_response).

    async def _feed_openai_realtime_audio(self, h, audio_data: bytes) -> None:
        """
        Route one MULAW/8kHz Twilio caller-audio frame to the (lazily
        created) OpenAIRealtimeSession for this call. Called instead of the
        SttPipeline feed for native-audio OpenAI agents. Unlike
        _feed_gemini_live_audio, no format conversion happens here — the
        session is started with audio_format="audio/pcmu" for a Twilio
        call, which OpenAI accepts natively (see
        openai_realtime_service.TWILIO_AUDIO_FORMAT).
        """
        if self._openai_realtime_setup_failed:
            return

        session = self._openai_realtime_session
        if session is None:
            async with self._openai_realtime_setup_lock:
                if self._openai_realtime_session is None and not self._openai_realtime_setup_failed:
                    await self._start_openai_realtime_session(h)
            session = self._openai_realtime_session

        if session is None:
            return  # setup failed and fallback (or shutdown) already handled it

        try:
            await session.send_audio(audio_data)
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] OpenAIRealtimeSession.send_audio failed: %s", exc, exc_info=True
            )

    async def _start_openai_realtime_session(self, h) -> None:
        """
        Lazily open the OpenAIRealtimeSession for this call (once). Mirrors
        _start_gemini_live_session's transport-aware system_instruction /
        Calendly-tool wiring exactly — only the session class, audio format,
        and voice-resolution helper differ.
        """
        from app.services.openai_realtime_service import (
            OpenAIRealtimeError,
            OpenAIRealtimeSession,
            LIVEKIT_AUDIO_FORMAT,
            TWILIO_AUDIO_FORMAT,
            resolve_voice_name as resolve_openai_voice_name,
        )

        agent = getattr(h, "agent", None)
        model_name = getattr(agent, "llm_model", None) if agent else None
        is_twilio_transport = hasattr(h, "build_system_prompt")

        try:
            if is_twilio_transport:
                system_instruction = await h.build_system_prompt(user_text="", confidence=1.0)
            else:
                from app.voice.conversation_orchestrator import ConversationOrchestrator

                conv = ConversationOrchestrator(h)
                system_instruction = await conv.build_system_prompt(user_text="", confidence=1.0)
        except Exception as exc:
            logger.error(
                "[OpenAIRealtime] build_system_prompt failed for session start; using minimal "
                "fallback instruction: %s", exc, exc_info=True,
            )
            agent_name = getattr(agent, "name", None) or "AI Assistant"
            system_instruction = f"You are {agent_name}, a helpful phone assistant."

        openai_tools = None
        on_tool_call = None
        if hasattr(h, "_calendly_enabled") and h._calendly_enabled():
            from app.services.openai_realtime_service import _build_calendly_tool_params

            openai_tools = _build_calendly_tool_params()
            on_tool_call = self._on_openai_realtime_tool_call

        voice_name = resolve_openai_voice_name(getattr(agent, "tts_voice_external_id", None))
        audio_format = TWILIO_AUDIO_FORMAT if is_twilio_transport else LIVEKIT_AUDIO_FORMAT

        session = OpenAIRealtimeSession(model_name)
        try:
            await session.start(
                system_instruction=system_instruction,
                audio_format=audio_format,
                voice_name=voice_name,
                on_audio_chunk=self._on_openai_realtime_audio_chunk,
                on_interrupted=self._on_openai_realtime_interrupted,
                on_input_transcript=self._on_openai_realtime_input_transcript,
                on_output_transcript=self._on_openai_realtime_output_transcript,
                on_tool_call=on_tool_call,
                on_error=self._on_openai_realtime_error,
                tools=openai_tools,
            )
        except OpenAIRealtimeError as exc:
            logger.error(
                "[OpenAIRealtime] pre-session start failed model=%s call_session_id=%s: %s",
                model_name, getattr(h, "call_session_id", None), exc,
            )
            await self._fallback_to_legacy_pipeline_openai(h, reason=str(exc))
            return
        except Exception as exc:
            logger.error(
                "[OpenAIRealtime] unexpected pre-session start failure model=%s call_session_id=%s: %s",
                model_name, getattr(h, "call_session_id", None), exc, exc_info=True,
            )
            await self._fallback_to_legacy_pipeline_openai(h, reason=str(exc))
            return

        self._openai_realtime_session = session
        logger.info(
            "[OpenAIRealtime] session started model=%s call_session_id=%s",
            model_name, getattr(h, "call_session_id", None),
        )

    async def _fallback_to_legacy_pipeline_openai(self, h, reason: str) -> None:
        """Pre-session OpenAIRealtimeSession.start() failure fallback — mirrors
        _fallback_to_legacy_pipeline (Gemini Live) exactly, swapping in the
        OpenAI-family fallback text model instead."""
        self._openai_realtime_setup_failed = True
        self._is_openai_realtime = False

        agent = getattr(h, "agent", None)
        if agent is not None:
            logger.warning(
                "[OpenAIRealtime] falling back to legacy pipeline for call_session_id=%s: "
                "%s -> %s (reason: %s)",
                getattr(h, "call_session_id", None),
                getattr(agent, "llm_model", None),
                _OPENAI_REALTIME_FALLBACK_TEXT_MODEL,
                reason,
            )
            agent.llm_model = _OPENAI_REALTIME_FALLBACK_TEXT_MODEL

            # Persist the fallback on CallSession.call_metadata so
            # credit_service's out-of-process billing tick (which re-queries
            # CallSession every 10s, never holds a reference to this
            # orchestrator instance) can stop billing the OpenAI Realtime
            # surcharge and start evaluating the ElevenLabs TTS surcharge for
            # the remainder of the call — see
            # CreditService.get_active_surcharges(realtime_fallen_back=...).
            # Mirrors the existing call_metadata write pattern in
            # CallControlMixin._transfer_call (human_transfer).
            try:
                call_session = getattr(h, "call_session", None)
                db = getattr(h, "db", None)
                if call_session is not None and db is not None:
                    meta = dict(call_session.call_metadata or {})
                    meta["openai_realtime_fallback"] = {
                        "fell_back": True,
                        "reason": reason,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                    call_session.call_metadata = meta
                    db.commit()
                    db.refresh(call_session)
            except Exception as exc:
                logger.error(
                    "[OpenAIRealtime] failed to persist realtime-fallback flag on "
                    "call_metadata for call_session_id=%s (credit billing may keep "
                    "charging the realtime surcharge for this call): %s",
                    getattr(h, "call_session_id", None), exc, exc_info=True,
                )
        else:
            logger.error(
                "[OpenAIRealtime] pre-session failure with no agent loaded for call_session_id=%s "
                "(reason: %s) — cannot fall back to a text pipeline; ending call.",
                getattr(h, "call_session_id", None), reason,
            )
            asyncio.create_task(h._full_shutdown())

    async def _on_openai_realtime_audio_chunk(self, audio_bytes: bytes) -> None:
        """
        OpenAIRealtimeSession.on_audio_chunk callback. Gates on
        self._openai_realtime_cancel exactly like _on_gemini_live_audio_chunk
        gates on self._gemini_live_cancel.

        Transport-aware (duck-typed via hasattr, same pattern as
        _on_gemini_live_audio_chunk):
          - Twilio: `audio_bytes` is already mu-law/8kHz (the session was
            started with audio_format="audio/pcmu") — streamed straight to
            `_stream_live_audio_chunk` with NO conversion step, unlike
            Gemini Live's pcm16_24k_to_mulaw8k.
          - LiveKit browser: `audio_bytes` is raw PCM16/24kHz (audio_format=
            "audio/pcm") — the exact format/rate
            `_publish_gemini_live_audio_chunk` (and the publisher it
            targets) already expects for Gemini Live, so that existing sink
            is reused as-is (see its own docstring — it's a generic
            "publish raw PCM16 at the publisher's rate" primitive despite
            the Gemini-specific name).
        """
        if self._openai_realtime_cancel.is_set():
            return
        try:
            if not self._openai_realtime_first_audio_marked:
                self._openai_realtime_first_audio_marked = True
                _vm = getattr(self._h, "_voice_metrics", None)
                if _vm:
                    _vm.mark_live_first_audio()

            if hasattr(self._h, "_stream_live_audio_chunk"):
                await self._h._stream_live_audio_chunk(audio_bytes)
            else:
                from app.services.openai_realtime_service import LIVEKIT_AUDIO_RATE_HZ

                await self._h._publish_gemini_live_audio_chunk(
                    audio_bytes, self._openai_realtime_cancel,
                    sample_rate_hz=LIVEKIT_AUDIO_RATE_HZ,
                )
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] openai realtime audio chunk send failed: %s", exc, exc_info=True
            )

    async def _on_openai_realtime_interrupted(self) -> None:
        """
        Barge-in for the OpenAI Realtime path. Server-side, OpenAI already
        cancels its own in-flight response when `interrupt_response=True`
        (session config) detects caller speech — this callback only handles
        the LOCAL half: stop our own outbound send loop and flush whatever
        this transport has already buffered/sent. Mirrors
        _on_gemini_live_interrupted's transport-aware dispatch exactly.
        """
        try:
            self._openai_realtime_cancel.set()
            if hasattr(self._h, "_send_twilio_clear_event"):
                await self._h._send_twilio_clear_event()
            else:
                await self._h._clear_gemini_live_playout_queue()
            self._openai_realtime_cancel.clear()
            self._openai_realtime_first_audio_marked = False
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] openai realtime interrupted handling failed: %s", exc, exc_info=True
            )

    async def _on_openai_realtime_input_transcript(self, text: str) -> None:
        """
        Caller-side FULL final transcript for one utterance — fires exactly
        once per utterance (see OpenAIRealtimeSession's
        on_input_transcript contract). No buffering/flush-signal dance
        needed here, unlike _on_gemini_live_input_transcript.
        """
        text = (text or "").strip()
        if not text:
            return
        try:
            await self._h._add_to_transcript("client", text, "speech")
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] openai realtime input transcript write failed: %s", exc, exc_info=True
            )

        # Mid-call RAG refresh (mirrors _flush_gemini_live_input_buffer's
        # fire-and-forget trigger, keyed here directly off the complete
        # utterance rather than off a buffer flush — there is no buffer to
        # flush on this path). Coalesced via
        # _openai_realtime_kb_refresh_in_flight for the same reason Gemini
        # Live's refresh is coalesced: a chatty back-and-forth shouldn't
        # fire several overlapping retrievals + unordered send_text() calls.
        if not self._openai_realtime_kb_refresh_in_flight:
            t = asyncio.create_task(self._refresh_openai_realtime_kb_context(text))
            self._pending_final_tasks.add(t)
            t.add_done_callback(lambda done_t, s=self: s._pending_final_tasks.discard(done_t))

    async def _refresh_openai_realtime_kb_context(self, transcript: str) -> None:
        """
        OpenAI Realtime analog of _refresh_gemini_live_kb_context. Sends the
        refreshed KB context as a `conversation.item.create` with
        `respond=False` — a context-only injection must NEVER trigger
        `response.create`, or OpenAI will speak the raw injected context
        aloud as if it were a real reply (see
        OpenAIRealtimeSession.send_text's docstring). Fails open on any
        error, same as every other RAG call site in this codebase.
        """
        # DIAGNOSTIC: temporary tracing for a "agent doesn't use KB" report —
        # logs entry/skip reasons and whether context was actually sent, but
        # never the transcript or KB content itself. Safe to strip once KB
        # refresh is confirmed working on a real call.
        call_session_id = getattr(self._h, "call_session_id", None)
        session = self._openai_realtime_session
        if session is None or not transcript:
            logger.info(
                "[OpenAIRealtime] DIAGNOSTIC kb_refresh skipped call_session_id=%s reason=%s",
                call_session_id, "no_session" if session is None else "empty_transcript",
            )
            return
        flow = getattr(self._h, "call_flow", None)
        kb_ids = (flow.knowledge_base_ids or []) if flow else []
        if not kb_ids:
            logger.info(
                "[OpenAIRealtime] DIAGNOSTIC kb_refresh skipped call_session_id=%s reason=%s "
                "flow_present=%s",
                call_session_id, "no_kb_ids_configured", flow is not None,
            )
            return
        self._openai_realtime_kb_refresh_in_flight = True
        try:
            from app.services.kb_retrieval_service import retrieve_kb_context_for_turn
            from app.utils.redis_client import get_redis

            kb_context, latency_ms = await retrieve_kb_context_for_turn(
                transcript=transcript, kb_ids=kb_ids, redis_client=get_redis()
            )
            if not kb_context:
                logger.info(
                    "[OpenAIRealtime] DIAGNOSTIC kb_refresh call_session_id=%s: retrieval returned "
                    "no matching context (kb_ids=%s, latency_ms=%.1f) — nothing sent",
                    call_session_id, kb_ids, latency_ms,
                )
                return
            await session.send_text(
                "[KNOWLEDGE BASE CONTEXT UPDATE — authoritative, use if relevant to the "
                f"caller's most recent question]\n{kb_context}",
                respond=False,
            )
            logger.info(
                "[OpenAIRealtime] DIAGNOSTIC kb_refresh call_session_id=%s: sent kb context "
                "latency_ms=%.1f chars=%d",
                call_session_id, latency_ms, len(kb_context),
            )
        except Exception as exc:
            logger.warning(
                "[OpenAIRealtime] mid-call KB refresh failed (non-fatal) call_session_id=%s: %s",
                call_session_id, exc, exc_info=True,
            )
        finally:
            self._openai_realtime_kb_refresh_in_flight = False

    async def _on_openai_realtime_output_transcript(self, text: str) -> None:
        """Agent-side FULL final transcript for one turn — fires exactly
        once (see OpenAIRealtimeSession's on_output_transcript contract).
        No buffering/turn_complete flush needed, unlike
        _on_gemini_live_output_transcript / _on_gemini_live_turn_complete."""
        text = (text or "").strip()
        if not text:
            return
        try:
            await self._h._add_to_transcript("agent", text, "agent_response")
        except Exception as exc:
            logger.error(
                "[VoiceOrchestrator] openai realtime output transcript write failed: %s", exc, exc_info=True
            )

    async def _on_openai_realtime_tool_call(self, call_id: str, name: str, arguments_json: str) -> None:
        """
        Calendly tool-calling for an OpenAI Realtime session. Only ever
        registered when the handler is Twilio's BidirectionalStreamHandler
        AND the call flow has Calendly enabled (see
        _start_openai_realtime_session) — mirrors
        _on_gemini_live_tool_call's gating and reuses
        BookingMixin._execute_calendly_tool_call unmodified, same as Gemini.
        """
        import json as _json

        session = self._openai_realtime_session
        if session is None:
            return

        try:
            args = _json.loads(arguments_json) if arguments_json else {}
            if not isinstance(args, dict):
                args = {}
        except (ValueError, TypeError):
            args = {}

        try:
            result = await self._h._execute_calendly_tool_call(name, args)
        except Exception as exc:
            logger.error(
                "[OpenAIRealtime] Calendly tool call %s failed unexpectedly: %s",
                name, exc, exc_info=True,
            )
            result = {"error": "internal error executing tool call"}

        from app.core.pii_redactor import redact_pii

        logger.info("[OpenAIRealtime] tool_call name=%s args=%s", name, redact_pii(args))
        try:
            await session.send_tool_response(call_id, result)
        except Exception as exc:
            logger.error("[OpenAIRealtime] send_tool_response failed: %s", exc, exc_info=True)

    async def _on_openai_realtime_error(self, err) -> None:
        """
        Mid-call OpenAIRealtimeSession failure (session had already started —
        pre-session failures are handled by _start_openai_realtime_session's
        own try/except, never reach here). Mirrors _on_gemini_live_error:
        end the call gracefully rather than attempting to reconstruct
        STT/TTS mid-call.
        """
        logger.error(
            "[OpenAIRealtime] mid-call session error call_session_id=%s error_type=%s: %s",
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
