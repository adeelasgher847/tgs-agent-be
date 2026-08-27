"""
LiveKitBrowserCallHandler — browser-only (no Twilio) LiveKit voice agent.

Powers the "Share Demo Link" feature: a visitor gets a LiveKit token from
POST /api/v1/sdk/demo/{token}/call-token (app/routers/sdk.py::demo_call_token)
and joins a LiveKit room directly from the browser (WebRTC mic in / speaker
out — no phone, no Twilio Media Streams).

This module is what makes the agent actually join that room and hold a real
conversation. It is a *parallel* implementation to
app/routers/bidirectional_stream.py::BidirectionalStreamHandler — it does not
modify, subclass, or import from that file. It exists because the Twilio
handler's audio I/O is fundamentally MULAW-over-WebSocket while this path is
LiveKit-native PCM-over-WebRTC.

Reused unmodified:
  - app.voice.stt_pipeline.SttPipeline            (provider-agnostic STT)
  - app.voice.livekit_audio_subscriber.LiveKitAudioSubscriber (caller audio in)
  - app.voice.voice_orchestrator.VoiceOrchestrator (TTS pipeline ownership +
    unified shutdown — see _run_livekit_browser_call for exactly how it's
    wired up without going through its Twilio-specific on_audio_chunk())
  - app.voice.tts_pipeline.TtsPipeline             (parallel TTS synthesis)
  - app.voice.conversation_orchestrator.ConversationOrchestrator (prompt
    building incl. call-flow prompt override, KB/RAG context retrieval, CRM
    read-only context blocks, LLM streaming — the actual "brain" of the call)

New in this module:
  - LiveKitBrowserCallHandler: implements exactly the handler surface that
    VoiceOrchestrator / TtsPipeline / ConversationOrchestrator expect (see
    each class's docstring for the attributes/methods they access on
    `self._h` / `self._handler`), so those three classes run completely
    unmodified against a LiveKit room instead of a Twilio WebSocket.
  - _LiveKitAgentAudioPublisher: publishes synthesized speech into the room
    as an outgoing WebRTC audio track.
  - run_livekit_browser_call(): the background task entrypoint called from
    demo_call_token.

Out of scope for this path (stubbed, never crashes):
  - Booking/Calendly, CRM write-back, transfer routing — see
    _update_booking_memory_from_user_turn / _end_call_after_agent_request /
    _transfer_after_agent_request below.
"""
from __future__ import annotations

import asyncio
import dataclasses
import struct
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.core.logger import logger
from app.services.bidirectional_stream_service import generate_mulaw_tts
from app.services.transcript_service import transcript_service
from app.utils.audio_utils import MULAW_FRAME_BYTES, ulaw_to_linear_sample
from app.utils.ssml_utils import strip_ssml_tags
from app.voice.call_control_mixin import CallControlMixin
from app.voice.conversation_orchestrator import ConversationOrchestrator, VOICE_TUNABLES
from app.voice.gemini_live_audio_bridge import GEMINI_OUTPUT_PCM_RATE_HZ
from app.services.openai_realtime_service import LIVEKIT_AUDIO_RATE_HZ as OPENAI_REALTIME_AUDIO_RATE_HZ
from app.voice.humanization_engine import pause_frames_for_chunk

if TYPE_CHECKING:
    from app.voice.humanization_engine import PacingHint

# ── Design decision: agent TTS-out audio format ─────────────────────────────
# We publish mu-law 8kHz audio converted to LINEAR16 PCM, not native
# provider PCM (e.g. ElevenLabs `pcm_16000`). This lets _prefetch_tts_audio
# below call the *exact same* generate_mulaw_tts() helper the Twilio path
# already uses for its non-streaming fallback/greeting turns — zero new
# per-provider (Google/ElevenLabs/Rime) synthesis code — and reuses the same
# ulaw_to_linear_sample() conversion LiveKitTwilioPublisher already uses for
# caller-audio mirroring in app/voice/livekit_twilio_bridge.py. Trade-off:
# 8kHz "phone quality" instead of wideband 16/24kHz, and whole-chunk
# synthesis instead of true incremental provider streaming — both acceptable
# for a demo/preview experience; TtsPipeline still pipelines chunk-by-chunk
# at LLM sentence-flush boundaries so perceived latency is reasonable.
_AGENT_AUDIO_SAMPLE_RATE = 8000

# STT-in is always fed via LiveKitAudioSubscriber (LiveKit PCM), never
# Twilio MULAW, regardless of which STT provider the agent is configured
# for — so we force LINEAR16 here rather than trusting resolve_stt_runtime's
# defaults (which assume Twilio MULAW 8kHz for Deepgram, the platform-wide
# default provider). Provider/model/language selection is still fully
# respected; only the wire format is overridden.
_STT_INPUT_SAMPLE_RATE = 16000
_STT_INPUT_ENCODING = "LINEAR16"

_GREETING_DELAY_SEC = 0.6
_TURN_TIMEOUT_SEC = 25.0

# Hard safety cap on total call duration. This is a new mechanism (no
# equivalent exists on the Twilio path or anywhere else in the codebase as of
# this writing) added specifically because this loop now bills the tenant's
# plan-included-minutes/credits via credit_service for every second the call
# is held open (see the credit-monitoring block in run_livekit_browser_call).
# Per the module docstring / Voice Pipeline vault notes, there is no reliable
# end-of-call signal for a browser-only LiveKit call if the visitor's network
# dies or the tab closes ungracefully — only participant_disconnected/
# disconnected room events, which never fire in that failure mode. Without
# this cap a dead connection could bill indefinitely. This is orthogonal to
# (and does not replace) CallFlowDemoLink.per_user_limit_minutes /
# total_budget_minutes, which are tenant-configured usage budgets enforced
# elsewhere — this is a fixed, code-level upper bound on any single call.
_MAX_CALL_DURATION_SEC = 1800.0

# asyncio.Task objects are only weakly referenced by the event loop (see the
# asyncio docs' explicit warning) — a fire-and-forget asyncio.create_task()
# with no other reference is eligible for GC at any point. For a short burst
# of work that's usually academic; here it would mean a live, minutes-long
# voice conversation silently going dead mid-call with no error surfaced
# anywhere. Mirrors VoiceOrchestrator._pending_final_tasks' pattern.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> asyncio.Task:
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def spawn_browser_call(call_session_id: uuid.UUID) -> asyncio.Task:
    """Fire-and-forget entrypoint for callers (e.g. app.routers.sdk) — keeps
    a strong reference to the task for its whole lifetime, unlike a bare
    asyncio.create_task() call at the call site."""
    return _track(asyncio.create_task(run_livekit_browser_call(call_session_id)))


class _LiveKitAgentAudioPublisher:
    """
    Publishes synthesized agent speech into a LiveKit room as its own
    'agent-tts-<room>' participant — deliberately a *different* identity
    from the 'agent-<room>' identity LiveKitAudioSubscriber connects with
    for caller-audio-in, so both can hold simultaneous connections to the
    same room (LiveKit reconnecting with a duplicate identity would kick the
    older session).

    Connect/publish pattern mirrors app.voice.livekit_twilio_bridge's
    LiveKitTwilioPublisher (kept as a separate, new class here instead of
    reusing that one directly, since that file backs the live Twilio
    recording-mirror path and is not in this task's touch list).
    """

    def __init__(self, room_name: str, sample_rate_hz: int = _AGENT_AUDIO_SAMPLE_RATE) -> None:
        self._room_name = room_name
        self._room: Any = None
        self._source: Any = None
        self._connected = False
        # Per-instance publish sample rate — defaults to the usual TTS-provider
        # rate (8kHz mu-law-derived PCM), but a Gemini Live native-audio call
        # opens this publisher at Gemini's own 24kHz output rate instead (see
        # run_livekit_browser_call), so publish_pcm() below can push Gemini's
        # raw PCM straight into the room with zero resampling. rtc.AudioSource
        # takes its sample rate per-instance at construction (confirmed by
        # reading this SDK usage), not once globally for the whole room, so
        # this is safe to vary per call.
        self._sample_rate = int(sample_rate_hz)
        # Rate-limits the "publish failed" warning to once per outage (rather
        # than once per ~20ms frame) so a sustained failure doesn't flood the
        # logs while still being visible at all — previously this was a bare
        # DEBUG log, so a capture_frame() failure meant TTS could be reported
        # as synthesized/queued successfully while zero audio ever reached
        # the browser, with no trace of why in normal logs.
        self._publish_failed_logged = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def connect(self) -> bool:
        if not settings.LIVEKIT_ENABLED:
            return False
        try:
            from livekit import rtc
        except ImportError:
            logger.error("[LiveKitBrowserCall] livekit package not available")
            return False

        try:
            from app.services.livekit_service import livekit_service, _http_to_ws_url

            livekit_service._validate_room_name(self._room_name)
            url, _, _ = livekit_service._get_credentials()
            ws_url = _http_to_ws_url(url)
            token = livekit_service.generate_agent_token(
                self._room_name, identity=f"agent-tts-{self._room_name}"
            )

            self._room = rtc.Room()
            await self._room.connect(ws_url, token)

            self._source = rtc.AudioSource(self._sample_rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track("agent-tts-audio", self._source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await self._room.local_participant.publish_track(track, options)

            self._connected = True
            logger.info(
                "[LiveKitBrowserCall] agent TTS-out track published room=%s",
                self._room_name,
            )
            return True
        except Exception as exc:
            logger.error(
                "[LiveKitBrowserCall] TTS publisher connect failed room=%s: %s",
                self._room_name,
                exc,
                exc_info=True,
            )
            await self.disconnect()
            return False

    async def publish_mulaw(self, mulaw_bytes: bytes, cancel: asyncio.Event | None = None) -> None:
        """
        Push mu-law bytes into the outgoing track, converting to PCM16 per frame.

        Callable with an arbitrary-length buffer (legacy/whole-utterance
        callers) or with a single already-frame-aligned MULAW_FRAME_BYTES
        chunk (the incremental streaming path in
        LiveKitBrowserCallHandler._publish_mulaw_stream) — either way, any
        final partial frame is padded with mu-law silence (0xFF) rather than
        dropped.
        """
        if not self._connected or not self._source or not mulaw_bytes:
            return
        try:
            from livekit import rtc
        except ImportError:
            return

        try:
            offset = 0
            total = len(mulaw_bytes)
            while offset < total:
                if cancel is not None and cancel.is_set():
                    try:
                        self._source.clear_queue()
                    except Exception:  # noqa: S110 - best-effort barge-in abort
                        pass
                    return

                chunk = mulaw_bytes[offset : offset + MULAW_FRAME_BYTES]
                offset += MULAW_FRAME_BYTES
                if len(chunk) < MULAW_FRAME_BYTES:
                    chunk = chunk + bytes([0xFF]) * (MULAW_FRAME_BYTES - len(chunk))

                samples = [ulaw_to_linear_sample(b) for b in chunk]
                pcm = struct.pack(f"<{len(samples)}h", *samples)
                frame = rtc.AudioFrame.create(self._sample_rate, 1, len(samples))
                # frame.data is a memoryview already cast to int16 ("h")
                # format (see livekit.rtc.AudioFrame.data); assigning a plain
                # `bytes` object (format "B") straight into it raises
                # "ValueError: memoryview assignment: lvalue and rvalue have
                # different structures" even though the byte lengths match.
                # Cast the destination view to bytes format first so both
                # sides agree on structure.
                frame.data.cast("B")[:] = pcm
                await self._source.capture_frame(frame)
                self._publish_failed_logged = False
        except Exception as exc:
            if not self._publish_failed_logged:
                logger.warning(
                    "[LiveKitBrowserCall] publish_mulaw failed room=%s: %s",
                    self._room_name, exc, exc_info=True,
                )
                self._publish_failed_logged = True

    async def publish_pcm(
        self, pcm_bytes: bytes, sample_rate_hz: int, cancel: asyncio.Event | None = None
    ) -> None:
        """
        Push raw PCM16 little-endian mono bytes directly into the outgoing
        track — no mu-law decode, unlike publish_mulaw(). Used exclusively by
        the Gemini Live native-audio path (see gemini_live_audio_bridge.py /
        run_livekit_browser_call): Gemini's own audio output is already raw
        PCM16 at 24kHz, and this publisher instance is opened at that same
        rate (self._sample_rate) for a native-audio call, so no resampling
        step is needed here at all — a deliberate design choice over
        resampling 24k->8k, since rtc.AudioSource's sample rate is set once
        per publisher instance at construction, not globally for the room.

        `sample_rate_hz` is the rate the caller believes `pcm_bytes` is
        encoded at — it must match `self._sample_rate` (the rate this
        publisher's AudioSource was actually opened at). A mismatch would
        silently play audio at the wrong pitch/speed, so instead of playing
        anything, this logs once and drops the chunk.
        """
        if not self._connected or not self._source or not pcm_bytes:
            return
        if cancel is not None and cancel.is_set():
            return
        if sample_rate_hz != self._sample_rate:
            if not self._publish_failed_logged:
                logger.error(
                    "[LiveKitBrowserCall] publish_pcm sample_rate mismatch room=%s "
                    "publisher_rate=%d chunk_rate=%d — dropping chunk",
                    self._room_name, self._sample_rate, sample_rate_hz,
                )
                self._publish_failed_logged = True
            return

        try:
            from livekit import rtc
        except ImportError:
            return

        try:
            # A trailing odd byte (incomplete final PCM16 sample) is dropped
            # rather than raising — mirrors gemini_live_audio_bridge's own
            # odd-length handling convention.
            num_samples = len(pcm_bytes) // 2
            if num_samples <= 0:
                return
            aligned = pcm_bytes[: num_samples * 2]
            frame = rtc.AudioFrame.create(self._sample_rate, 1, num_samples)
            frame.data.cast("B")[:] = aligned
            await self._source.capture_frame(frame)
            self._publish_failed_logged = False
        except Exception as exc:
            if not self._publish_failed_logged:
                logger.warning(
                    "[LiveKitBrowserCall] publish_pcm failed room=%s: %s",
                    self._room_name, exc, exc_info=True,
                )
                self._publish_failed_logged = True

    async def disconnect(self) -> None:
        self._connected = False
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception as exc:
                logger.debug("[LiveKitBrowserCall] room disconnect failed: %s", exc)
        self._room = None
        self._source = None


class _GeminiLiveAudioSink:
    """
    Minimal ``stt_pipeline``-shaped adapter so ``LiveKitAudioSubscriber``
    (which unconditionally calls ``self._stt_pipeline.feed_audio_chunk(...)``
    on every decoded frame) can feed a ``GeminiLiveSession`` instead of an
    ``SttPipeline``, for native-audio agents — without widening
    ``LiveKitAudioSubscriber``'s constructor to accept a bare callback. This
    is the smaller, more targeted change: ``LiveKitAudioSubscriber`` already
    resamples caller audio to LINEAR16 16kHz mono
    (``_STT_INPUT_SAMPLE_RATE``), exactly the format
    ``GeminiLiveSession.send_audio`` expects, so no conversion happens here
    — only the sink changes (see the Gemini Live integration plan's
    "Browser | in" row).
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    async def feed_audio_chunk(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        try:
            await self._session.send_audio(pcm_bytes)
        except Exception as exc:
            logger.error(
                "[LiveKitBrowserCall] GeminiLiveSession.send_audio failed: %s", exc, exc_info=True
            )


class _OpenAIRealtimeAudioSink:
    """
    OpenAI Realtime analog of ``_GeminiLiveAudioSink`` — same
    ``stt_pipeline``-shaped adapter pattern, targeting an
    ``OpenAIRealtimeSession`` instead. ``LiveKitAudioSubscriber`` is opened
    with ``output_sample_rate=OPENAI_REALTIME_AUDIO_RATE_HZ`` (24kHz) for a
    native-audio OpenAI call (see ``run_livekit_browser_call``) — exactly
    the rate ``OpenAIRealtimeSession.send_audio`` expects when the session
    was started with ``audio_format="audio/pcm"``, so no conversion happens
    here either.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    async def feed_audio_chunk(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        try:
            await self._session.send_audio(pcm_bytes)
        except Exception as exc:
            logger.error(
                "[LiveKitBrowserCall] OpenAIRealtimeSession.send_audio failed: %s", exc, exc_info=True
            )


class LiveKitBrowserCallHandler(CallControlMixin):
    """
    Transport adapter: implements exactly the attribute/method surface that
    VoiceOrchestrator, TtsPipeline, and ConversationOrchestrator need on
    their `handler` so those classes can be constructed against this object
    unmodified. See the module docstring above and each of those classes'
    own docstrings for the authoritative list.

    Deliberately written generally enough (no assumption baked in that this
    call came from the demo-link flow specifically — only that it is a
    LiveKit browser call bound to a real CallSession/Agent) that it could
    back the public-call-token widget flow too without modification, once
    that path gets its own lifecycle wiring (out of scope here — see
    run_livekit_browser_call's docstring).
    """

    # Same tunables ConversationOrchestrator reads off the handler for TTS
    # chunk flushing — mirrors BidirectionalStreamHandler's class attributes.
    TTS_FLUSH_MIN_WORDS = VOICE_TUNABLES.tts_flush_min_words
    TTS_FLUSH_MAX_WORDS = VOICE_TUNABLES.tts_flush_max_words

    def __init__(self, db, call_session, agent, call_flow=None) -> None:
        self.db = db
        self.call_session = call_session
        self.agent = agent
        self.call_flow = call_flow
        self.call_session_id = str(call_session.id)
        self.agent_id = str(agent.id) if agent else None
        self.call_sid: str | None = None
        self._call_ended: bool = False

        # Twilio-only naming (`streamSid`). Nothing in this path sends Twilio
        # media-stream frames, but VoiceOrchestrator reads this attribute for
        # logging in a couple of places, so it must exist and never crash.
        self.stream_sid: str | None = f"livekit_{self.call_session_id}"

        # ── TTS state (read by TtsPipeline / ConversationOrchestrator) ──────
        self.is_speaking = False
        self._is_tts_playing = False
        self._tts_cancel = asyncio.Event()
        self._tts_lock = asyncio.Lock()
        self._tts_worker_task = None  # set by VoiceOrchestrator
        self._tts_pipeline = None  # set by VoiceOrchestrator
        self._prev_tts_tail = b""
        # No Twilio jitter buffer exists on this path — treat as always
        # primed so TtsPipeline/ConversationOrchestrator never wait on it.
        self._twilio_buffer_primed = True
        self._use_ssml = True

        from app.voice.metrics import VoiceTurnMetrics
        self._voice_metrics = VoiceTurnMetrics(
            call_sid=self.stream_sid,
            transport="livekit_demo",
            agent_id=self.agent_id,
        )

        self._llm_response_task: asyncio.Task | None = None
        self._turn_response_started = False

        # ── Barge-in / interim tunables — same defaults BidirectionalStreamHandler
        # uses, so behaviour matches the phone experience as closely as possible.
        self._enable_interim_llm: bool = bool(getattr(settings, "VOICE_ENABLE_INTERIM_LLM", False))
        self._min_interim_words: int = max(1, int(getattr(settings, "VOICE_MIN_INTERIM_WORDS", 4) or 4))
        self._min_interim_confidence: float = float(
            getattr(settings, "VOICE_MIN_INTERIM_CONFIDENCE", 0.52) or 0.52
        )
        self._min_interim_interval_sec: float = VOICE_TUNABLES.stt_interim_interval_ms / 1000.0
        self._barge_in_min_conf: float = float(getattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE", 0.26) or 0.26)
        self._barge_in_min_conf_1w: float = float(
            getattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE_1W", 0.52) or 0.52
        )
        self._barge_in_min_words: int = max(1, int(getattr(settings, "VOICE_BARGE_IN_MIN_WORDS", 2) or 2))

        # ── RAG interim-prefetch ─────────────────────────────────────────────
        # Mirrors bidirectional_stream.py's _prefetch_rag_context pattern
        # (fire vector/KB retrieval in the background on the first qualifying
        # STT interim of a turn, so it overlaps STT endpointing instead of
        # blocking the eventual LLM start), adapted to this transport's
        # retrieval call (kb_retrieval_service.retrieve_kb_context_for_turn —
        # see _maybe_start_rag_prefetch / _prefetch_kb_context below).
        # Single-slot state, same model as _llm_response_task: None means
        # "not fired for this turn"; consumed-once-and-reset-to-None by
        # ConversationOrchestrator.generate_and_stream_response on STT final,
        # or discarded by _cancel_inflight_llm_response on barge-in — so it
        # can never leak into a later, unrelated turn.
        self._rag_prefetch_task: asyncio.Task | None = None
        self._rag_prefetch_source_text: str = ""
        self._rag_prefetch_min_words: int = max(
            1, int(getattr(settings, "VOICE_RAG_PREFETCH_MIN_WORDS", 1) or 1)
        )
        self._rag_prefetch_min_confidence: float = float(
            getattr(settings, "VOICE_RAG_PREFETCH_MIN_CONFIDENCE", 0.05) or 0.05
        )
        # Pickup-detection tunables: unused by this path (LiveKit rooms have no
        # Twilio ringing/system-message phase to skip) but VoiceOrchestrator's
        # __init__ reads them off the handler unconditionally.
        self._min_audio_level_threshold: int = int(getattr(settings, "VOICE_MIN_AUDIO_RMS_FOR_PICKUP", 70) or 70)
        self._audio_samples_needed: int = max(4, int(getattr(settings, "VOICE_PICKUP_SAMPLE_WINDOW", 6) or 6))
        self._audio_non_silent_needed: int = self._audio_samples_needed

        self._stop_event = asyncio.Event()

        self._conversation = ConversationOrchestrator(self)

        # Wired up by run_livekit_browser_call() once connected.
        self._agent_publisher: _LiveKitAgentAudioPublisher | None = None

    # ── Transcript ────────────────────────────────────────────────────────

    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: float | None = None,
        message_metadata: dict | None = None,
        defer_post_write: bool = False,
    ) -> None:
        """
        Persist a transcript line. Cross-turn state lives entirely in
        `callsession` / `transcript_message` (never in memory) — mirrors
        app.voice.call_control_mixin.CallControlMixin._add_to_transcript's
        core DB write, without that mixin's Twilio-adjacent dedupe/contact-
        intake side effects (booking/CRM intake sync is out of scope here).
        """
        if not self.call_session:
            return
        try:
            clean_message = strip_ssml_tags(message)
            hipaa_enabled = bool(getattr(self.call_flow, "hipaa_compliance", False)) if self.call_flow else False

            added = await transcript_service.add_and_broadcast_message(
                db=self.db,
                call_session_id=self.call_session.id,
                role=role,
                message=clean_message,
                message_type=message_type,
                agent_id=self.agent.id if self.agent else None,
                user_id=self.call_session.user_id,
                confidence=confidence,
                metadata=message_metadata,
                hipaa_enabled=hipaa_enabled,
            )
            if added is None:
                return

            if not defer_post_write:
                conversation = transcript_service.get_conversation_array(self.db, self.call_session.id)
                self.call_session.call_transcript = conversation
                self.db.commit()

            try:
                self.db.refresh(self.call_session)
            except Exception as exc:
                logger.debug(
                    "[LiveKitBrowserCall] failed to refresh call_session %s: %s",
                    self.call_session.id, exc,
                )
        except Exception as exc:
            logger.error("[LiveKitBrowserCall] _add_to_transcript failed: %s", exc, exc_info=True)

    # ── Gemini Live (native-audio speech-to-speech) audio-out sink ──────────
    # Wired by VoiceOrchestrator._on_gemini_live_audio_chunk /
    # _on_gemini_live_interrupted (duck-typed via hasattr, since the Twilio
    # handler exposes its own differently-shaped `_stream_live_audio_chunk` /
    # `_send_twilio_clear_event` methods for the same two hooks — see that
    # module for the branch).

    async def _publish_gemini_live_audio_chunk(
        self,
        pcm16_24k_bytes: bytes,
        cancel: asyncio.Event,
        sample_rate_hz: int = GEMINI_OUTPUT_PCM_RATE_HZ,
    ) -> None:
        """
        Native-audio-S2S audio-out sink for this transport, shared by both
        Gemini Live and OpenAI Realtime (see the Gemini-specific name's note
        below). Publishes raw PCM16 output directly via
        `_agent_publisher.publish_pcm` — `_agent_publisher` is opened at the
        matching rate for a native-audio call (see run_livekit_browser_call),
        so unlike the Twilio path's `_stream_live_audio_chunk` (mu-law/8kHz),
        no resampling or codec conversion happens here at all.

        `sample_rate_hz` defaults to Gemini's own output rate for the
        original Gemini call site; OpenAI Realtime's call site passes its own
        rate explicitly instead of relying on the two providers' rates
        happening to both be 24kHz today — a future rate change to either
        provider must not silently mismatch the other and get audio dropped
        by `publish_pcm`'s rate-mismatch guard.
        """
        publisher = self._agent_publisher
        if publisher is None or not publisher.connected:
            return
        await publisher.publish_pcm(
            pcm16_24k_bytes, sample_rate_hz=sample_rate_hz, cancel=cancel
        )

    async def _clear_gemini_live_playout_queue(self) -> None:
        """
        Barge-in for the Gemini Live path on this transport: discard whatever
        LiveKit's own AudioSource has already buffered internally (up to
        ~1000ms) so an interruption is heard immediately instead of after the
        already-queued audio finishes playing out. Mirrors
        `_cancel_inflight_llm_response`'s existing `source.clear_queue()` call
        for the regular TTS barge-in path — there is no Twilio
        `_send_twilio_clear_event` equivalent on this transport.
        """
        publisher = getattr(self, "_agent_publisher", None)
        source = getattr(publisher, "_source", None) if publisher else None
        if source is not None:
            try:
                source.clear_queue()
            except Exception:  # noqa: S110 - best-effort barge-in abort
                pass

    # ── Out-of-scope side effects — clean stubs, never crash ────────────────

    def _update_booking_memory_from_user_turn(self, transcript: str) -> None:
        """Booking is out of scope for the browser demo path — no-op."""
        return

    async def _end_call_after_agent_request(self) -> None:
        """
        Twilio path hangs up when the LLM emits [END_CALL]. The browser demo
        path intentionally keeps the conversation open instead — there is no
        "hang up the phone" equivalent for a WebRTC tab the visitor controls,
        and ending the LiveKit room out from under an active browser tab
        would be a worse experience than the agent saying goodbye and the
        conversation simply continuing if the visitor speaks again.
        """
        logger.warning(
            "[LiveKitBrowserCall] agent requested end-call (call_session_id=%s) — "
            "not supported on the browser demo path yet, continuing the call",
            self.call_session_id,
        )

    async def _transfer_after_agent_request(self) -> None:
        """Call transfer has no meaning for a browser-only room — no-op, log only."""
        logger.warning(
            "[LiveKitBrowserCall] agent requested transfer (call_session_id=%s) — "
            "not supported on the browser demo path yet, continuing the call",
            self.call_session_id,
        )

    # ── STT → turn handling (wired as SttPipeline on_interim/on_final) ─────

    async def _maybe_process_interim(self, transcript: str, confidence: float) -> None:
        """
        Barge-in + RAG interim-prefetch trigger.

        Barge-in is resolved FIRST via a single `is_barge_in` boolean, and
        `return`s immediately when it applies — mirrors
        bidirectional_stream.py's `_maybe_process_interim` ordering exactly
        (computes `is_barge_in`, cancels + `return`s when true; its RAG-
        prefetch block is only reached when `is_barge_in` is false). This
        ordering matters here specifically because the RAG-prefetch gate
        (VOICE_RAG_PREFETCH_MIN_WORDS/MIN_CONFIDENCE) is strictly weaker than
        the barge-in gate — every interim that qualifies for barge-in also
        qualifies to fire a prefetch, and `_cancel_inflight_llm_response()`
        (called on barge-in) unconditionally discards `_rag_prefetch_task`.
        Firing the prefetch before checking barge-in would mean every
        barge-in-initiated turn wastes a prefetch attempt (fired then
        immediately cancelled) and gets zero prefetch benefit — precisely
        the highest-value case (a user interrupting the agent) would get
        none of the benefit.

        Note `is_barge_in` being false does NOT require `_is_tts_playing` to
        be false — an interim while TTS *is* playing but that doesn't meet
        the barge-in word/confidence thresholds is also `is_barge_in=False`
        and still eligible to fire a prefetch, exactly matching Twilio's
        structure (its prefetch block isn't gated on `_is_tts_playing`
        either, only on `is_barge_in`).

        LiveKit rooms have their own echo cancellation and the default
        platform setting (VOICE_ENABLE_INTERIM_LLM=False) means the Twilio
        path doesn't start early LLM generation on interims either — so this
        mirrors that default behaviour rather than reimplementing the
        early-LLM path's seed/regeneration bookkeeping.
        """
        try:
            text = (transcript or "").strip()
            if not text:
                return
            word_count = len(text.split())

            min_conf = self._barge_in_min_conf_1w if word_count < 2 else self._barge_in_min_conf
            is_barge_in = (
                self._is_tts_playing
                and word_count >= self._barge_in_min_words
                and confidence >= min_conf
            )
            if is_barge_in:
                logger.info(
                    "[LiveKitBrowserCall] barge-in: words=%d conf=%.2f text=%r",
                    word_count, confidence, text[:40],
                )
                await self._cancel_inflight_llm_response()
                return

            # Barge-in did not apply this call — safe to consider firing a
            # prefetch (whether or not TTS happens to still be playing).
            self._maybe_start_rag_prefetch(text, confidence, word_count)
        except Exception as exc:
            logger.error("[LiveKitBrowserCall] _maybe_process_interim error: %s", exc, exc_info=True)

    def _maybe_start_rag_prefetch(self, text: str, confidence: float, word_count: int) -> None:
        """
        Fire KB-context retrieval in the background on the first qualifying
        interim of a turn (mirrors bidirectional_stream.py's
        `_prefetch_rag_context` trigger gating). Consumed exactly once, by
        `ConversationOrchestrator.generate_and_stream_response` on STT final
        — never invoked twice for the same turn (guarded below) and never
        left to leak into a later turn (reset to None on consume or on
        barge-in cancellation in `_cancel_inflight_llm_response`).
        """
        if self._rag_prefetch_task is not None:
            return  # already fired for this turn — never fire a duplicate request
        flow_kb_ids = (self.call_flow.knowledge_base_ids or []) if self.call_flow else []
        if not flow_kb_ids or not self.db:
            return  # RAG/KB not configured for this call flow — nothing to prefetch
        if word_count < self._rag_prefetch_min_words or confidence < self._rag_prefetch_min_confidence:
            return
        self._rag_prefetch_source_text = text
        self._rag_prefetch_task = asyncio.create_task(
            self._prefetch_kb_context(text, list(flow_kb_ids))
        )

    async def _prefetch_kb_context(self, transcript: str, kb_ids: list) -> tuple[str, float]:
        """
        Background KB-context retrieval for RAG interim-prefetch.

        `kb_retrieval_service.retrieve_kb_context_for_turn` is already fully
        async and fails open internally (returns `("", latency_ms)` on any
        internal error) — this wrapper adds the same
        `RAG_KB_RETRIEVAL_TIMEOUT_SEC` bound
        `ConversationOrchestrator`'s synchronous fallback path uses, so a
        slow/hanging retrieval can never block the eventual LLM turn: on
        timeout or any other exception this always returns a valid empty
        result rather than raising, matching
        `_prefetch_rag_context`'s fail-open contract on the Twilio path.
        """
        try:
            from app.services.kb_retrieval_service import retrieve_kb_context_for_turn
            from app.utils.redis_client import get_redis

            timeout_sec = float(getattr(settings, "RAG_KB_RETRIEVAL_TIMEOUT_SEC", 0.7) or 0.7)
            return await asyncio.wait_for(
                retrieve_kb_context_for_turn(
                    transcript=transcript,
                    kb_ids=kb_ids,
                    redis_client=get_redis(),
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.debug("[RAG prefetch] timed out for '%s…'", transcript[:20])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[RAG prefetch] failed: %s", exc)
        return "", 0.0

    async def _cancel_inflight_llm_response(self) -> None:
        task = self._llm_response_task
        self._llm_response_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("[LiveKitBrowserCall] cancelled turn raised: %s", exc)
        if self._tts_pipeline:
            await self._tts_pipeline.cancel_current_and_clear_queue()
        # Discard stale RAG prefetch so the next turn gets a fresh retrieval —
        # mirrors bidirectional_stream.py's _cancel_inflight_llm_response.
        prefetch = self._rag_prefetch_task
        if prefetch and not prefetch.done():
            prefetch.cancel()
        self._rag_prefetch_task = None
        self._rag_prefetch_source_text = ""
        # LiveKit protocol: cancelling our TTS task locally only stops *us*
        # from pushing more frames — it does not stop frames already pushed
        # into rtc.AudioSource's own internal playout queue (up to 1000ms of
        # buffered audio) from continuing to play out to the browser. Mirrors
        # TtsStreamMixin._send_twilio_clear_event's role on the Twilio path.
        # Unconditional (not gated on the streaming loop's own cancel check —
        # that check lives inside publish_mulaw()'s per-call loop and is not
        # reliably reached on the primary single-frame streaming path).
        publisher = getattr(self, "_agent_publisher", None)
        source = getattr(publisher, "_source", None) if publisher else None
        if source is not None:
            try:
                source.clear_queue()
            except Exception:  # noqa: S110 - best-effort barge-in abort
                pass

    async def _process_transcript(
        self,
        transcript: str,
        confidence: float,
        acoustic_speech_end_mono: float | None = None,
    ) -> None:
        """STT final callback (wired via VoiceOrchestrator._on_final)."""
        try:
            text = (transcript or "").strip()
            if not text:
                return

            if self._is_tts_playing:
                logger.info("[LiveKitBrowserCall] barge-in (final): %r", text[:40])
                await self._cancel_inflight_llm_response()

            if hasattr(self, "_voice_metrics") and self._voice_metrics:
                self._voice_metrics.begin_turn_at_stt_final(acoustic_speech_end_mono)

            # 🎯 Check for call screening detection - end call if action is hang_up
            if await self._check_and_handle_call_screener(text):
                return  # Stop processing - call is ending

            await self._add_to_transcript("client", text, "speech", confidence)
            self._update_booking_memory_from_user_turn(text)
            await self._complete_llm_turn_after_stt_final(text, confidence)
        except Exception as exc:
            logger.error("[LiveKitBrowserCall] _process_transcript error: %s", exc, exc_info=True)

    async def _complete_llm_turn_after_stt_final(self, transcript: str, confidence: float) -> None:
        self._tts_cancel.clear()

        async def _run() -> None:
            try:
                await self.generate_and_stream_response(transcript, confidence, is_greeting=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[LiveKitBrowserCall] turn generation failed: %s", exc, exc_info=True)

        task = asyncio.create_task(_run())
        self._llm_response_task = task
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_TURN_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.error("[LiveKitBrowserCall] turn timed out after %.0fs", _TURN_TIMEOUT_SEC)
        except asyncio.CancelledError:
            pass
        finally:
            if self._llm_response_task is task:
                self._llm_response_task = None

    async def generate_and_stream_response(
        self, user_text: str, confidence: float, is_greeting: bool = False
    ) -> None:
        """Delegates to ConversationOrchestrator — the reused LLM/RAG/prompt pipeline."""
        await self._conversation.generate_and_stream_response(user_text, confidence, is_greeting=is_greeting)

    def _schedule_recreate_stt_for_email_collection(self, agent_text: str) -> None:
        """
        Twilio-only Deepgram endpointing upgrade for spelled-out email
        collection. Browser STT input already runs at a fixed LINEAR16
        16kHz configuration for this path (see module docstring) — no-op.
        """
        return

    # ── TTS synthesis + LiveKit-native playback ─────────────────────────────
    #
    # _prefetch_tts_audio / _stream_tts_chunk below mirror
    # TtsStreamMixin._prefetch_tts_audio / _stream_tts_chunk
    # (app/voice/tts_stream_mixin.py) so this path gets the provider's *true*
    # incremental streaming API (Rime/ElevenLabs async_stream_synthesize,
    # Google's stream_text_to_speech) instead of buffering a whole
    # utterance's mu-law bytes via generate_mulaw_tts() before publishing
    # anything — the first audio now reaches the LiveKit room as soon as the
    # provider emits its first chunk, matching the Twilio path's latency
    # characteristics. generate_mulaw_tts() is kept imported only as a last-
    # resort fallback for providers/paths that expose neither streaming API.

    async def _prefetch_tts_audio(self, task: dict) -> Any:
        """
        Resolve the TTS provider's real streaming API for this chunk and
        return an async iterator of raw mu-law byte fragments as they're
        generated (or None on empty text / cancellation / unresolvable
        provider). Never buffers the whole utterance itself — TtsPipeline
        awaits this to get the iterator object, then _stream_tts_chunk
        below actually drains it while publishing incrementally.
        """
        text = (task.get("text") or "").strip()
        if not text or self._tts_cancel.is_set():
            return None
        use_ssml = bool(task.get("use_ssml", False))

        try:
            from app.core.agent_runtime import resolve_tts_runtime
            from app.services.google_tts_service import google_tts_service
            from app.utils.eleven_tts_text import prepare_tts_text_for_provider
            from app.utils.tts_adapter import get_tts_adapter
            from app.voice.tts_provider_capabilities import build_voice_settings_overlay

            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            tts_runtime = resolve_tts_runtime(self.agent, db=self.db)
            tts_provider_slug = tts_runtime.adapter_slug

            streaming_text = strip_ssml_tags(text) if use_ssml or text.lstrip().startswith("<speak>") else text
            streaming_text = prepare_tts_text_for_provider(streaming_text, tts_provider_slug)
            if not streaming_text or not streaming_text.strip():
                return None

            if tts_provider_slug and tts_provider_slug not in ("google", ""):
                external_voice_id = tts_runtime.voice_external_id
                if not external_voice_id:
                    tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
                    external_voice_id = getattr(tts_voice, "external_voice_id", None)
                if not external_voice_id and tts_provider_slug == "rime":
                    external_voice_id = "mistv2_Wildflower"
                elif not external_voice_id and tts_provider_slug == "hume":
                    from app.services.hume_tts_service import HUME_DEFAULT_VOICE

                    external_voice_id = HUME_DEFAULT_VOICE
                elif not external_voice_id and tts_provider_slug == "xai":
                    from app.services.xai_tts_service import XAI_DEFAULT_VOICE

                    external_voice_id = XAI_DEFAULT_VOICE
                if not external_voice_id:
                    logger.warning(
                        "[LiveKitBrowserCall] TTS voice not configured for streaming provider=%s",
                        tts_provider_slug,
                    )
                    return None

                adapter = get_tts_adapter(tts_provider_slug)
                provider_settings = dict(tts_runtime.settings_json)
                if tts_provider_slug == "elevenlabs":
                    provider_settings.setdefault("output_format", "ulaw_8000")
                    # Phase 4D-2: read the previously QUEUED chunk's text — captured
                    # synchronously by TtsPipeline.queue_tts() at enqueue time — not
                    # a shared instance attribute mutated after some other chunk's
                    # playback completes (that was the source of the stale
                    # `previous_text` race: chunk N+1's prefetch commonly runs
                    # concurrently with chunk N still playing).
                    previous_text = (task.get("_previous_text") or "").strip()
                    if previous_text:
                        provider_settings["previous_text"] = previous_text[-500:]
                elif tts_provider_slug == "rime":
                    # Rime uses async_stream_synthesize — no output_format key needed
                    # (mulaw 8 kHz is the default in RimeTTSAdapter).
                    pass
                else:
                    provider_settings.setdefault("output_format", "ulaw_8000")

                # Fold in any humanization overlay (e.g. ElevenLabs stability hint)
                # computed by TtsPipeline._process_chunk. Capability-gated and a
                # no-op for providers/decisions with nothing safe to apply — see
                # app.voice.tts_provider_capabilities.build_voice_settings_overlay.
                # Isolated try/except: a humanization failure must never prevent
                # this chunk's TTS request from going out with normal settings.
                try:
                    provider_settings.update(
                        build_voice_settings_overlay(
                            tts_provider_slug, task.get("_humanization_decision")
                        )
                    )
                except Exception as exc:
                    logger.debug("[TTS] humanization overlay skipped: %s", exc)

                # Prefer true async streaming for providers that support it (Rime, ElevenLabs).
                if hasattr(adapter, "async_stream_synthesize"):
                    _cancel_ref = self._tts_cancel

                    async def _async_provider_iter(
                        _adapter=adapter,
                        _text=streaming_text,
                        _vid=external_voice_id,
                        _cfg=provider_settings,
                        _cancel=_cancel_ref,
                    ):
                        async for chunk in _adapter.async_stream_synthesize(
                            text=_text,
                            voice_external_id=_vid,
                            settings_json=_cfg,
                        ):
                            if _cancel.is_set():
                                break
                            if chunk:
                                yield chunk

                    return _async_provider_iter()

                sync_iter = adapter.stream_synthesize(
                    text=streaming_text,
                    voice_external_id=external_voice_id,
                    settings_json=provider_settings,
                )

                async def _async_iter_from_sync(sync_source):
                    iterator = iter(sync_source)
                    sentinel = object()
                    while True:
                        if self._tts_cancel.is_set():
                            break
                        chunk = await asyncio.to_thread(next, iterator, sentinel)
                        if chunk is sentinel:
                            break
                        if chunk:
                            yield chunk

                return _async_iter_from_sync(sync_iter)

            # Google (or unresolved provider): native async streaming API.
            tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
            google_voice_name = getattr(tts_voice, "external_voice_id", None)
            audio_iter = google_tts_service.stream_text_to_speech(
                text=streaming_text,
                language=lang,
                voice_type=voice,
                speaking_rate=1.0,
                output_format="mulaw",
                use_chirp3_hd=True,
                sample_rate_hz=_AGENT_AUDIO_SAMPLE_RATE,
                voice_name_override=google_voice_name,
            )

            async def _checked_async_iter(source_iter):
                async for chunk in source_iter:
                    if self._tts_cancel.is_set():
                        break
                    if chunk:
                        yield chunk

            return _checked_async_iter(audio_iter)
        except Exception as exc:
            logger.warning(
                "[LiveKitBrowserCall] TTS streaming setup failed for %r: %s — "
                "falling back to whole-utterance synthesis", text[:30], exc,
            )
            if self._tts_cancel.is_set():
                return None
            try:
                lang = self.agent.language if self.agent and self.agent.language else "en"
                voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
                return await generate_mulaw_tts(
                    text=text,
                    lang=lang,
                    voice=voice,
                    use_chirp3_hd=True,
                    speaking_rate=1.0,
                    use_ssml=use_ssml,
                    add_office_bg=False,
                    agent=self.agent,
                    db=self.db,
                ) or None
            except Exception as fallback_exc:
                logger.warning(
                    "[LiveKitBrowserCall] TTS fallback synthesis also failed for %r: %s",
                    text[:30], fallback_exc,
                )
                return None

    async def _publish_mulaw_stream(
        self,
        publisher: "_LiveKitAgentAudioPublisher",
        audio_iter: Any,
        cancel: asyncio.Event,
    ) -> None:
        """
        Drain an async iterator of provider mu-law fragments and publish each
        full MULAW_FRAME_BYTES frame into the LiveKit room as soon as it's
        assembled — mirrors TtsStreamMixin._stream_tts_chunk's
        stream_mulaw_from_audio_iter, but targets publisher.publish_mulaw()
        instead of the Twilio WebSocket. Frames are aligned across provider
        chunk boundaries (buffered, not padded per-chunk) so an arbitrary
        provider fragment size never introduces mid-utterance silence
        padding — only the final remainder gets padded.
        """
        byte_buf = bytearray()
        async for chunk_bytes in audio_iter:
            if cancel.is_set():
                return
            if not chunk_bytes:
                continue
            byte_buf.extend(chunk_bytes)
            while len(byte_buf) >= MULAW_FRAME_BYTES:
                frame = bytes(byte_buf[:MULAW_FRAME_BYTES])
                del byte_buf[:MULAW_FRAME_BYTES]
                await publisher.publish_mulaw(frame, cancel=cancel)
                if cancel.is_set():
                    return

        if cancel.is_set():
            return

        if byte_buf:
            pad = MULAW_FRAME_BYTES - (len(byte_buf) % MULAW_FRAME_BYTES)
            if pad != MULAW_FRAME_BYTES:
                byte_buf.extend(bytes([0xFF]) * pad)
            await publisher.publish_mulaw(bytes(byte_buf), cancel=cancel)

    async def _stream_tts_chunk(
        self,
        text: str,
        use_ssml: bool = False,
        is_final: bool = False,
        prefetched_bytes: Any = None,
        pacing: "PacingHint | None" = None,
        previous_text: str | None = None,
    ) -> None:
        """
        Publish one TTS chunk's audio into the LiveKit room (TtsPipeline's audio sink).

        `pacing` (Phase 4C-2, optional): the HumanizationDecision.pacing hint
        already computed once by TtsPipeline._process_chunk — never
        recomputed here. When eligible, a small trailing silence is appended
        after this chunk's real audio (see
        app.voice.humanization_engine.pause_frames_for_chunk). Defaults to
        None and is fully inert when omitted or when
        VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES is 0.

        `previous_text` (Phase 4D-2, optional): the previously QUEUED chunk's
        text, captured synchronously by TtsPipeline.queue_tts() — threaded
        through only for the rare fallback re-prefetch below (when
        `prefetched_bytes` wasn't already supplied); the common path never
        reaches this since `_prefetch_tts_audio` reads the same value
        directly off the task dict TtsPipeline built.
        """
        if not text or not text.strip() or self._tts_cancel.is_set():
            return

        publisher = self._agent_publisher
        if publisher is None or not publisher.connected:
            logger.warning(
                "[LiveKitBrowserCall] no agent audio publisher connected — dropping TTS chunk"
            )
            return

        async with self._tts_lock:
            self.is_speaking = True
            try:
                source = prefetched_bytes
                if source is None:
                    source = await self._prefetch_tts_audio(
                        {"text": text, "use_ssml": use_ssml, "_previous_text": previous_text}
                    )
                if source is None or self._tts_cancel.is_set():
                    return

                self._is_tts_playing = True

                if hasattr(source, "__aiter__"):
                    logger.debug(
                        "[LiveKitBrowserCall] LiveKit playback: streaming TTS chunk "
                        "incrementally (call_session_id=%s)", self.call_session_id,
                    )
                    try:
                        await self._publish_mulaw_stream(publisher, source, self._tts_cancel)
                    except asyncio.CancelledError:
                        raise
                    except Exception as stream_exc:
                        logger.warning(
                            "[LiveKitBrowserCall] streaming TTS chunk playback error: %s",
                            stream_exc,
                        )
                    # NOTE (Phase 4D-2): previously wrote `text` into
                    # self._elevenlabs_prev_tts_text here, post-playback, as the
                    # "previous_text" source for the NEXT chunk. That write raced
                    # the next chunk's prefetch (which typically starts while THIS
                    # chunk is still playing) and is no longer read by anything —
                    # TtsPipeline.queue_tts() now captures the next chunk's
                    # previous_text synchronously at queue time instead. See
                    # app.voice.tts_pipeline.TtsPipeline._last_queued_text.
                elif isinstance(source, (bytes, bytearray)) and source:
                    # Defensive fallback: _prefetch_tts_audio always returns an
                    # async iterator or None above, but keep this path so a
                    # caller passing raw bytes directly (e.g. tests) still works.
                    logger.debug(
                        "[LiveKitBrowserCall] LiveKit playback: publishing %d mu-law bytes "
                        "(call_session_id=%s)", len(source), self.call_session_id,
                    )
                    await publisher.publish_mulaw(source, cancel=self._tts_cancel)

                # Phase 4C-2: optional small trailing silence after a non-final
                # chunk ending at a real sentence boundary — same eligibility
                # rule as Twilio (pause_frames_for_chunk), executed via the
                # same publisher.publish_mulaw() used for all other frames so
                # cancellation is checked identically. Deliberately placed
                # INSIDE this try block, before `finally` resets
                # _is_tts_playing — LiveKit's barge-in gate reads
                # _is_tts_playing per chunk (see _maybe_process_interim), so
                # this keeps barge-in active through the pause instead of
                # silently going inactive between chunks.
                if not self._tts_cancel.is_set():
                    try:
                        for _ in range(pause_frames_for_chunk(pacing, is_final)):
                            if self._tts_cancel.is_set():
                                break
                            await publisher.publish_mulaw(
                                bytes([0xFF]) * MULAW_FRAME_BYTES, cancel=self._tts_cancel
                            )
                    except Exception as pause_err:
                        logger.debug(
                            "[LiveKitBrowserCall] inter-sentence pause failed (non-fatal): %s",
                            pause_err,
                        )
            finally:
                self.is_speaking = False
                self._is_tts_playing = False
                if is_final:
                    self._prev_tts_tail = b""

    async def _full_shutdown(self) -> None:
        """Idempotent call-end signal for the main run loop (mirrors BidirectionalStreamHandler's)."""
        self._stop_event.set()


def _forced_browser_stt_runtime(resolved: Any) -> Any:
    """Override wire format to LINEAR16/16kHz — see module docstring."""
    return dataclasses.replace(
        resolved,
        sample_rate_hz=_STT_INPUT_SAMPLE_RATE,
        encoding=_STT_INPUT_ENCODING,
    )


async def _start_browser_call_recording(db, call_session) -> str | None:
    """
    Start a room-composite egress on the call's own native LiveKit room
    (room_{call_session.id} — already created for the caller/agent by
    demo_call_token / _LiveKitAgentAudioPublisher.connect(), unlike the
    Twilio path which has to stand up a *separate* mirror room + duplicate
    publishers purely for recording since a plain Twilio call has no
    native LiveKit room of its own).

    Fail-open: any error here must never abort or degrade the actual call —
    same convention as BidirectionalStreamHandler._start_livekit_recording.
    Returns the egress_id on success, None otherwise (including when
    recording is disabled for this call).

    Marks call_session.recording_error=True on a genuine start failure
    (egress exception, or a falsy egress_id with no exception) so
    GET /api/v1/recordings/{id} can report a real failure instead of the
    generic "still processing" 404 — "disabled" is left unmarked since it
    already gets its own distinct 404 via get_recording_enabled_for_call().
    """
    from app.services.call_recording_upload_service import mark_recording_error
    from app.services.recording_config_service import get_recording_enabled_for_call

    try:
        if not get_recording_enabled_for_call(db, call_session):
            logger.info(
                "[LiveKitBrowserCall] recording not enabled for session=%s — skipping",
                call_session.id,
            )
            return None

        from app.services.livekit_recording_service import livekit_recording_service
        from app.services.s3_recording_service import build_s3_key

        gcs_path = build_s3_key(
            workspace_id=call_session.tenant_id,
            call_id=call_session.id,
            end_time=call_session.end_time,
        )
        egress_id = await livekit_recording_service.start_room_recording(
            call_id=call_session.id,
            workspace_id=call_session.tenant_id,
            gcs_path=gcs_path,
        )
        if not egress_id:
            logger.warning(
                "[LiveKitBrowserCall] egress start returned no egress_id (no exception "
                "raised) for session=%s — treating as a start failure",
                call_session.id,
            )
            mark_recording_error(db, call_session)
            return None

        meta = dict(call_session.call_metadata or {})
        # Same shape Twilio's _start_livekit_recording stores under
        # call_metadata["recording"] — call_recording_upload_service reads
        # exactly these two keys (egress_id, gcs_path) and needs no changes
        # to pick up browser-call recordings.
        meta["recording"] = {"egress_id": egress_id, "gcs_path": gcs_path}
        call_session.call_metadata = meta
        db.commit()
        logger.info(
            "[LiveKitBrowserCall] recording started: session=%s egress_id=%s",
            call_session.id, egress_id,
        )
        return egress_id
    except Exception as exc:
        logger.warning(
            "[LiveKitBrowserCall] could not start recording for session %s: %s",
            call_session.id, exc, exc_info=True,
        )
        mark_recording_error(db, call_session)
        return None


async def _stop_browser_call_recording(call_session_id: uuid.UUID, egress_id: str | None) -> None:
    """Stop the egress (if one was started) then schedule the S3 upload/finalize job."""
    if not egress_id:
        return
    try:
        from app.services.livekit_recording_service import livekit_recording_service

        await livekit_recording_service.stop_room_recording(egress_id)
    except Exception as exc:
        logger.debug("[LiveKitBrowserCall] recording stop failed: %s", exc)

    try:
        from app.services.call_recording_upload_service import schedule_recording_upload

        schedule_recording_upload(call_session_id)
    except Exception as exc:
        logger.debug("[LiveKitBrowserCall] schedule_recording_upload failed: %s", exc)


async def _load_browser_call_context(db, call_session_id: uuid.UUID):
    """Load CallSession + Agent + CallFlow — mirrors BidirectionalStreamHandler._load_session_data."""
    from app.models.call_flow import CallFlow
    from app.services.agent_service import agent_service
    from app.services.call_session_service import call_session_service

    call_session = call_session_service.get_call_session_by_id(db, call_session_id)
    if call_session is None:
        return None, None, None

    agent = None
    if call_session.agent_id:
        agent = agent_service.get_agent_by_id(db, call_session.agent_id, call_session.tenant_id)
        if agent is not None:
            agent_service.ensure_agent_prompt_ingested(db, agent)

    call_flow = None
    if call_session.call_flow_id:
        call_flow = (
            db.query(CallFlow)
            .filter(CallFlow.id == call_session.call_flow_id, CallFlow.is_deleted == False)  # noqa: E712
            .first()
        )

    return call_session, agent, call_flow


async def run_livekit_browser_call(call_session_id: uuid.UUID) -> None:
    """
    Background task entrypoint: joins the LiveKit room for `call_session_id`
    as the agent, greets the caller, and holds the conversation until the
    caller disconnects or the room connection drops.

    Called (fire-and-forget) from app.routers.sdk::demo_call_token right
    after the CallSession row + caller LiveKit token are created — never
    call this synchronously from a request handler.

    Not currently wired into public-call-token (the embedded-widget flow) —
    out of scope for this task. LiveKitBrowserCallHandler itself has no
    demo-link-specific assumptions baked in, so reusing it there later is
    just a matter of adding the same background-task call at that call
    site; this function's CallSession-loading approach would need a small
    twin (public-call-token's flow doesn't create a CallSession row today).
    """
    if not settings.LIVEKIT_ENABLED:
        logger.info("[LiveKitBrowserCall] LIVEKIT_ENABLED=false — agent will not join room=%s", call_session_id)
        return

    from app.db.session import SessionLocal

    db = SessionLocal()
    handler: LiveKitBrowserCallHandler | None = None
    voice_orchestrator = None
    stt_pipeline = None
    audio_subscriber = None
    audio_subscriber_task: asyncio.Task | None = None
    publisher: _LiveKitAgentAudioPublisher | None = None
    room_name = f"room_{call_session_id}"
    call_session = None
    # Distinguishes "the agent actually joined and the conversation ran" from
    # "we aborted before ever connecting" — the two early-return paths below
    # (missing agent, TTS publisher connect failure) must not be finalized as
    # a successful "completed" call: that would both mislabel a zero-content
    # attempt in dashboards/analytics and (via update_call_session_status)
    # trigger CRM write-back scheduling for a call nothing ever happened on.
    agent_joined = False
    recording_egress_id: str | None = None
    # Set only when the hard max-duration cap fires, so the finally block can
    # record a distinct, identifiable ended_reason instead of the generic
    # "completed" it uses for a normal caller/room disconnect.
    forced_end_reason: str | None = None

    try:
        call_session, agent, call_flow = await _load_browser_call_context(db, call_session_id)
        if call_session is None:
            logger.error("[LiveKitBrowserCall] call_session %s not found — aborting agent join", call_session_id)
            return
        if agent is None:
            logger.error(
                "[LiveKitBrowserCall] agent not found for call_session=%s — aborting agent join",
                call_session_id,
            )
            return

        handler = LiveKitBrowserCallHandler(db=db, call_session=call_session, agent=agent, call_flow=call_flow)

        # VoiceOrchestrator is constructed unmodified purely for the parts of
        # its public surface that don't require Twilio MULAW frames: it
        # creates + owns TtsPipeline (writes handler._tts_pipeline /
        # _tts_worker_task) and provides a single unified shutdown() that
        # cancels any in-flight LLM turn, drains TtsPipeline, and closes
        # whatever SttPipeline is registered on it. Its on_audio_chunk()
        # (Twilio MULAW pickup-detection + lazy Deepgram-session creation)
        # is intentionally never called here — this path always has a
        # connected LiveKit room as its only audio source, so SttPipeline is
        # created directly below and its callbacks are wired to
        # VoiceOrchestrator's own _on_interim/_on_final (the exact same
        # callables it would have wired itself inside on_audio_chunk).
        from app.voice.voice_orchestrator import VoiceOrchestrator

        voice_orchestrator = VoiceOrchestrator(handler)
        # Back-reference so ConversationOrchestrator's greeting branch (which
        # only has access to self._h, i.e. this handler) can check
        # _is_gemini_live / reach _gemini_live_session — mirrors
        # BidirectionalStreamHandler.__init__'s self._voice_orchestrator.
        handler._voice_orchestrator = voice_orchestrator

        # ── Gemini Live (native-audio speech-to-speech) fork ─────────────────
        # VoiceOrchestrator.__init__ already resolved _is_gemini_live from
        # agent.llm_model (is_gemini_live_native_audio_model) — same flag the
        # Twilio transport uses, just read here instead of re-derived, so both
        # transports agree on routing for the same agent. Unlike Twilio (which
        # lazily starts the session on the first audio chunk inside
        # on_audio_chunk — this transport's equivalent per-chunk fork point,
        # see module docstring), this transport has no such per-chunk hook: a
        # LiveKit browser call's whole audio path is set up once, up front, so
        # the session is started eagerly here, before anything audio-related
        # (including which sample rate to open the outbound publisher at) is
        # decided. On a pre-session failure, _start_gemini_live_session's own
        # fallback (_fallback_to_legacy_pipeline) flips
        # voice_orchestrator._is_gemini_live back to False and swaps
        # handler.agent.llm_model to a text model in-memory — re-read the flag
        # below so this function falls through to the normal STT/TTS
        # construction path for the rest of the call.
        is_gemini_live = voice_orchestrator._is_gemini_live
        if is_gemini_live:
            await voice_orchestrator._start_gemini_live_session(handler)
            is_gemini_live = voice_orchestrator._is_gemini_live

        # ── OpenAI Realtime (native-audio speech-to-speech) fork ─────────────
        # Parallel to the Gemini Live fork above, not shared/refactored code
        # (per this task's "strictly additive, don't touch Gemini-Live-
        # specific code" constraint) — mutually exclusive with is_gemini_live
        # since VoiceOrchestrator.__init__ resolves each provider from a
        # disjoint model-ID allow-list.
        is_openai_realtime = voice_orchestrator._is_openai_realtime
        if is_openai_realtime:
            await voice_orchestrator._start_openai_realtime_session(handler)
            is_openai_realtime = voice_orchestrator._is_openai_realtime

        # ── Connect agent audio-out (TTS / Gemini Live / OpenAI Realtime) ────
        # A native-audio call opens the publisher at the provider's own 24kHz
        # PCM output rate (no mu-law, no resampling on this path — see
        # _LiveKitAgentAudioPublisher.publish_pcm); every other call keeps the
        # existing 8kHz mu-law-derived rate. Gemini Live and OpenAI Realtime
        # both happen to use 24kHz PCM16 for this transport, so the same
        # publisher/sink plumbing serves either provider unmodified.
        if is_gemini_live:
            publisher_sample_rate = GEMINI_OUTPUT_PCM_RATE_HZ
        elif is_openai_realtime:
            publisher_sample_rate = OPENAI_REALTIME_AUDIO_RATE_HZ
        else:
            publisher_sample_rate = _AGENT_AUDIO_SAMPLE_RATE
        publisher = _LiveKitAgentAudioPublisher(room_name, sample_rate_hz=publisher_sample_rate)
        connected = await publisher.connect()
        if not connected:
            logger.error("[LiveKitBrowserCall] failed to connect TTS publisher room=%s", room_name)
            return
        handler._agent_publisher = publisher

        # ── Recording (fail-open — never aborts/degrades the call) ──────────
        recording_egress_id = await _start_browser_call_recording(db, call_session)

        # Detect the caller leaving / our own connection dropping via the
        # publisher's room object (the one connection we own directly here).
        def _on_participant_disconnected(participant) -> None:
            identity = (getattr(participant, "identity", "") or "").lower()
            if "caller" in identity:
                logger.info("[LiveKitBrowserCall] caller left room=%s — ending call", room_name)
                _track(asyncio.create_task(handler._full_shutdown()))

        def _on_room_disconnected(*_args) -> None:
            logger.info("[LiveKitBrowserCall] room connection dropped room=%s", room_name)
            _track(asyncio.create_task(handler._full_shutdown()))

        if publisher._room is not None:
            publisher._room.on("participant_disconnected", _on_participant_disconnected)
            publisher._room.on("disconnected", _on_room_disconnected)

        # ── Resolve + start inbound audio (STT, or Gemini Live) ──────────────
        from app.voice.livekit_audio_subscriber import LiveKitAudioSubscriber

        if is_gemini_live:
            # Native-audio agents skip STT entirely — SttPipeline is never
            # constructed for this call. LiveKitAudioSubscriber already
            # resamples caller audio to LINEAR16 16kHz mono
            # (_STT_INPUT_SAMPLE_RATE), exactly the format
            # GeminiLiveSession.send_audio expects, so only the sink changes
            # (via the _GeminiLiveAudioSink adapter) — no new conversion work.
            session = voice_orchestrator._gemini_live_session
            audio_subscriber = LiveKitAudioSubscriber(
                room_name=room_name,
                stt_pipeline=_GeminiLiveAudioSink(session),
                output_sample_rate=_STT_INPUT_SAMPLE_RATE,
            )
        elif is_openai_realtime:
            # Native-audio OpenAI agents skip STT entirely too. Unlike the
            # Gemini Live branch above, this requests LiveKitAudioSubscriber
            # resample caller audio to 24kHz (not 16kHz) — the rate
            # OpenAIRealtimeSession.send_audio expects when the session was
            # started with audio_format="audio/pcm" (symmetric 24kHz in/out,
            # unlike Gemini Live's asymmetric 16kHz-in/24kHz-out).
            session = voice_orchestrator._openai_realtime_session
            audio_subscriber = LiveKitAudioSubscriber(
                room_name=room_name,
                stt_pipeline=_OpenAIRealtimeAudioSink(session),
                output_sample_rate=OPENAI_REALTIME_AUDIO_RATE_HZ,
            )
        else:
            from app.core.agent_runtime import resolve_stt_runtime

            flow_lang = None
            if call_flow and isinstance(call_flow.settings, dict):
                raw = call_flow.settings.get("sttLanguageCode") or call_flow.settings.get("stt_language_code")
                if isinstance(raw, str) and raw.strip():
                    flow_lang = raw.strip()

            resolved_stt = _forced_browser_stt_runtime(
                resolve_stt_runtime(agent, flow_language_code=flow_lang, db=db)
            )

            from app.voice.stt_pipeline import SttPipeline

            stt_pipeline = SttPipeline.from_runtime_config(
                resolved=resolved_stt,
                on_interim=voice_orchestrator._on_interim,
                on_final=voice_orchestrator._on_final,
                call_session_id=handler.call_session_id,
                agent_id=handler.agent_id,
                event_bus=voice_orchestrator.stt_event_bus,
            )
            # Register onto VoiceOrchestrator so its shutdown() closes this
            # session too (it only closes a pipeline it knows about).
            voice_orchestrator._stt_pipeline = stt_pipeline
            voice_orchestrator._stt_active = True

            audio_subscriber = LiveKitAudioSubscriber(
                room_name=room_name,
                stt_pipeline=stt_pipeline,
                output_sample_rate=_STT_INPUT_SAMPLE_RATE,
            )

        audio_subscriber_task = asyncio.create_task(audio_subscriber.run())

        # From here on, the agent has actually joined and audio is flowing —
        # a subsequent failure/disconnect is a real (if possibly short) call,
        # not an aborted-before-connecting attempt.
        agent_joined = True

        # ── Mark call live + greet ───────────────────────────────────────────
        call_session.status = "in-progress"
        if not call_session.start_time:
            call_session.start_time = datetime.now(timezone.utc)
        db.commit()

        # 🎯 START CREDIT MONITORING - demo/browser calls bill the demo link's
        # owning tenant through the same plan-aware credit system as real calls.
        try:
            from app.services.credit_service import credit_service

            if str(call_session.id) not in credit_service._active_monitors:
                asyncio.create_task(credit_service.start_credit_monitoring(
                    db=db,
                    call_session_id=call_session.id,
                    tenant_id=call_session.tenant_id,
                    agent_id=call_session.agent_id
                ))
        except Exception as e:
            logger.debug("[LiveKitBrowserCall] Could not start credit monitoring: %s", e)

        await asyncio.sleep(_GREETING_DELAY_SEC)
        if not handler._stop_event.is_set():
            await handler.generate_and_stream_response("", 1.0, is_greeting=True)

        # ── Hold the call until the caller/room disconnects, or until the ──
        # hard max-duration safety cap trips (see _MAX_CALL_DURATION_SEC).
        try:
            await asyncio.wait_for(handler._stop_event.wait(), timeout=_MAX_CALL_DURATION_SEC)
        except asyncio.TimeoutError:
            forced_end_reason = "max_duration_exceeded"
            logger.warning(
                "[LiveKitBrowserCall] hard max-call-duration cap (%.0fs) reached — "
                "force-ending call room=%s call_session=%s",
                _MAX_CALL_DURATION_SEC, room_name, call_session_id,
            )
            # Mirrors the participant/room-disconnect handlers above: signal
            # the stop event and fall through to the shared finally cleanup
            # below (recording teardown, audio subscriber stop, publisher
            # disconnect — which actually drops the agent's LiveKit
            # connection/room — credit-monitor stop, and status finalize).
            handler._stop_event.set()

    except Exception as exc:
        logger.error(
            "[LiveKitBrowserCall] agent-join/conversation loop failed room=%s: %s",
            room_name, exc, exc_info=True,
        )
    finally:
        try:
            if voice_orchestrator is not None:
                await voice_orchestrator.shutdown()
        except Exception as exc:
            logger.debug("[LiveKitBrowserCall] voice_orchestrator shutdown failed: %s", exc)

        # ── Recording teardown (mirrors Twilio's _full_shutdown ordering:
        # stop the egress, then schedule the async S3 finalize/upload job) ──
        try:
            await _stop_browser_call_recording(call_session_id, recording_egress_id)
        except Exception as exc:
            logger.debug("[LiveKitBrowserCall] recording teardown failed: %s", exc)

        if audio_subscriber is not None:
            try:
                await audio_subscriber.stop()
            except Exception as exc:
                logger.debug("[LiveKitBrowserCall] audio subscriber stop failed: %s", exc)
        if audio_subscriber_task is not None and not audio_subscriber_task.done():
            audio_subscriber_task.cancel()
            try:
                await audio_subscriber_task
            except (asyncio.CancelledError, Exception):  # noqa: S110 - expected from cancelling above
                pass

        if publisher is not None:
            try:
                await publisher.disconnect()
            except Exception as exc:
                logger.debug("[LiveKitBrowserCall] publisher disconnect failed: %s", exc)

        if agent_joined:
            try:
                from app.services.credit_service import credit_service

                credit_service.stop_credit_monitoring(call_session_id)
                logger.info(
                    "[LiveKitBrowserCall] Stopped credit monitoring for call session %s",
                    call_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "[LiveKitBrowserCall] failed to stop credit monitoring call_session=%s: %s",
                    call_session_id, exc,
                )

        if call_session is not None:
            try:
                from app.services.call_session_service import call_session_service

                if agent_joined:
                    if forced_end_reason:
                        call_session_service.update_call_session_status(
                            db, call_session_id, "completed", ended_reason=forced_end_reason
                        )
                    else:
                        call_session_service.update_call_session_status(db, call_session_id, "completed")
                else:
                    call_session_service.update_call_session_status(
                        db, call_session_id, "failed", ended_reason="agent_join_failed"
                    )
            except Exception as exc:
                logger.warning(
                    "[LiveKitBrowserCall] failed to finalize call_session=%s status: %s",
                    call_session_id, exc,
                )

        db.close()
        logger.info("[LiveKitBrowserCall] call ended room=%s", room_name)
