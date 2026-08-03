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
from typing import Any

from app.core.config import settings
from app.core.logger import logger
from app.services.bidirectional_stream_service import generate_mulaw_tts
from app.services.transcript_service import transcript_service
from app.utils.audio_utils import MULAW_FRAME_BYTES, ulaw_to_linear_sample
from app.utils.ssml_utils import strip_ssml_tags
from app.voice.conversation_orchestrator import ConversationOrchestrator, VOICE_TUNABLES

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

    def __init__(self, room_name: str) -> None:
        self._room_name = room_name
        self._room: Any = None
        self._source: Any = None
        self._connected = False
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

            self._source = rtc.AudioSource(_AGENT_AUDIO_SAMPLE_RATE, 1)
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
        """Push mu-law bytes into the outgoing track, converting to PCM16 per frame."""
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
                frame = rtc.AudioFrame.create(_AGENT_AUDIO_SAMPLE_RATE, 1, len(samples))
                frame.data[:] = pcm
                await self._source.capture_frame(frame)
                self._publish_failed_logged = False
        except Exception as exc:
            if not self._publish_failed_logged:
                logger.warning(
                    "[LiveKitBrowserCall] publish_mulaw failed room=%s: %s",
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


class LiveKitBrowserCallHandler:
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
        self._elevenlabs_prev_tts_text = ""
        self._use_ssml = True

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
        Barge-in only. LiveKit rooms have their own echo cancellation and the
        default platform setting (VOICE_ENABLE_INTERIM_LLM=False) means the
        Twilio path doesn't start early LLM generation on interims either —
        so this mirrors that default behaviour rather than reimplementing the
        early-LLM path's seed/regeneration bookkeeping.
        """
        try:
            if not self._is_tts_playing or not transcript:
                return
            text = transcript.strip()
            if not text:
                return
            word_count = len(text.split())
            if word_count < self._barge_in_min_words:
                return
            min_conf = self._barge_in_min_conf_1w if word_count < 2 else self._barge_in_min_conf
            if confidence < min_conf:
                return
            logger.info(
                "[LiveKitBrowserCall] barge-in: words=%d conf=%.2f text=%r",
                word_count, confidence, text[:40],
            )
            await self._cancel_inflight_llm_response()
        except Exception as exc:
            logger.error("[LiveKitBrowserCall] _maybe_process_interim error: %s", exc, exc_info=True)

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

    async def _process_transcript(self, transcript: str, confidence: float) -> None:
        """STT final callback (wired via VoiceOrchestrator._on_final)."""
        try:
            text = (transcript or "").strip()
            if not text:
                return

            if self._is_tts_playing:
                logger.info("[LiveKitBrowserCall] barge-in (final): %r", text[:40])
                await self._cancel_inflight_llm_response()

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

    async def _prefetch_tts_audio(self, task: dict) -> bytes | None:
        """Synthesize a TTS chunk to raw mu-law bytes (see module docstring for format rationale)."""
        text = (task.get("text") or "").strip()
        if not text or self._tts_cancel.is_set():
            return None
        use_ssml = bool(task.get("use_ssml", False))
        try:
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            audio_bytes = await generate_mulaw_tts(
                text=text,
                lang=lang,
                voice=voice,
                use_chirp3_hd=True,
                speaking_rate=1.0,
                use_ssml=use_ssml,
                add_office_bg=False,
                agent=self.agent,
                db=self.db,
            )
            return audio_bytes or None
        except Exception as exc:
            logger.warning("[LiveKitBrowserCall] TTS synthesis failed for %r: %s", text[:30], exc)
            return None

    async def _stream_tts_chunk(
        self,
        text: str,
        use_ssml: bool = False,
        is_final: bool = False,
        prefetched_bytes: Any = None,
    ) -> None:
        """Publish one TTS chunk's audio into the LiveKit room (TtsPipeline's audio sink)."""
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
                mulaw_bytes = prefetched_bytes if isinstance(prefetched_bytes, (bytes, bytearray)) else None
                if mulaw_bytes is None:
                    mulaw_bytes = await self._prefetch_tts_audio({"text": text, "use_ssml": use_ssml})
                if not mulaw_bytes or self._tts_cancel.is_set():
                    return

                self._is_tts_playing = True
                logger.debug(
                    "[LiveKitBrowserCall] LiveKit playback: publishing %d mu-law bytes "
                    "(call_session_id=%s)", len(mulaw_bytes), self.call_session_id,
                )
                await publisher.publish_mulaw(mulaw_bytes, cancel=self._tts_cancel)
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

        # ── Connect agent audio-out (TTS) ───────────────────────────────────
        publisher = _LiveKitAgentAudioPublisher(room_name)
        connected = await publisher.connect()
        if not connected:
            logger.error("[LiveKitBrowserCall] failed to connect TTS publisher room=%s", room_name)
            return
        handler._agent_publisher = publisher

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

        # ── Resolve + start STT ─────────────────────────────────────────────
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

        from app.voice.livekit_audio_subscriber import LiveKitAudioSubscriber

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

        await asyncio.sleep(_GREETING_DELAY_SEC)
        if not handler._stop_event.is_set():
            await handler.generate_and_stream_response("", 1.0, is_greeting=True)

        # ── Hold the call until the caller/room disconnects ─────────────────
        await handler._stop_event.wait()

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

        if call_session is not None:
            try:
                from app.services.call_session_service import call_session_service

                if agent_joined:
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
