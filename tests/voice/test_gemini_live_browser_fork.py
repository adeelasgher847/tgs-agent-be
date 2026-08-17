"""
Browser-transport (LiveKit) fork-point tests for Gemini Live native-audio
routing — the mirror-image counterpart of
tests/voice/test_gemini_live_twilio_fork.py.

Covers, per the integration task write-up:
  - native-audio agents never construct SttPipeline on this transport either
  - non-native-audio agent behavior on this transport is unchanged
  - caller audio flows to GeminiLiveSession.send_audio via the
    _GeminiLiveAudioSink adapter (LiveKitAudioSubscriber unmodified)
  - agent audio flows to _LiveKitAgentAudioPublisher.publish_pcm (raw
    PCM16/24kHz, no mu-law, no resampling) — NOT publish_mulaw
  - the outbound publisher is opened at Gemini's 24kHz output rate for a
    native-audio call, and at the usual 8kHz rate otherwise
  - barge-in (on_interrupted) clears the LiveKit AudioSource's own playout
    queue directly (no _send_twilio_clear_event on this transport)
  - transcript callbacks buffer every fragment (finished flag is unreliable)
    and flush to call_transcript via _add_to_transcript on turn_complete /
    next-turn's first output fragment / interrupted / shutdown (shared
    VoiceOrchestrator logic — reasserted here against a real
    LiveKitBrowserCallHandler instance)
  - _start_gemini_live_session must build the Gemini Live system_instruction
    from ConversationOrchestrator.build_system_prompt (this transport's own
    real prompt-assembly), never from Twilio's build_system_prompt — the
    mirror-image regression guard of the bug fixed on the Twilio side.

GeminiLiveSession itself is mocked throughout — no live API calls.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm_models import GEMINI_LIVE_MODELS
from app.voice.gemini_live_audio_bridge import GEMINI_OUTPUT_PCM_RATE_HZ
from app.voice.livekit_browser_call_handler import (
    LiveKitBrowserCallHandler,
    _AGENT_AUDIO_SAMPLE_RATE,
    _GeminiLiveAudioSink,
    _LiveKitAgentAudioPublisher,
    run_livekit_browser_call,
)
from app.voice.voice_orchestrator import VoiceOrchestrator

NATIVE_AUDIO_MODEL = next(iter(GEMINI_LIVE_MODELS))


def _base_handler(llm_model: str | None) -> LiveKitBrowserCallHandler:
    db = MagicMock()
    call_session = MagicMock()
    call_session.id = uuid.uuid4()
    call_session.user_id = uuid.uuid4()
    call_session.tenant_id = uuid.uuid4()
    call_session.call_metadata = {}
    call_session.call_transcript = []

    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.name = "Demo Agent"
    agent.language = "en"
    agent.voice_type = "female"
    agent.llm_model = llm_model
    agent.greeting_message = None
    agent.first_message = None

    return LiveKitBrowserCallHandler(db=db, call_session=call_session, agent=agent, call_flow=None)


# ─────────────────────────────────────────────────────────────────────────
# _LiveKitAgentAudioPublisher.publish_pcm
# ─────────────────────────────────────────────────────────────────────────

class TestPublishPcm:
    @pytest.mark.asyncio
    async def test_noop_when_not_connected(self):
        publisher = _LiveKitAgentAudioPublisher("room_x", sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)
        await publisher.publish_pcm(b"\x00\x01" * 10, sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)
        # No exception, no self._source ever touched (still None).
        assert publisher._source is None

    @pytest.mark.asyncio
    async def test_noop_when_cancel_already_set(self):
        publisher = _LiveKitAgentAudioPublisher("room_x", sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)
        publisher._connected = True
        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock()
        publisher._source = mock_source

        cancel = asyncio.Event()
        cancel.set()

        await publisher.publish_pcm(b"\x00\x01" * 10, sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ, cancel=cancel)

        mock_source.capture_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_publishes_raw_pcm_frame_at_publisher_rate(self):
        publisher = _LiveKitAgentAudioPublisher("room_x", sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)
        publisher._connected = True
        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock()
        publisher._source = mock_source

        mock_rtc = MagicMock()
        mock_frame = MagicMock()  # frame.data is a plain MagicMock so .cast(...)[:] = ... never raises
        mock_rtc.AudioFrame.create.return_value = mock_frame

        pcm = b"\x01\x02" * 10  # 10 samples
        with patch.dict("sys.modules", {"livekit": MagicMock(rtc=mock_rtc)}):
            await publisher.publish_pcm(pcm, sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)

        mock_rtc.AudioFrame.create.assert_called_once_with(GEMINI_OUTPUT_PCM_RATE_HZ, 1, 10)
        mock_source.capture_frame.assert_awaited_once_with(mock_frame)

    @pytest.mark.asyncio
    async def test_drops_trailing_odd_byte_rather_than_raising(self):
        publisher = _LiveKitAgentAudioPublisher("room_x", sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)
        publisher._connected = True
        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock()
        publisher._source = mock_source

        mock_rtc = MagicMock()
        mock_rtc.AudioFrame.create.return_value = MagicMock(data=bytearray(20))

        pcm = (b"\x01\x02" * 10) + b"\x99"  # trailing odd byte
        with patch.dict("sys.modules", {"livekit": MagicMock(rtc=mock_rtc)}):
            await publisher.publish_pcm(pcm, sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)

        mock_rtc.AudioFrame.create.assert_called_once_with(GEMINI_OUTPUT_PCM_RATE_HZ, 1, 10)

    @pytest.mark.asyncio
    async def test_sample_rate_mismatch_drops_chunk_without_publishing(self):
        """
        A publisher opened at 24kHz must never be handed a chunk claiming a
        different rate — no resampling is performed by publish_pcm (see its
        docstring); mismatches are dropped, not silently mis-played.
        """
        publisher = _LiveKitAgentAudioPublisher("room_x", sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ)
        publisher._connected = True
        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock()
        publisher._source = mock_source

        await publisher.publish_pcm(b"\x01\x02" * 10, sample_rate_hz=_AGENT_AUDIO_SAMPLE_RATE)

        mock_source.capture_frame.assert_not_called()

    def test_default_sample_rate_matches_legacy_tts_constant(self):
        publisher = _LiveKitAgentAudioPublisher("room_x")
        assert publisher.sample_rate == _AGENT_AUDIO_SAMPLE_RATE


# ─────────────────────────────────────────────────────────────────────────
# _GeminiLiveAudioSink — LiveKitAudioSubscriber's sink adapter
# ─────────────────────────────────────────────────────────────────────────

class TestGeminiLiveAudioSink:
    @pytest.mark.asyncio
    async def test_forwards_to_session_send_audio(self):
        session = MagicMock()
        session.send_audio = AsyncMock()
        sink = _GeminiLiveAudioSink(session)

        await sink.feed_audio_chunk(b"\x01\x02pcm16-16k")

        session.send_audio.assert_awaited_once_with(b"\x01\x02pcm16-16k")

    @pytest.mark.asyncio
    async def test_empty_chunk_is_a_noop(self):
        session = MagicMock()
        session.send_audio = AsyncMock()
        sink = _GeminiLiveAudioSink(session)

        await sink.feed_audio_chunk(b"")

        session.send_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_audio_exception_is_swallowed(self):
        session = MagicMock()
        session.send_audio = AsyncMock(side_effect=RuntimeError("session closed"))
        sink = _GeminiLiveAudioSink(session)

        # Must not raise.
        await sink.feed_audio_chunk(b"\x01\x02")


# ─────────────────────────────────────────────────────────────────────────
# LiveKitBrowserCallHandler Gemini Live audio-out / barge-in glue
# ─────────────────────────────────────────────────────────────────────────

class TestHandlerGeminiLiveGlue:
    @pytest.mark.asyncio
    async def test_publish_gemini_live_audio_chunk_uses_publish_pcm_at_24k(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        publisher = MagicMock()
        publisher.connected = True
        publisher.publish_pcm = AsyncMock()
        h._agent_publisher = publisher

        cancel = asyncio.Event()
        await h._publish_gemini_live_audio_chunk(b"pcm24k-bytes", cancel)

        publisher.publish_pcm.assert_awaited_once_with(
            b"pcm24k-bytes", sample_rate_hz=GEMINI_OUTPUT_PCM_RATE_HZ, cancel=cancel
        )

    @pytest.mark.asyncio
    async def test_publish_gemini_live_audio_chunk_noop_when_no_publisher(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        h._agent_publisher = None
        # Must not raise.
        await h._publish_gemini_live_audio_chunk(b"pcm24k-bytes", asyncio.Event())

    @pytest.mark.asyncio
    async def test_clear_gemini_live_playout_queue_clears_source(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        publisher = MagicMock()
        publisher._source = MagicMock()
        h._agent_publisher = publisher

        await h._clear_gemini_live_playout_queue()

        publisher._source.clear_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_gemini_live_playout_queue_noop_when_no_publisher(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        h._agent_publisher = None
        await h._clear_gemini_live_playout_queue()  # must not raise

    @pytest.mark.asyncio
    async def test_clear_gemini_live_playout_queue_swallows_exception(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        publisher = MagicMock()
        publisher._source = MagicMock()
        publisher._source.clear_queue.side_effect = RuntimeError("ffi handle disposed")
        h._agent_publisher = publisher

        await h._clear_gemini_live_playout_queue()  # must not raise

    def test_handler_has_no_twilio_only_gemini_hooks(self):
        """
        VoiceOrchestrator._on_gemini_live_audio_chunk / _on_gemini_live_interrupted
        duck-type on hasattr(h, "_stream_live_audio_chunk") /
        hasattr(h, "_send_twilio_clear_event") to pick the Twilio branch —
        the browser handler must NOT define either, or it would silently
        (and incorrectly) take the Twilio mu-law path instead of its own.
        """
        h = _base_handler(NATIVE_AUDIO_MODEL)
        assert not hasattr(h, "_stream_live_audio_chunk")
        assert not hasattr(h, "_send_twilio_clear_event")


# ─────────────────────────────────────────────────────────────────────────
# VoiceOrchestrator's transport-aware branch, exercised against a real
# LiveKitBrowserCallHandler (no build_system_prompt / _stream_live_audio_chunk
# / _send_twilio_clear_event attributes — natural duck-typed browser fork).
# ─────────────────────────────────────────────────────────────────────────

class TestVoiceOrchestratorAudioConversionRoutingBrowser:
    def test_is_gemini_live_resolved_from_agent_llm_model(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        assert orch._is_gemini_live is True

    def test_non_native_model_is_gemini_live_false(self):
        h = _base_handler("gpt-4o-mini")
        orch = VoiceOrchestrator(h)
        assert orch._is_gemini_live is False

    def test_on_audio_chunk_callback_publishes_raw_pcm_not_mulaw(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._publish_gemini_live_audio_chunk = AsyncMock()

        asyncio.run(orch._on_gemini_live_audio_chunk(b"pcm24k-in"))

        h._publish_gemini_live_audio_chunk.assert_awaited_once_with(
            b"pcm24k-in", orch._gemini_live_cancel
        )

    def test_on_audio_chunk_callback_gated_by_cancel_flag(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._publish_gemini_live_audio_chunk = AsyncMock()
        orch._gemini_live_cancel.set()

        asyncio.run(orch._on_gemini_live_audio_chunk(b"pcm24k-in"))

        h._publish_gemini_live_audio_chunk.assert_not_awaited()

    def test_interrupted_clears_livekit_queue_not_twilio_event(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._clear_gemini_live_playout_queue = AsyncMock()
        orch._gemini_live_first_audio_marked = True

        asyncio.run(orch._on_gemini_live_interrupted())

        h._clear_gemini_live_playout_queue.assert_awaited_once()
        assert not orch._gemini_live_cancel.is_set()
        assert orch._gemini_live_first_audio_marked is False

    def test_input_transcript_fragments_buffered_until_flush(self):
        """Mirrors tests/voice/test_gemini_live_twilio_fork.py::
        TestTranscriptCallbacks — the buffering/flush behavior is identical
        on both transports since it lives entirely in VoiceOrchestrator."""
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._add_to_transcript = AsyncMock()

        asyncio.run(orch._on_gemini_live_input_transcript("partial", False))
        h._add_to_transcript.assert_not_awaited()
        assert orch._gemini_live_input_buffer == ["partial"]

        asyncio.run(orch._on_gemini_live_input_transcript(" hello there", True))
        h._add_to_transcript.assert_not_awaited()

        asyncio.run(orch._on_gemini_live_turn_complete())
        h._add_to_transcript.assert_awaited_once_with("client", "partial hello there", "speech")

    def test_output_transcript_fragments_buffered_until_turn_complete(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._add_to_transcript = AsyncMock()

        asyncio.run(orch._on_gemini_live_output_transcript("partial reply", False))
        h._add_to_transcript.assert_not_awaited()

        asyncio.run(orch._on_gemini_live_output_transcript(" Sure, I can help.", True))
        h._add_to_transcript.assert_not_awaited()

        asyncio.run(orch._on_gemini_live_turn_complete())
        h._add_to_transcript.assert_awaited_once_with(
            "agent", "partial reply Sure, I can help.", "agent_response"
        )

    def test_mid_call_error_ends_call_via_full_shutdown(self):
        from app.services.vertex_gemini_service import VertexLlmError, VertexLlmErrorType

        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._full_shutdown = AsyncMock()

        asyncio.run(orch._on_gemini_live_error(VertexLlmError("dropped", VertexLlmErrorType.UNKNOWN)))

        h._full_shutdown.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# System-instruction source — mirror-image regression guard: browser must
# use ConversationOrchestrator.build_system_prompt, never Twilio's method.
# ─────────────────────────────────────────────────────────────────────────

class TestGeminiLiveSessionUsesConversationOrchestratorPromptAssembly:
    def test_start_gemini_live_session_uses_conversation_orchestrator(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        assert not hasattr(h, "build_system_prompt")  # duck-type must fail for browser

        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.voice.conversation_orchestrator.ConversationOrchestrator.build_system_prompt",
            new=AsyncMock(return_value="browser system prompt"),
        ) as browser_build:
            asyncio.run(orch._start_gemini_live_session(h))

        browser_build.assert_awaited_once()
        fake_session.start.assert_awaited_once()
        assert (
            fake_session.start.call_args.kwargs["system_instruction"]
            == "browser system prompt"
        )
        assert orch._gemini_live_session is fake_session

    def test_no_calendly_tool_calling_on_browser_transport(self):
        """
        LiveKitBrowserCallHandler has no BookingMixin at all (Calendly is
        Twilio-only, per booking_mixin.py's own docstring) — confirms
        _start_gemini_live_session's hasattr(h, "_calendly_enabled") guard
        correctly passes tools=None / on_tool_call=None on this transport,
        the mirror-image of
        tests/voice/test_gemini_live_twilio_fork.py::TestCalendlyToolCallWiring.
        """
        h = _base_handler(NATIVE_AUDIO_MODEL)
        assert not hasattr(h, "_calendly_enabled")

        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.voice.conversation_orchestrator.ConversationOrchestrator.build_system_prompt",
            new=AsyncMock(return_value="browser system prompt"),
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["tools"] is None
        assert kwargs["on_tool_call"] is None


# ─────────────────────────────────────────────────────────────────────────
# run_livekit_browser_call — full-lifecycle fork wiring
# ─────────────────────────────────────────────────────────────────────────

class TestRunLiveKitBrowserCallGeminiLiveFork:
    def _call_session(self, call_session_id):
        call_session = MagicMock()
        call_session.id = call_session_id
        call_session.call_flow_id = None
        call_session.call_metadata = {}
        return call_session

    def _agent(self, llm_model):
        agent = MagicMock()
        agent.id = uuid.uuid4()
        agent.llm_model = llm_model
        agent.stt_provider_slug = None
        agent.stt_model_id = None
        return agent

    @pytest.mark.asyncio
    async def test_native_audio_call_never_constructs_stt_pipeline_and_opens_24k_publisher(self):
        call_session_id = uuid.uuid4()
        call_session = self._call_session(call_session_id)
        agent = self._agent(NATIVE_AUDIO_MODEL)
        mock_db = MagicMock()

        publisher_ctor_calls: list = []

        class _FakePublisher:
            def __init__(self, room_name, sample_rate_hz=_AGENT_AUDIO_SAMPLE_RATE):
                publisher_ctor_calls.append(sample_rate_hz)
                self._room_name = room_name
                self._sample_rate = sample_rate_hz
                self._room = MagicMock()
                self.connected = True

            async def connect(self):
                return True

            async def disconnect(self):
                return None

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        mock_subscriber_instance = MagicMock()
        mock_subscriber_instance.run = AsyncMock()
        mock_subscriber_instance.stop = AsyncMock()
        subscriber_ctor_kwargs: list = []

        class _FakeSubscriber:
            def __init__(self, **kwargs):
                subscriber_ctor_kwargs.append(kwargs)

            async def run(self):
                return None

            async def stop(self):
                return None

        async def _immediately_stop(self, *_args, **_kwargs):
            self._stop_event.set()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, agent, None)),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._LiveKitAgentAudioPublisher",
                 new=_FakePublisher,
             ), \
             patch(
                 "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
             ), \
             patch(
                 "app.voice.conversation_orchestrator.ConversationOrchestrator.build_system_prompt",
                 new=AsyncMock(return_value="browser system prompt"),
             ), \
             patch("app.voice.stt_pipeline.SttPipeline.from_runtime_config") as mock_stt_ctor, \
             patch(
                 "app.voice.livekit_audio_subscriber.LiveKitAudioSubscriber",
                 new=_FakeSubscriber,
             ), \
             patch("app.services.call_session_service.call_session_service"), \
             patch(
                 "app.voice.livekit_browser_call_handler._start_browser_call_recording",
                 new=AsyncMock(return_value=None),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._stop_browser_call_recording",
                 new=AsyncMock(),
             ), \
             patch.object(
                 LiveKitBrowserCallHandler,
                 "generate_and_stream_response",
                 new=_immediately_stop,
             ):
            mock_settings.LIVEKIT_ENABLED = True
            await asyncio.wait_for(run_livekit_browser_call(call_session_id), timeout=5.0)

        # SttPipeline never constructed for a native-audio call.
        mock_stt_ctor.assert_not_called()
        # Publisher opened at Gemini's 24kHz output rate, not the usual 8kHz.
        assert publisher_ctor_calls == [GEMINI_OUTPUT_PCM_RATE_HZ]
        # The audio subscriber was fed a _GeminiLiveAudioSink, not a raw SttPipeline.
        assert len(subscriber_ctor_kwargs) == 1
        sink = subscriber_ctor_kwargs[0]["stt_pipeline"]
        assert isinstance(sink, _GeminiLiveAudioSink)
        assert sink._session is fake_session
        fake_session.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_native_call_still_constructs_stt_pipeline_and_8k_publisher(self):
        """Existing browser call paths must be provably unchanged."""
        from app.core.agent_runtime import ResolvedSttRuntime

        call_session_id = uuid.uuid4()
        call_session = self._call_session(call_session_id)
        agent = self._agent("gpt-4o-mini")
        mock_db = MagicMock()

        publisher_ctor_calls: list = []

        class _FakePublisher:
            def __init__(self, room_name, sample_rate_hz=_AGENT_AUDIO_SAMPLE_RATE):
                publisher_ctor_calls.append(sample_rate_hz)
                self._room = MagicMock()
                self.connected = True

            async def connect(self):
                return True

            async def disconnect(self):
                return None

        mock_subscriber_instance = MagicMock()
        mock_subscriber_instance.run = AsyncMock()
        mock_subscriber_instance.stop = AsyncMock()

        async def _immediately_stop(self, *_args, **_kwargs):
            self._stop_event.set()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, agent, None)),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._LiveKitAgentAudioPublisher",
                 new=_FakePublisher,
             ), \
             patch("app.core.agent_runtime.resolve_stt_runtime") as mock_resolve_stt, \
             patch(
                 "app.voice.stt_pipeline.SttPipeline.from_runtime_config",
                 return_value=MagicMock(),
             ) as mock_stt_ctor, \
             patch(
                 "app.voice.livekit_audio_subscriber.LiveKitAudioSubscriber",
                 return_value=mock_subscriber_instance,
             ), \
             patch("app.services.call_session_service.call_session_service"), \
             patch(
                 "app.voice.livekit_browser_call_handler._start_browser_call_recording",
                 new=AsyncMock(return_value=None),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._stop_browser_call_recording",
                 new=AsyncMock(),
             ), \
             patch.object(
                 LiveKitBrowserCallHandler,
                 "generate_and_stream_response",
                 new=_immediately_stop,
             ):
            mock_settings.LIVEKIT_ENABLED = True
            mock_resolve_stt.return_value = ResolvedSttRuntime(
                provider_slug="deepgram",
                model_id="nova-3",
                language_code="en",
                sample_rate_hz=8000,
                encoding="MULAW",
                silence_threshold_ms=1500,
                api_config={},
            )
            await asyncio.wait_for(run_livekit_browser_call(call_session_id), timeout=5.0)

        mock_stt_ctor.assert_called_once()
        assert publisher_ctor_calls == [_AGENT_AUDIO_SAMPLE_RATE]

    @pytest.mark.asyncio
    async def test_pre_session_failure_falls_back_to_legacy_stt_pipeline(self):
        """
        A GeminiLiveSession.start() failure before the session became usable
        must fall back to the normal STT/TTS construction path for the rest
        of the call (per _fallback_to_legacy_pipeline), not leave the
        browser call silent.
        """
        from app.core.agent_runtime import ResolvedSttRuntime
        from app.services.vertex_gemini_service import VertexLlmError, VertexLlmErrorType

        call_session_id = uuid.uuid4()
        call_session = self._call_session(call_session_id)
        agent = self._agent(NATIVE_AUDIO_MODEL)
        mock_db = MagicMock()

        publisher_ctor_calls: list = []

        class _FakePublisher:
            def __init__(self, room_name, sample_rate_hz=_AGENT_AUDIO_SAMPLE_RATE):
                publisher_ctor_calls.append(sample_rate_hz)
                self._room = MagicMock()
                self.connected = True

            async def connect(self):
                return True

            async def disconnect(self):
                return None

        fake_session = MagicMock()
        fake_session.start = AsyncMock(side_effect=VertexLlmError("boom", VertexLlmErrorType.QUOTA))

        mock_subscriber_instance = MagicMock()
        mock_subscriber_instance.run = AsyncMock()
        mock_subscriber_instance.stop = AsyncMock()

        async def _immediately_stop(self, *_args, **_kwargs):
            self._stop_event.set()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, agent, None)),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._LiveKitAgentAudioPublisher",
                 new=_FakePublisher,
             ), \
             patch(
                 "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
             ), \
             patch(
                 "app.voice.conversation_orchestrator.ConversationOrchestrator.build_system_prompt",
                 new=AsyncMock(return_value="browser system prompt"),
             ), \
             patch("app.core.agent_runtime.resolve_stt_runtime") as mock_resolve_stt, \
             patch(
                 "app.voice.stt_pipeline.SttPipeline.from_runtime_config",
                 return_value=MagicMock(),
             ) as mock_stt_ctor, \
             patch(
                 "app.voice.livekit_audio_subscriber.LiveKitAudioSubscriber",
                 return_value=mock_subscriber_instance,
             ), \
             patch("app.services.call_session_service.call_session_service"), \
             patch(
                 "app.voice.livekit_browser_call_handler._start_browser_call_recording",
                 new=AsyncMock(return_value=None),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._stop_browser_call_recording",
                 new=AsyncMock(),
             ), \
             patch.object(
                 LiveKitBrowserCallHandler,
                 "generate_and_stream_response",
                 new=_immediately_stop,
             ):
            mock_settings.LIVEKIT_ENABLED = True
            mock_resolve_stt.return_value = ResolvedSttRuntime(
                provider_slug="deepgram",
                model_id="nova-3",
                language_code="en",
                sample_rate_hz=8000,
                encoding="MULAW",
                silence_threshold_ms=1500,
                api_config={},
            )
            await asyncio.wait_for(run_livekit_browser_call(call_session_id), timeout=5.0)

        # Fallback happened -> publisher opened at the normal 8kHz rate, and
        # the legacy SttPipeline construction path ran after all.
        assert publisher_ctor_calls == [_AGENT_AUDIO_SAMPLE_RATE]
        mock_stt_ctor.assert_called_once()
        assert agent.llm_model == "gemini-2.5-flash"


# ─────────────────────────────────────────────────────────────────────────
# Greeting must bypass external TTS for native-audio agents (mirror-image
# regression guard of the same bug fixed on the Twilio side — see
# tests/voice/test_gemini_live_twilio_fork.py::TestGreetingBypassesExternalTtsForNativeAudio
# for the full bug writeup: the auto-greeting call site
# (run_livekit_browser_call -> handler.generate_and_stream_response(...,
# is_greeting=True)) originally queued the scripted greeting through
# TtsPipeline unconditionally, producing pitch-shifted audio on this
# transport since it flowed through publish_mulaw (assumes 8kHz) even though
# the outbound AudioSource was opened at Gemini's 24kHz rate for these calls.
# ─────────────────────────────────────────────────────────────────────────

class TestBrowserGreetingBypassesExternalTtsForNativeAudio:
    def _greeting_handler(self, *, is_gemini_live: bool) -> LiveKitBrowserCallHandler:
        h = _base_handler(NATIVE_AUDIO_MODEL if is_gemini_live else "gpt-4o-mini")
        h.agent.greeting_message = "Hi, thanks for calling the demo."
        h.call_session.call_type = "inbound"
        h._add_to_transcript = AsyncMock()
        h._tts_pipeline = MagicMock()
        h._tts_pipeline.queue_tts = AsyncMock()

        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        vo = MagicMock()
        vo._is_gemini_live = is_gemini_live
        vo._gemini_live_session = fake_session if is_gemini_live else None
        h._voice_orchestrator = vo
        h._fake_session = fake_session
        return h

    @pytest.mark.asyncio
    async def test_native_audio_greeting_uses_live_session_send_text_not_tts(self):
        h = self._greeting_handler(is_gemini_live=True)

        await h.generate_and_stream_response("", 1.0, is_greeting=True)

        h._fake_session.send_text.assert_awaited_once_with(
            "Hi, thanks for calling the demo.", turn_complete=True
        )
        h._tts_pipeline.queue_tts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_native_audio_greeting_still_uses_external_tts(self):
        h = self._greeting_handler(is_gemini_live=False)

        await h.generate_and_stream_response("", 1.0, is_greeting=True)

        h._tts_pipeline.queue_tts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_voice_orchestrator_attribute_falls_back_to_external_tts(self):
        h = self._greeting_handler(is_gemini_live=False)
        del h._voice_orchestrator

        await h.generate_and_stream_response("", 1.0, is_greeting=True)

        h._tts_pipeline.queue_tts.assert_awaited_once()
