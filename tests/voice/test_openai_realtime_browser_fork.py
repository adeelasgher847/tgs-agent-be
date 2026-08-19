"""
Browser-transport (LiveKit) fork-point tests for OpenAI Realtime native-audio
routing — the mirror-image counterpart of
tests/voice/test_openai_realtime_twilio_fork.py, structured like
tests/voice/test_gemini_live_browser_fork.py.

Covers:
  - native-audio OpenAI agents never construct SttPipeline on this transport
  - caller audio flows to OpenAIRealtimeSession.send_audio via the
    _OpenAIRealtimeAudioSink adapter (LiveKitAudioSubscriber unmodified,
    requested at 24kHz — not Gemini Live's 16kHz)
  - agent audio flows through the SAME generic 24kHz-PCM publish sink
    Gemini Live's browser wiring already established
    (_publish_gemini_live_audio_chunk) — reused unmodified, not duplicated,
    since OpenAI's audio/pcm output is also PCM16/24kHz
  - the outbound publisher is opened at 24kHz for a native-audio OpenAI call
  - barge-in clears the LiveKit AudioSource's own playout queue directly,
    same shared method Gemini Live uses on this transport
  - _start_openai_realtime_session builds system_instruction from
    ConversationOrchestrator.build_system_prompt (this transport's own real
    prompt-assembly), never Twilio's build_system_prompt
  - no Calendly tool-calling on this transport (no BookingMixin here)

OpenAIRealtimeSession itself is mocked throughout — no live API calls.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm_models import OPENAI_REALTIME_MODELS
from app.services.openai_realtime_service import LIVEKIT_AUDIO_RATE_HZ as OPENAI_REALTIME_AUDIO_RATE_HZ
from app.voice.livekit_browser_call_handler import (
    LiveKitBrowserCallHandler,
    _AGENT_AUDIO_SAMPLE_RATE,
    _OpenAIRealtimeAudioSink,
    run_livekit_browser_call,
)
from app.voice.voice_orchestrator import VoiceOrchestrator

NATIVE_AUDIO_MODEL = next(iter(OPENAI_REALTIME_MODELS))


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
# _OpenAIRealtimeAudioSink — LiveKitAudioSubscriber's sink adapter
# ─────────────────────────────────────────────────────────────────────────


class TestOpenAIRealtimeAudioSink:
    @pytest.mark.asyncio
    async def test_forwards_to_session_send_audio(self):
        session = MagicMock()
        session.send_audio = AsyncMock()
        sink = _OpenAIRealtimeAudioSink(session)

        await sink.feed_audio_chunk(b"\x01\x02pcm16-24k")

        session.send_audio.assert_awaited_once_with(b"\x01\x02pcm16-24k")

    @pytest.mark.asyncio
    async def test_empty_chunk_is_a_noop(self):
        session = MagicMock()
        session.send_audio = AsyncMock()
        sink = _OpenAIRealtimeAudioSink(session)

        await sink.feed_audio_chunk(b"")

        session.send_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_audio_exception_is_swallowed(self):
        session = MagicMock()
        session.send_audio = AsyncMock(side_effect=RuntimeError("session closed"))
        sink = _OpenAIRealtimeAudioSink(session)

        # Must not raise.
        await sink.feed_audio_chunk(b"\x01\x02")


# ─────────────────────────────────────────────────────────────────────────
# VoiceOrchestrator's transport-aware branch, exercised against a real
# LiveKitBrowserCallHandler — same generic sink methods Gemini Live's
# browser wiring already established, reused unmodified.
# ─────────────────────────────────────────────────────────────────────────


class TestVoiceOrchestratorAudioConversionRoutingBrowser:
    def test_is_openai_realtime_resolved_from_agent_llm_model(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        assert orch._is_openai_realtime is True
        assert orch._is_gemini_live is False

    def test_non_native_model_is_openai_realtime_false(self):
        h = _base_handler("gpt-4o-mini")
        orch = VoiceOrchestrator(h)
        assert orch._is_openai_realtime is False

    def test_on_audio_chunk_callback_reuses_generic_publish_pcm_sink(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._publish_gemini_live_audio_chunk = AsyncMock()

        asyncio.run(orch._on_openai_realtime_audio_chunk(b"pcm24k-in"))

        h._publish_gemini_live_audio_chunk.assert_awaited_once_with(
            b"pcm24k-in", orch._openai_realtime_cancel, sample_rate_hz=24000
        )

    def test_on_audio_chunk_callback_gated_by_cancel_flag(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._publish_gemini_live_audio_chunk = AsyncMock()
        orch._openai_realtime_cancel.set()

        asyncio.run(orch._on_openai_realtime_audio_chunk(b"pcm24k-in"))

        h._publish_gemini_live_audio_chunk.assert_not_awaited()

    def test_interrupted_clears_livekit_queue_not_twilio_event(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._clear_gemini_live_playout_queue = AsyncMock()
        orch._openai_realtime_first_audio_marked = True

        asyncio.run(orch._on_openai_realtime_interrupted())

        h._clear_gemini_live_playout_queue.assert_awaited_once()
        assert not orch._openai_realtime_cancel.is_set()
        assert orch._openai_realtime_first_audio_marked is False

    def test_input_transcript_written_immediately_no_buffering(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._add_to_transcript = AsyncMock()

        asyncio.run(orch._on_openai_realtime_input_transcript("hello there"))

        h._add_to_transcript.assert_awaited_once_with("client", "hello there", "speech")

    def test_output_transcript_written_immediately_no_buffering(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._add_to_transcript = AsyncMock()

        asyncio.run(orch._on_openai_realtime_output_transcript("Sure, I can help."))

        h._add_to_transcript.assert_awaited_once_with("agent", "Sure, I can help.", "agent_response")

    def test_mid_call_error_ends_call_via_full_shutdown(self):
        from app.services.openai_realtime_service import OpenAIRealtimeError, OpenAIRealtimeErrorType

        h = _base_handler(NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h._full_shutdown = AsyncMock()

        asyncio.run(
            orch._on_openai_realtime_error(
                OpenAIRealtimeError("dropped", OpenAIRealtimeErrorType.UNKNOWN)
            )
        )

        h._full_shutdown.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────
# System-instruction source — mirror-image regression guard: browser must
# use ConversationOrchestrator.build_system_prompt, never Twilio's method.
# ─────────────────────────────────────────────────────────────────────────


class TestOpenAIRealtimeSessionUsesConversationOrchestratorPromptAssembly:
    def test_start_openai_realtime_session_uses_conversation_orchestrator(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        assert not hasattr(h, "build_system_prompt")  # duck-type must fail for browser

        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.voice.conversation_orchestrator.ConversationOrchestrator.build_system_prompt",
            new=AsyncMock(return_value="browser system prompt"),
        ) as browser_build:
            asyncio.run(orch._start_openai_realtime_session(h))

        browser_build.assert_awaited_once()
        fake_session.start.assert_awaited_once()
        assert (
            fake_session.start.call_args.kwargs["system_instruction"]
            == "browser system prompt"
        )
        assert orch._openai_realtime_session is fake_session

    def test_no_calendly_tool_calling_on_browser_transport(self):
        h = _base_handler(NATIVE_AUDIO_MODEL)
        assert not hasattr(h, "_calendly_enabled")

        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.voice.conversation_orchestrator.ConversationOrchestrator.build_system_prompt",
            new=AsyncMock(return_value="browser system prompt"),
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["tools"] is None
        assert kwargs["on_tool_call"] is None


# ─────────────────────────────────────────────────────────────────────────
# run_livekit_browser_call — full-lifecycle fork wiring
# ─────────────────────────────────────────────────────────────────────────


class TestRunLiveKitBrowserCallOpenAIRealtimeFork:
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
                 "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
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
        # Publisher opened at OpenAI's 24kHz output rate, not the usual 8kHz.
        assert publisher_ctor_calls == [OPENAI_REALTIME_AUDIO_RATE_HZ]
        # The audio subscriber was fed an _OpenAIRealtimeAudioSink at 24kHz,
        # not a raw SttPipeline (and not Gemini's 16kHz-targeted sink).
        assert len(subscriber_ctor_kwargs) == 1
        kwargs = subscriber_ctor_kwargs[0]
        sink = kwargs["stt_pipeline"]
        assert isinstance(sink, _OpenAIRealtimeAudioSink)
        assert sink._session is fake_session
        assert kwargs["output_sample_rate"] == OPENAI_REALTIME_AUDIO_RATE_HZ
        fake_session.start.assert_awaited_once()
