"""
Twilio-transport fork-point tests for Gemini Live native-audio routing
(VoiceOrchestrator.on_audio_chunk -> _feed_gemini_live_audio /
_start_gemini_live_session / _on_gemini_live_* callbacks).

Covers, per the integration task write-up:
  - native-audio agents never construct SttPipeline
  - non-native-audio agent behavior is unchanged (SttPipeline still lazily
    created, GeminiLiveSession never touched)
  - caller/agent audio flows through the right conversion functions
    (mulaw8k_to_pcm16_16k / pcm16_24k_to_mulaw8k)
  - barge-in (on_interrupted) gates the outbound send loop
  - transcript callbacks write to call_transcript via _add_to_transcript
    only on finished=True chunks
  - pre-session GeminiLiveSession.start() failure falls back to the legacy
    text pipeline (in-memory agent.llm_model swap), not a silent hang

GeminiLiveSession itself is mocked throughout — no live API calls.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.llm_models import GEMINI_LIVE_MODELS
from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler
from app.voice.voice_orchestrator import (
    VoiceOrchestrator,
    _GEMINI_LIVE_FALLBACK_TEXT_MODEL,
)


NATIVE_AUDIO_MODEL = next(iter(GEMINI_LIVE_MODELS))


class _FalsyCallSession:
    """Falsy call_session so build_system_prompt's DB-dependent branches
    (CRM/KB/caller-memory) are skipped — mirrors the pattern already used in
    tests/voice/test_bracket_tag_prompt_consistency.py."""

    def __bool__(self) -> bool:
        return False


def _fake_handler(*, llm_model: str | None) -> Handler:
    """Minimal Twilio handler via object.__new__ — enough attributes for
    VoiceOrchestrator.__init__ + on_audio_chunk's pickup-detection gate +
    the Gemini Live fork, without touching a real DB/WebSocket."""
    h = object.__new__(Handler)

    h.agent = MagicMock()
    h.agent.id = uuid.uuid4()
    h.agent.name = "TestAgent"
    h.agent.llm_model = llm_model
    h.agent.system_prompt = None
    h.agent.model = MagicMock()
    h.agent.model.system_prompt = None
    h.agent.transfer_route = None

    h.call_session = _FalsyCallSession()
    h.call_flow = None
    h.db = None
    h.call_session_id = "test-session-id"
    h.agent_id = str(h.agent.id)
    h.call_sid = "CA_test"
    h.stream_sid = "MZ_test"

    # Audio thresholds read by VoiceOrchestrator.__init__
    h._min_audio_level_threshold = 30
    h._audio_samples_needed = 1
    h._audio_non_silent_needed = 1

    h._enable_interim_llm = False
    h._min_interim_words = 3
    h._min_interim_confidence = 0.4
    h._min_interim_interval_sec = 0.2
    h._barge_in_min_conf = 0.26
    h._barge_in_min_conf_1w = 0.52

    h._voice_metrics = MagicMock()
    h._add_to_transcript = AsyncMock()
    h._send_twilio_clear_event = AsyncMock()
    h._stream_live_audio_chunk = AsyncMock()
    h._full_shutdown = AsyncMock()
    h.websocket = MagicMock()

    return h


def _loud_frame() -> bytes:
    """A single MULAW byte pattern whose linear RMS clears the pickup
    threshold immediately (threshold=30, samples_needed=1 above)."""
    return bytes([0x00]) * 160  # MULAW 0x00 decodes to a large-magnitude sample


class TestNativeAudioRoutingResolution:
    def test_native_audio_model_sets_is_gemini_live_true(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        assert orch._is_gemini_live is True

    def test_non_native_model_sets_is_gemini_live_false(self):
        h = _fake_handler(llm_model="gpt-4o-mini")
        orch = VoiceOrchestrator(h)
        assert orch._is_gemini_live is False

    def test_no_agent_defaults_to_false(self):
        h = _fake_handler(llm_model="gpt-4o-mini")
        h.agent = None
        orch = VoiceOrchestrator(h)
        assert orch._is_gemini_live is False


class TestNativeAudioNeverConstructsSttPipeline:
    def test_native_audio_agent_skips_stt_pipeline_construction(self, monkeypatch):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        orch._user_picked_up = True  # skip pickup-detection gate for this test
        h._handle_user_pickup = AsyncMock()

        fake_session = MagicMock()
        fake_session.start = AsyncMock()
        fake_session.send_audio = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch.on_audio_chunk(_loud_frame()))

        assert orch._stt_pipeline is None
        fake_session.start.assert_awaited_once()
        fake_session.send_audio.assert_awaited_once()

    def test_non_native_agent_still_constructs_stt_pipeline(self, monkeypatch):
        h = _fake_handler(llm_model="gpt-4o-mini")
        orch = VoiceOrchestrator(h)
        orch._user_picked_up = True
        h._handle_user_pickup = AsyncMock()

        fake_stt = MagicMock()
        fake_stt.feed_audio_chunk = AsyncMock()
        with patch("app.voice.voice_orchestrator.SttPipeline", return_value=fake_stt):
            asyncio.run(orch.on_audio_chunk(_loud_frame()))

        assert orch._stt_pipeline is fake_stt
        assert orch._gemini_live_session is None
        fake_stt.feed_audio_chunk.assert_awaited_once()


class TestAudioConversionRouting:
    def test_send_audio_receives_pcm16_16k_conversion(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.send_audio = AsyncMock()
        orch._gemini_live_session = fake_session  # session already "started"

        mulaw = _loud_frame()
        with patch(
            "app.voice.voice_orchestrator.mulaw8k_to_pcm16_16k",
            return_value=b"\x01\x02converted",
        ) as conv:
            asyncio.run(orch._feed_gemini_live_audio(h, mulaw))

        conv.assert_called_once_with(mulaw)
        fake_session.send_audio.assert_awaited_once_with(b"\x01\x02converted")

    def test_on_audio_chunk_callback_converts_and_streams_mulaw(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        with patch(
            "app.voice.voice_orchestrator.pcm16_24k_to_mulaw8k",
            return_value=b"mulaw-out",
        ) as conv:
            asyncio.run(orch._on_gemini_live_audio_chunk(b"pcm24k-in"))

        conv.assert_called_once_with(b"pcm24k-in")
        h._stream_live_audio_chunk.assert_awaited_once_with(b"mulaw-out")

    def test_on_audio_chunk_callback_gated_by_cancel_flag(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        orch._gemini_live_cancel.set()

        asyncio.run(orch._on_gemini_live_audio_chunk(b"pcm24k-in"))

        h._stream_live_audio_chunk.assert_not_awaited()


class TestBargeIn:
    def test_interrupted_sets_cancel_and_clears_twilio_buffer(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        orch._gemini_live_first_audio_marked = True

        asyncio.run(orch._on_gemini_live_interrupted())

        h._send_twilio_clear_event.assert_awaited_once()
        # Reset for the next turn, mirroring _tts_cancel.clear() convention.
        assert not orch._gemini_live_cancel.is_set()
        assert orch._gemini_live_first_audio_marked is False


class TestTranscriptCallbacks:
    def test_input_transcript_written_only_when_finished(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_gemini_live_input_transcript("partial", False))
        h._add_to_transcript.assert_not_awaited()

        asyncio.run(orch._on_gemini_live_input_transcript("hello there", True))
        h._add_to_transcript.assert_awaited_once_with("client", "hello there", "speech")

    def test_output_transcript_written_only_when_finished(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_gemini_live_output_transcript("partial reply", False))
        h._add_to_transcript.assert_not_awaited()

        asyncio.run(orch._on_gemini_live_output_transcript("Sure, I can help.", True))
        h._add_to_transcript.assert_awaited_once_with(
            "agent", "Sure, I can help.", "agent_response"
        )

    def test_blank_finished_transcript_not_written(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_gemini_live_input_transcript("   ", True))
        h._add_to_transcript.assert_not_awaited()


class TestPreSessionFallback:
    def test_start_failure_falls_back_to_legacy_text_model(self):
        from app.services.vertex_gemini_service import VertexLlmError, VertexLlmErrorType

        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock(
            side_effect=VertexLlmError("boom", VertexLlmErrorType.QUOTA)
        )

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        assert orch._gemini_live_setup_failed is True
        assert orch._is_gemini_live is False
        assert orch._gemini_live_session is None
        assert h.agent.llm_model == _GEMINI_LIVE_FALLBACK_TEXT_MODEL
        h._full_shutdown.assert_not_awaited()

    def test_fallback_with_no_agent_ends_call_gracefully(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h.agent = None

        asyncio.run(orch._fallback_to_legacy_pipeline(h, reason="no agent"))

        assert orch._gemini_live_setup_failed is True
        h._full_shutdown.assert_called_once()


class TestMidCallError:
    def test_mid_call_error_ends_call_gracefully(self):
        from app.services.vertex_gemini_service import VertexLlmError, VertexLlmErrorType

        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(
            orch._on_gemini_live_error(VertexLlmError("dropped", VertexLlmErrorType.UNKNOWN))
        )

        h._full_shutdown.assert_called_once()


def _fake_handler_for_real_prompt_build(*, llm_model: str) -> Handler:
    """
    A Twilio handler with enough real attributes for
    `BidirectionalStreamHandler.build_system_prompt` (the REAL implementation,
    not mocked) to run to completion without a DB/WebSocket. Mirrors the
    attribute set `ConversationOrchestrator.build_system_prompt`'s own test
    fixture (`tests/voice/test_bracket_tag_prompt_consistency.py::
    _fake_livekit_handler`) uses for the browser path, adapted to the
    Twilio-only locals this method also touches (booking-memory tri-state,
    JD/recruitment screening check, RAG prefetch slot).
    """
    h = _fake_handler(llm_model=llm_model)
    h.agent.language = "en"
    h.agent.model.api_key = None
    h.agent.is_inbound_agent = False
    h.agent.tts_provider = None
    h.call_session = None  # no DB-backed session -> CRM/KB/history branches skipped
    h.db = None
    h._last_offered_calendar_slots = []
    h._last_requested_calendar_date = None
    h._last_selected_calendar_slot = None
    h._conversation_history_cache = []
    h.HISTORY_MAX_MESSAGES = 20
    h._metric_stt_final_ts = 0.0
    h._rag_prefetch_task = None
    h._kb_cache_ready = False
    h._cached_inbound_kb_block = ""
    h._jd_recruitment_screening_active = lambda: False
    h._send_quick_acknowledgement = AsyncMock()
    return h


class TestGeminiLiveSessionUsesTwilioOwnPromptAssembly:
    """
    Regression guard for the bug this task fixes: `_start_gemini_live_session`
    must build the Gemini Live `system_instruction` from
    `BidirectionalStreamHandler.build_system_prompt` (Twilio's own, real
    prompt-assembly logic — the one `generate_and_stream_response` actually
    uses), never from `ConversationOrchestrator.build_system_prompt` (the
    separate, LiveKit-browser-only implementation, which does not build a
    "# CURRENT DATE & TIME" header, a booking-memory block, or any of the
    other Twilio-specific blocks asserted below).
    """

    def test_build_system_prompt_contains_twilio_only_date_time_header(self):
        h = _fake_handler_for_real_prompt_build(llm_model="gpt-4o-mini")

        prompt = asyncio.run(h.build_system_prompt(user_text="", confidence=1.0))

        # ConversationOrchestrator.build_system_prompt (browser) never emits
        # this header at all — see app/voice/conversation_orchestrator.py.
        assert "# CURRENT DATE & TIME" in prompt
        assert "Now:" in prompt

    def test_start_gemini_live_session_uses_handlers_own_build_system_prompt(self):
        """
        End-to-end through the real fork point: only GeminiLiveSession is
        mocked, so the system_instruction passed to `session.start(...)` is
        whatever `BidirectionalStreamHandler.build_system_prompt` (the real,
        unmocked implementation) produces.
        """
        h = _fake_handler_for_real_prompt_build(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        fake_session.start.assert_awaited_once()
        system_instruction = fake_session.start.call_args.kwargs["system_instruction"]
        assert "# CURRENT DATE & TIME" in system_instruction

    def test_browser_handler_still_uses_conversation_orchestrator(self):
        """
        The browser branch (LiveKitBrowserCallHandler — has no
        `build_system_prompt` method of its own) must remain unchanged: it
        still routes through `ConversationOrchestrator.build_system_prompt`.
        """
        h = MagicMock(spec=[])  # no build_system_prompt attribute -> duck-type fails
        h.agent = MagicMock()
        h.agent.llm_model = NATIVE_AUDIO_MODEL
        h.call_session_id = "test-session-id"

        assert not hasattr(h, "build_system_prompt")

        orch = VoiceOrchestrator.__new__(VoiceOrchestrator)
        orch._gemini_live_session = None

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


class TestGreetingBypassesExternalTtsForNativeAudio:
    """
    Regression guard for a real bug caught in code review: the auto-greeting
    path (BidirectionalStreamHandler.generate_and_stream_response's
    is_greeting=True branch) originally queued the scripted greeting through
    TtsPipeline/the external TTS provider unconditionally, for EVERY agent —
    including native-audio ones, which must never invoke external TTS. Ran
    up real vendor cost and (on the browser transport) produced pitch-shifted
    audio. Neither of this file's other test classes (which only exercise
    on_audio_chunk/_start_gemini_live_session directly) would have caught
    this, since the bug was in a separate call path (_handle_user_pickup ->
    generate_and_stream_response(is_greeting=True)) that never goes through
    on_audio_chunk at all.
    """

    def _greeting_handler(self, *, is_gemini_live: bool) -> Handler:
        h = object.__new__(Handler)
        h.agent = MagicMock()
        h.agent.greeting_message = "Hi, thanks for calling Acme."
        h.agent.first_message = None
        h.call_session = MagicMock()
        h.call_session.call_type = "inbound"
        h.call_session.call_metadata = {}
        h.call_session.id = uuid.uuid4()
        h.db = None
        h._flow_executor = None
        h._add_to_transcript = AsyncMock()
        h._twilio_buffer_primed = True
        h._use_ssml = True
        h._jd_recruitment_screening_active = MagicMock(return_value=False)

        h._tts_pipeline = MagicMock()
        h._tts_pipeline.queue_tts = AsyncMock()

        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        vo = MagicMock()
        vo._is_gemini_live = is_gemini_live
        vo._gemini_live_session = fake_session if is_gemini_live else None
        h._voice_orchestrator = vo
        h._fake_session = fake_session  # test-only handle
        return h

    def test_native_audio_greeting_uses_live_session_send_text_not_tts(self):
        h = self._greeting_handler(is_gemini_live=True)

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._fake_session.send_text.assert_awaited_once_with(
            "Hi, thanks for calling Acme.", turn_complete=True
        )
        h._tts_pipeline.queue_tts.assert_not_awaited()

    def test_native_audio_greeting_skips_gracefully_if_session_not_ready(self):
        h = self._greeting_handler(is_gemini_live=True)
        h._voice_orchestrator._gemini_live_session = None

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._tts_pipeline.queue_tts.assert_not_awaited()

    def test_non_native_audio_greeting_still_uses_external_tts(self):
        h = self._greeting_handler(is_gemini_live=False)

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._tts_pipeline.queue_tts.assert_awaited_once()
        queued = h._tts_pipeline.queue_tts.call_args[0][0]
        assert queued["text"] == "Hi, thanks for calling Acme."

    def test_missing_voice_orchestrator_attribute_falls_back_to_external_tts(self):
        """A handler built without a real __init__ (e.g. other lightweight
        test doubles in this test suite) has no _voice_orchestrator at all —
        must not crash, must behave like a non-native-audio agent."""
        h = self._greeting_handler(is_gemini_live=False)
        del h._voice_orchestrator

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._tts_pipeline.queue_tts.assert_awaited_once()


def _fc(*, id: str, name: str, args: dict) -> MagicMock:
    """Fake google.genai.types.FunctionCall — MagicMock(name=...) sets the
    mock's own repr name, not a gettable `.name` attribute, so it must be
    assigned after construction."""
    fc = MagicMock(id=id, args=args)
    fc.name = name
    return fc


class TestCalendlyToolCallWiring:
    """
    Phase 2: Calendly tool-calling wired into the Live session's async
    on_tool_call/send_tool_response exchange, reusing
    BookingMixin._execute_calendly_tool_call (Twilio-only — the browser
    handler has no BookingMixin at all, so this is naturally never wired on
    that transport; see the browser fork test file for the mirror-image
    "tools stay None" assertion).
    """

    def _handler_with_calendly(self, *, enabled: bool) -> Handler:
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h._calendly_enabled = MagicMock(return_value=enabled)
        h._execute_calendly_tool_call = AsyncMock(return_value={"slots": []})
        return h

    def test_session_start_passes_calendly_tool_when_enabled(self):
        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["tools"] is not None
        assert len(kwargs["tools"]) == 1
        assert kwargs["on_tool_call"] == orch._on_gemini_live_tool_call

    def test_session_start_passes_no_tools_when_calendly_disabled(self):
        h = self._handler_with_calendly(enabled=False)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["tools"] is None
        assert kwargs["on_tool_call"] is None

    # Note: a handler genuinely lacking `_calendly_enabled` altogether (the
    # real browser-transport shape, since LiveKitBrowserCallHandler has no
    # BookingMixin) can't be constructed from `Handler`/BidirectionalStream-
    # Handler fakes here — BookingMixin's `_calendly_enabled` is a class-level
    # method present on every instance regardless of state. That exact case
    # (hasattr is False) is covered instead in
    # tests/voice/test_gemini_live_browser_fork.py against a real
    # LiveKitBrowserCallHandler instance.

    def test_on_tool_call_resolves_each_call_and_sends_responses(self):
        # google.genai is stubbed as a plain MagicMock in tests/conftest.py —
        # types.FunctionResponse(...) doesn't behave like a real dataclass
        # (calling it returns the same fixed .return_value regardless of
        # kwargs), so assert on the *constructor* call args instead of
        # attributes on the returned object.
        from google.genai import types as genai_types

        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_tool_response = AsyncMock()
        orch._gemini_live_session = fake_session

        fc1 = _fc(id="call-1", name="check_availability", args={"date": "tomorrow"})
        fc2 = _fc(
            id="call-2",
            name="book_appointment",
            args={"slot": "x", "attendee_email": "a@b.com"},
        )
        h._execute_calendly_tool_call = AsyncMock(
            side_effect=[{"slots": ["09:00"]}, {"booked": True}]
        )
        genai_types.FunctionResponse.reset_mock()

        asyncio.run(orch._on_gemini_live_tool_call([fc1, fc2]))

        assert h._execute_calendly_tool_call.await_count == 2
        h._execute_calendly_tool_call.assert_any_await("check_availability", {"date": "tomorrow"})
        h._execute_calendly_tool_call.assert_any_await(
            "book_appointment", {"slot": "x", "attendee_email": "a@b.com"}
        )
        fake_session.send_tool_response.assert_awaited_once()
        sent = fake_session.send_tool_response.call_args.kwargs["function_responses"]
        assert len(sent) == 2

        constructor_calls = genai_types.FunctionResponse.call_args_list
        assert constructor_calls[0].kwargs == {
            "id": "call-1", "name": "check_availability", "response": {"slots": ["09:00"]},
        }
        assert constructor_calls[1].kwargs == {
            "id": "call-2", "name": "book_appointment", "response": {"booked": True},
        }

    def test_on_tool_call_noop_when_session_not_set(self):
        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        orch._gemini_live_session = None

        # Must not raise even though there's nowhere to send a response.
        asyncio.run(orch._on_gemini_live_tool_call(
            [_fc(id="c1", name="check_availability", args={})]
        ))
        h._execute_calendly_tool_call.assert_not_awaited()

    def test_on_tool_call_survives_execute_exception(self):
        from google.genai import types as genai_types

        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_tool_response = AsyncMock()
        orch._gemini_live_session = fake_session
        h._execute_calendly_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        genai_types.FunctionResponse.reset_mock()

        asyncio.run(orch._on_gemini_live_tool_call(
            [_fc(id="c1", name="check_availability", args={})]
        ))

        fake_session.send_tool_response.assert_awaited_once()
        constructor_calls = genai_types.FunctionResponse.call_args_list
        assert constructor_calls[0].kwargs["response"] == {
            "error": "internal error executing tool call"
        }


class TestMidCallRagRefresh:
    """
    Phase 2: mid-call KB context refresh (VoiceOrchestrator.
    _refresh_gemini_live_kb_context), triggered fire-and-forget from a
    finished input transcript since there's no per-turn STT-interim signal
    in a live session to key a prefetch off.
    """

    def _handler_with_flow(self, *, kb_ids):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h.call_flow = MagicMock()
        h.call_flow.knowledge_base_ids = kb_ids
        return h

    def test_refresh_noop_when_no_session(self):
        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        orch._gemini_live_session = None

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("some context", 5.0)),
        ) as mock_retrieve:
            asyncio.run(orch._refresh_gemini_live_kb_context("what's your refund policy"))

        mock_retrieve.assert_not_awaited()

    def test_refresh_noop_when_no_kb_ids(self):
        h = self._handler_with_flow(kb_ids=[])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._gemini_live_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("some context", 5.0)),
        ) as mock_retrieve:
            asyncio.run(orch._refresh_gemini_live_kb_context("what's your refund policy"))

        mock_retrieve.assert_not_awaited()
        fake_session.send_text.assert_not_awaited()

    def test_refresh_sends_context_as_non_blocking_text_turn(self):
        kb_id = uuid.uuid4()
        h = self._handler_with_flow(kb_ids=[kb_id])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._gemini_live_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("Refunds within 30 days.", 42.0)),
        ) as mock_retrieve, patch("app.utils.redis_client.get_redis", return_value=None):
            asyncio.run(orch._refresh_gemini_live_kb_context("what's your refund policy"))

        mock_retrieve.assert_awaited_once()
        call_kwargs = mock_retrieve.call_args.kwargs
        assert call_kwargs["transcript"] == "what's your refund policy"
        assert call_kwargs["kb_ids"] == [kb_id]

        fake_session.send_text.assert_awaited_once()
        sent_text, sent_kwargs = fake_session.send_text.call_args.args, fake_session.send_text.call_args.kwargs
        assert "Refunds within 30 days." in sent_text[0]
        assert sent_kwargs["turn_complete"] is False

    def test_refresh_sends_nothing_when_kb_returns_empty(self):
        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._gemini_live_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("", 3.0)),
        ):
            asyncio.run(orch._refresh_gemini_live_kb_context("unrelated small talk"))

        fake_session.send_text.assert_not_awaited()

    def test_refresh_fails_open_on_retrieval_exception(self):
        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._gemini_live_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(side_effect=RuntimeError("kb backend down")),
        ):
            # Must not raise.
            asyncio.run(orch._refresh_gemini_live_kb_context("what's your refund policy"))

        fake_session.send_text.assert_not_awaited()

    def test_input_transcript_callback_schedules_refresh_task(self):
        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)

        async def _run():
            with patch.object(
                orch, "_refresh_gemini_live_kb_context", new=AsyncMock()
            ) as mock_refresh:
                await orch._on_gemini_live_input_transcript("hello there", True)
                # The refresh runs as a tracked background task, not inline —
                # await the same task set _full_shutdown drains, within this
                # same event loop (a task is bound to the loop it was created
                # on — a second, separate asyncio.run() call cannot resume it).
                pending = list(orch._pending_final_tasks)
                if pending:
                    await asyncio.gather(*pending)
                mock_refresh.assert_called_once_with("hello there")

        asyncio.run(_run())


class TestPerAgentVoiceWiring:
    """Phase 2: agent.tts_voice_external_id / agent.tts_language are reused
    (no new schema) to pick the Gemini Live session's native voice."""

    def test_valid_agent_voice_passed_through(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h.agent.tts_voice_external_id = "charon"
        h.agent.tts_language = "en-AU"
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["voice_name"] == "Charon"
        assert kwargs["language_code"] == "en-AU"

    def test_unset_or_invalid_agent_voice_falls_back_to_default(self):
        from app.services.gemini_live_service import DEFAULT_VOICE_NAME

        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h.agent.tts_voice_external_id = "21m00Tcm4TlvDq8ikWAM"  # leftover ElevenLabs ID
        h.agent.tts_language = None
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.gemini_live_service.GeminiLiveSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_gemini_live_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["voice_name"] == DEFAULT_VOICE_NAME
        assert kwargs["language_code"] is None


class TestMidCallRagRefreshCoalescing:
    def test_second_trigger_skipped_while_refresh_in_flight(self):
        h = TestMidCallRagRefresh()._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        orch._gemini_live_kb_refresh_in_flight = True

        with patch.object(
            orch, "_refresh_gemini_live_kb_context", new=AsyncMock()
        ) as mock_refresh:
            asyncio.run(orch._on_gemini_live_input_transcript("are you still there", True))

        mock_refresh.assert_not_called()

    def test_flag_cleared_after_refresh_completes(self):
        kb_id = uuid.uuid4()
        h = TestMidCallRagRefresh()._handler_with_flow(kb_ids=[kb_id])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._gemini_live_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("some context", 5.0)),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            asyncio.run(orch._refresh_gemini_live_kb_context("hello"))

        assert orch._gemini_live_kb_refresh_in_flight is False

    def test_flag_cleared_even_on_exception(self):
        h = TestMidCallRagRefresh()._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        orch._gemini_live_session = MagicMock()

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            asyncio.run(orch._refresh_gemini_live_kb_context("hello"))

        assert orch._gemini_live_kb_refresh_in_flight is False


class TestToolCallLoggingRedactsPii:
    def test_attendee_email_redacted_in_log(self, caplog):
        import logging

        h = TestCalendlyToolCallWiring()._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_tool_response = AsyncMock()
        orch._gemini_live_session = fake_session
        h._execute_calendly_tool_call = AsyncMock(return_value={"booked": True})

        fc = _fc(
            id="c1",
            name="book_appointment",
            args={"slot": "x", "attendee_email": "realuser@example.com"},
        )

        with caplog.at_level(logging.INFO):
            asyncio.run(orch._on_gemini_live_tool_call([fc]))

        logged = "\n".join(r.message for r in caplog.records)
        assert "realuser@example.com" not in logged
