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
