"""
Twilio-transport fork-point tests for OpenAI Realtime native-audio routing
(VoiceOrchestrator.on_audio_chunk -> _feed_openai_realtime_audio /
_start_openai_realtime_session / _on_openai_realtime_* callbacks).

Mirrors tests/voice/test_gemini_live_twilio_fork.py's structure, adapted for
this provider's genuine divergences (see
app/services/openai_realtime_service.py's module docstring):
  - no MULAW->PCM16 conversion on the Twilio path at all (OpenAI accepts
    audio/pcmu directly) — send_audio receives the RAW mulaw frame
  - transcripts are written directly, no fragment-buffering
  - barge-in is a single flag-set/clear, no dual-buffer flush

OpenAIRealtimeSession itself is mocked throughout — no live API calls.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.llm_models import OPENAI_REALTIME_MODELS
from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler
from app.voice.voice_orchestrator import (
    VoiceOrchestrator,
    _OPENAI_REALTIME_FALLBACK_TEXT_MODEL,
)

NATIVE_AUDIO_MODEL = next(iter(OPENAI_REALTIME_MODELS))


class _FalsyCallSession:
    def __bool__(self) -> bool:
        return False


def _fake_handler(*, llm_model: str | None) -> Handler:
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
    return bytes([0x00]) * 160


class TestNativeAudioRoutingResolution:
    def test_native_audio_model_sets_is_openai_realtime_true(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        assert orch._is_openai_realtime is True
        assert orch._is_gemini_live is False

    def test_non_native_model_sets_is_openai_realtime_false(self):
        h = _fake_handler(llm_model="gpt-4o-mini")
        orch = VoiceOrchestrator(h)
        assert orch._is_openai_realtime is False

    def test_no_agent_defaults_to_false(self):
        h = _fake_handler(llm_model="gpt-4o-mini")
        h.agent = None
        orch = VoiceOrchestrator(h)
        assert orch._is_openai_realtime is False


class TestNativeAudioNeverConstructsSttPipeline:
    def test_native_audio_agent_skips_stt_pipeline_construction(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        orch._user_picked_up = True
        h._handle_user_pickup = AsyncMock()

        fake_session = MagicMock()
        fake_session.start = AsyncMock()
        fake_session.send_audio = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch.on_audio_chunk(_loud_frame()))

        assert orch._stt_pipeline is None
        fake_session.start.assert_awaited_once()
        fake_session.send_audio.assert_awaited_once()

    def test_non_native_agent_still_constructs_stt_pipeline(self):
        h = _fake_handler(llm_model="gpt-4o-mini")
        orch = VoiceOrchestrator(h)
        orch._user_picked_up = True
        h._handle_user_pickup = AsyncMock()

        fake_stt = MagicMock()
        fake_stt.feed_audio_chunk = AsyncMock()
        with patch("app.voice.voice_orchestrator.SttPipeline", return_value=fake_stt):
            asyncio.run(orch.on_audio_chunk(_loud_frame()))

        assert orch._stt_pipeline is fake_stt
        assert orch._openai_realtime_session is None
        fake_stt.feed_audio_chunk.assert_awaited_once()


class TestAudioRoutingNoConversion:
    """Regression guard: unlike Gemini Live, the Twilio path must NEVER
    resample/convert audio for OpenAI Realtime — mu-law/8kHz goes straight
    through in both directions (audio/pcmu is natively supported)."""

    def test_send_audio_receives_raw_mulaw_unconverted(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.send_audio = AsyncMock()
        orch._openai_realtime_session = fake_session

        mulaw = _loud_frame()
        asyncio.run(orch._feed_openai_realtime_audio(h, mulaw))

        fake_session.send_audio.assert_awaited_once_with(mulaw)

    def test_on_audio_chunk_callback_streams_raw_bytes_unconverted(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_openai_realtime_audio_chunk(b"mulaw-from-openai"))

        h._stream_live_audio_chunk.assert_awaited_once_with(b"mulaw-from-openai")

    def test_on_audio_chunk_callback_gated_by_cancel_flag(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        orch._openai_realtime_cancel.set()

        asyncio.run(orch._on_openai_realtime_audio_chunk(b"mulaw-from-openai"))

        h._stream_live_audio_chunk.assert_not_awaited()


class TestBargeIn:
    def test_interrupted_sets_and_clears_cancel_and_clears_twilio_buffer(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        orch._openai_realtime_first_audio_marked = True

        asyncio.run(orch._on_openai_realtime_interrupted())

        h._send_twilio_clear_event.assert_awaited_once()
        assert not orch._openai_realtime_cancel.is_set()
        assert orch._openai_realtime_first_audio_marked is False


class TestTranscriptCallbacksWriteDirectly:
    """No fragment-buffering — each transcript callback fires once per
    utterance with the full text and is written immediately."""

    def test_input_transcript_written_immediately(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_openai_realtime_input_transcript("hello there"))

        h._add_to_transcript.assert_awaited_once_with("client", "hello there", "speech")

    def test_output_transcript_written_immediately(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_openai_realtime_output_transcript("Sure, I can help."))

        h._add_to_transcript.assert_awaited_once_with("agent", "Sure, I can help.", "agent_response")

    def test_blank_transcript_not_written(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(orch._on_openai_realtime_input_transcript("   "))
        asyncio.run(orch._on_openai_realtime_output_transcript(""))

        h._add_to_transcript.assert_not_awaited()

    def test_input_transcript_schedules_kb_refresh_task(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        async def _run():
            with patch.object(
                orch, "_refresh_openai_realtime_kb_context", new=AsyncMock()
            ) as mock_refresh:
                await orch._on_openai_realtime_input_transcript("what is your refund policy")
                pending = list(orch._pending_final_tasks)
                if pending:
                    await asyncio.gather(*pending)
                mock_refresh.assert_called_once_with("what is your refund policy")

        asyncio.run(_run())


class TestMidCallRagRefresh:
    """
    Mid-call KB context refresh (VoiceOrchestrator.
    _refresh_openai_realtime_kb_context), triggered fire-and-forget from
    _on_openai_realtime_input_transcript. Mirrors
    test_gemini_live_twilio_fork.py's TestMidCallRagRefresh, adapted for
    this provider's send_text(text, respond=False) signature (vs Gemini's
    send_text(text, turn_complete=False)) and its DIAGNOSTIC skip-reason
    logging / upgraded warning-with-exc_info on retrieval failure.
    """

    def _handler_with_flow(self, *, kb_ids):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h.call_flow = MagicMock()
        h.call_flow.knowledge_base_ids = kb_ids
        return h

    def test_refresh_noop_when_no_session(self, caplog):
        import logging

        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        orch._openai_realtime_session = None

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("some context", 5.0)),
        ) as mock_retrieve, caplog.at_level(logging.INFO):
            asyncio.run(orch._refresh_openai_realtime_kb_context("what's your refund policy"))

        mock_retrieve.assert_not_awaited()
        assert any("no_session" in rec.message for rec in caplog.records)

    def test_refresh_noop_when_empty_transcript(self, caplog):
        import logging

        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._openai_realtime_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("some context", 5.0)),
        ) as mock_retrieve, caplog.at_level(logging.INFO):
            asyncio.run(orch._refresh_openai_realtime_kb_context(""))

        mock_retrieve.assert_not_awaited()
        fake_session.send_text.assert_not_awaited()
        assert any("empty_transcript" in rec.message for rec in caplog.records)

    def test_refresh_noop_when_no_kb_ids(self, caplog):
        import logging

        h = self._handler_with_flow(kb_ids=[])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._openai_realtime_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("some context", 5.0)),
        ) as mock_retrieve, caplog.at_level(logging.INFO):
            asyncio.run(orch._refresh_openai_realtime_kb_context("what's your refund policy"))

        mock_retrieve.assert_not_awaited()
        fake_session.send_text.assert_not_awaited()
        assert any("no_kb_ids_configured" in rec.message for rec in caplog.records)

    def test_refresh_sends_context_as_non_blocking_text_turn(self):
        kb_id = uuid.uuid4()
        h = self._handler_with_flow(kb_ids=[kb_id])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._openai_realtime_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("Refunds within 30 days.", 42.0)),
        ) as mock_retrieve, patch("app.utils.redis_client.get_redis", return_value=None):
            asyncio.run(orch._refresh_openai_realtime_kb_context("what's your refund policy"))

        mock_retrieve.assert_awaited_once()
        call_kwargs = mock_retrieve.call_args.kwargs
        assert call_kwargs["transcript"] == "what's your refund policy"
        assert call_kwargs["kb_ids"] == [kb_id]

        fake_session.send_text.assert_awaited_once()
        sent_args, sent_kwargs = fake_session.send_text.call_args.args, fake_session.send_text.call_args.kwargs
        assert "Refunds within 30 days." in sent_args[0]
        assert sent_kwargs["respond"] is False

    def test_refresh_sends_nothing_when_kb_returns_empty(self):
        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._openai_realtime_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("", 3.0)),
        ):
            asyncio.run(orch._refresh_openai_realtime_kb_context("unrelated small talk"))

        fake_session.send_text.assert_not_awaited()

    def test_refresh_fails_open_and_logs_warning_with_exc_info_on_retrieval_exception(self, caplog):
        """Regression guard for the debug->warning upgrade: a silently
        debug-logged exception here would look identical in production logs
        to 'KB refresh never fired at all', so this must be logged at
        WARNING with exc_info=True, not swallowed at DEBUG."""
        h = self._handler_with_flow(kb_ids=[uuid.uuid4()])
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_text = AsyncMock()
        orch._openai_realtime_session = fake_session

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(side_effect=RuntimeError("kb backend down")),
        ), patch("app.voice.voice_orchestrator.logger") as mock_logger:
            # Must not raise.
            asyncio.run(orch._refresh_openai_realtime_kb_context("what's your refund policy"))

        fake_session.send_text.assert_not_awaited()
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.kwargs.get("exc_info") is True
        mock_logger.debug.assert_not_called()


class TestPreSessionFallback:
    def test_start_failure_falls_back_to_openai_text_model(self):
        from app.services.openai_realtime_service import OpenAIRealtimeError, OpenAIRealtimeErrorType

        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock(
            side_effect=OpenAIRealtimeError("boom", OpenAIRealtimeErrorType.QUOTA)
        )

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        assert orch._openai_realtime_setup_failed is True
        assert orch._is_openai_realtime is False
        assert orch._openai_realtime_session is None
        assert h.agent.llm_model == _OPENAI_REALTIME_FALLBACK_TEXT_MODEL
        h._full_shutdown.assert_not_awaited()

    def test_fallback_with_no_agent_ends_call_gracefully(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)
        h.agent = None

        asyncio.run(orch._fallback_to_legacy_pipeline_openai(h, reason="no agent"))

        assert orch._openai_realtime_setup_failed is True
        h._full_shutdown.assert_called_once()

    def test_fallback_persists_flag_on_call_metadata_for_credit_billing(self):
        """Fix #4: the fallback must persist a flag on
        CallSession.call_metadata so credit_service's out-of-process billing
        tick (which re-queries CallSession every 10s and has no reference to
        this orchestrator instance) can stop billing the realtime surcharge
        and start evaluating the ElevenLabs surcharge for the rest of the
        call."""
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        call_session = MagicMock()
        call_session.call_metadata = {"existing_key": "kept"}
        h.call_session = call_session
        h.db = MagicMock()

        asyncio.run(orch._fallback_to_legacy_pipeline_openai(h, reason="pre-session boom"))

        assert orch._is_openai_realtime is False
        assert call_session.call_metadata["existing_key"] == "kept"
        fallback_meta = call_session.call_metadata["openai_realtime_fallback"]
        assert fallback_meta["fell_back"] is True
        assert fallback_meta["reason"] == "pre-session boom"
        assert "at" in fallback_meta
        h.db.commit.assert_called_once()

    def test_fallback_write_failure_does_not_raise(self):
        """A DB error while persisting the fallback flag must not blow up the
        call — it's a best-effort billing-accuracy signal, not a correctness
        requirement for the call itself."""
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        call_session = MagicMock()
        call_session.call_metadata = {}
        h.call_session = call_session
        h.db = MagicMock()
        h.db.commit.side_effect = RuntimeError("db down")

        # Must not raise.
        asyncio.run(orch._fallback_to_legacy_pipeline_openai(h, reason="boom"))

        assert orch._is_openai_realtime is False
        h._full_shutdown.assert_not_awaited()


class TestMidCallError:
    def test_mid_call_error_ends_call_gracefully(self):
        from app.services.openai_realtime_service import OpenAIRealtimeError, OpenAIRealtimeErrorType

        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        asyncio.run(
            orch._on_openai_realtime_error(
                OpenAIRealtimeError("dropped", OpenAIRealtimeErrorType.CONNECTION_CLOSED)
            )
        )

        h._full_shutdown.assert_called_once()


def _fake_handler_for_real_prompt_build(*, llm_model: str) -> Handler:
    h = _fake_handler(llm_model=llm_model)
    h.agent.language = "en"
    h.agent.model.api_key = None
    h.agent.is_inbound_agent = False
    h.agent.tts_provider = None
    h.call_session = None
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


class TestOpenAIRealtimeSessionUsesTwilioOwnPromptAssembly:
    def test_start_openai_realtime_session_uses_handlers_own_build_system_prompt(self):
        h = _fake_handler_for_real_prompt_build(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        fake_session.start.assert_awaited_once()
        system_instruction = fake_session.start.call_args.kwargs["system_instruction"]
        assert "# CURRENT DATE & TIME" in system_instruction

    def test_start_uses_twilio_audio_pcmu_format(self):
        from app.services.openai_realtime_service import TWILIO_AUDIO_FORMAT

        h = _fake_handler_for_real_prompt_build(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        assert fake_session.start.call_args.kwargs["audio_format"] == TWILIO_AUDIO_FORMAT

    def test_browser_handler_still_uses_conversation_orchestrator(self):
        from app.services.openai_realtime_service import LIVEKIT_AUDIO_FORMAT

        h = MagicMock(spec=[])
        h.agent = MagicMock()
        h.agent.llm_model = NATIVE_AUDIO_MODEL
        h.call_session_id = "test-session-id"

        assert not hasattr(h, "build_system_prompt")

        orch = VoiceOrchestrator.__new__(VoiceOrchestrator)
        orch._openai_realtime_session = None

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
        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["system_instruction"] == "browser system prompt"
        assert kwargs["audio_format"] == LIVEKIT_AUDIO_FORMAT


class TestGreetingBypassesExternalTtsForNativeAudio:
    def _greeting_handler(self, *, is_openai_realtime: bool) -> Handler:
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
        vo._is_gemini_live = False
        vo._gemini_live_session = None
        vo._is_openai_realtime = is_openai_realtime
        vo._openai_realtime_session = fake_session if is_openai_realtime else None
        h._voice_orchestrator = vo
        h._fake_session = fake_session
        return h

    def test_native_audio_greeting_uses_live_session_send_text_not_tts(self):
        h = self._greeting_handler(is_openai_realtime=True)

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._fake_session.send_text.assert_awaited_once_with(
            "Hi, thanks for calling Acme.", respond=True
        )
        h._tts_pipeline.queue_tts.assert_not_awaited()

    def test_native_audio_greeting_skips_gracefully_if_session_not_ready(self):
        h = self._greeting_handler(is_openai_realtime=True)
        h._voice_orchestrator._openai_realtime_session = None

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._tts_pipeline.queue_tts.assert_not_awaited()

    def test_non_native_audio_greeting_still_uses_external_tts(self):
        h = self._greeting_handler(is_openai_realtime=False)

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        h._tts_pipeline.queue_tts.assert_awaited_once()
        queued = h._tts_pipeline.queue_tts.call_args[0][0]
        assert queued["text"] == "Hi, thanks for calling Acme."


def _fc(*, id: str, name: str, args: dict) -> tuple[str, str, str]:
    """OpenAI Realtime tool calls are plain (call_id, name, arguments_json)
    args (see OpenAIRealtimeSession.on_tool_call's contract), unlike
    Gemini's google.genai.types.FunctionCall object — no fake type needed."""
    import json

    return (id, name, json.dumps(args))


class TestCalendlyToolCallWiring:
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
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["tools"] is not None
        assert len(kwargs["tools"]) == 2
        assert kwargs["on_tool_call"] == orch._on_openai_realtime_tool_call

    def test_session_start_passes_no_tools_when_calendly_disabled(self):
        h = self._handler_with_calendly(enabled=False)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        kwargs = fake_session.start.call_args.kwargs
        assert kwargs["tools"] is None
        assert kwargs["on_tool_call"] is None

    def test_on_tool_call_resolves_and_sends_response(self):
        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_tool_response = AsyncMock()
        orch._openai_realtime_session = fake_session

        call_id, name, args_json = _fc(id="call-1", name="check_availability", args={"date": "tomorrow"})
        h._execute_calendly_tool_call = AsyncMock(return_value={"slots": ["09:00"]})

        asyncio.run(orch._on_openai_realtime_tool_call(call_id, name, args_json))

        h._execute_calendly_tool_call.assert_awaited_once_with("check_availability", {"date": "tomorrow"})
        fake_session.send_tool_response.assert_awaited_once_with("call-1", {"slots": ["09:00"]})

    def test_on_tool_call_noop_when_session_not_set(self):
        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        orch._openai_realtime_session = None

        asyncio.run(orch._on_openai_realtime_tool_call("c1", "check_availability", "{}"))
        h._execute_calendly_tool_call.assert_not_awaited()

    def test_on_tool_call_survives_execute_exception(self):
        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_tool_response = AsyncMock()
        orch._openai_realtime_session = fake_session
        h._execute_calendly_tool_call = AsyncMock(side_effect=RuntimeError("boom"))

        asyncio.run(orch._on_openai_realtime_tool_call("c1", "check_availability", "{}"))

        fake_session.send_tool_response.assert_awaited_once_with(
            "c1", {"error": "internal error executing tool call"}
        )

    def test_on_tool_call_survives_malformed_arguments_json(self):
        h = self._handler_with_calendly(enabled=True)
        orch = VoiceOrchestrator(h)
        fake_session = MagicMock()
        fake_session.send_tool_response = AsyncMock()
        orch._openai_realtime_session = fake_session
        h._execute_calendly_tool_call = AsyncMock(return_value={"ok": True})

        asyncio.run(orch._on_openai_realtime_tool_call("c1", "check_availability", "not json"))

        h._execute_calendly_tool_call.assert_awaited_once_with("check_availability", {})


class TestPerAgentVoiceWiring:
    def test_valid_agent_voice_passed_through(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h.agent.tts_voice_external_id = "cedar"
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        assert fake_session.start.call_args.kwargs["voice_name"] == "cedar"

    def test_unset_or_invalid_agent_voice_falls_back_to_default(self):
        from app.services.openai_realtime_service import DEFAULT_VOICE_NAME

        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        h.agent.tts_voice_external_id = "21m00Tcm4TlvDq8ikWAM"
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.start = AsyncMock()

        with patch(
            "app.services.openai_realtime_service.OpenAIRealtimeSession", return_value=fake_session
        ), patch(
            "app.routers.bidirectional_stream.BidirectionalStreamHandler.build_system_prompt",
            new=AsyncMock(return_value="system prompt"),
        ):
            asyncio.run(orch._start_openai_realtime_session(h))

        assert fake_session.start.call_args.kwargs["voice_name"] == DEFAULT_VOICE_NAME


class TestShutdownClosesSession:
    def test_shutdown_closes_openai_realtime_session(self):
        h = _fake_handler(llm_model=NATIVE_AUDIO_MODEL)
        orch = VoiceOrchestrator(h)

        fake_session = MagicMock()
        fake_session.close = AsyncMock()
        orch._openai_realtime_session = fake_session

        asyncio.run(orch.shutdown())

        fake_session.close.assert_awaited_once()
        assert orch._openai_realtime_session is None
