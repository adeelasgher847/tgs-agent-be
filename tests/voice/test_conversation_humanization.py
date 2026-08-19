"""
Conversation Flow & AI Humanization Tests
==========================================
Tests the full prompt assembly, multi-turn conversation flow, and AI response
humanization without any third-party services (no STT/TTS/LLM providers needed).

Validates:
- System prompt structure and humanization directives
- Mood-aware tone adaptation across user emotions
- Quick-ack eligibility for natural pacing
- Turn context signals (mood, brevity, phase) injected into prompts
- Multi-turn conversation continuity and history management
- Control token stripping from spoken output
- Greeting flow (first_message vs greeting_message preference)

Run:
    pytest tests/voice/test_conversation_humanization.py -v
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler
from app.voice.conversation_orchestrator import (
    ConversationOrchestrator,
    VOICE_TUNABLES,
    should_send_quick_ack,
)
from app.voice.tone_adapter import tone_adapter
from app.voice.turn_signals import (
    TurnContext,
    UserMood,
    build_turn_context,
    build_user_signals_block,
    detect_mood,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test handler builder
# ─────────────────────────────────────────────────────────────────────────────

def _base_handler(
    *,
    system_prompt: str = "You are a friendly scheduling assistant for Acme Plumbing.",
    agent_name: str = "Sarah",
    language: str = "en",
    business_knowledge: str = "",
) -> Handler:
    """Build a minimal Handler for prompt/conversation testing — no DB, WS, or providers."""
    h = object.__new__(Handler)

    # STT state
    h._turn_response_started = False
    h._turn_response_seed_text = ""
    h._last_interim_text = ""
    h._last_interim_sent_ts = 0.0
    h._enable_interim_llm = False
    h._min_interim_words = 3
    h._min_interim_confidence = 0.4
    h._min_interim_interval_sec = 0.2
    h._rag_prefetch_min_words = 2
    h._rag_prefetch_min_confidence = 0.05
    h._stt_min_final_confidence = 0.26
    h._enable_soft_final_fallback = True
    h._stt_soft_min_final_confidence = 0.16
    h._stt_soft_min_words = 2
    h._STT_DEDUP_FINAL_WINDOW_SEC = 6.0
    h._stt_last_final_raw = ""
    h._stt_last_final_monotonic = 0.0

    # Barge-in
    h.is_speaking = False
    h._is_tts_playing = False
    h._barge_in_min_conf = 0.26
    h._barge_in_min_conf_1w = 0.52
    h._barge_in_min_words = 2
    h._barge_in_rejected_while_playing = 0
    h._tts_cancel = asyncio.Event()
    h._tts_play_start_ts = 0.0
    h._barge_in_dead_zone_ms = 600.0

    # TTS
    h._tts_lock = asyncio.Lock()
    h._tts_pipeline = MagicMock()
    h._tts_pipeline.queue_tts = AsyncMock()
    h._tts_pipeline.cancel_current_and_clear_queue = AsyncMock()
    h._tts_pipeline.is_speaking = False
    h._tts_pipeline.reset_previous_text_continuity = MagicMock()
    h._llm_response_task = None
    h._use_ssml = False
    h._prev_tts_tail = b""

    # Prefetch
    h._rag_prefetch_task = None
    h._rag_prefetch_user_text = ""
    h._speculative_prefetch_task = None

    # Locks
    h._voice_transcript_lock = asyncio.Lock()
    h._llm_turn_serial_lock = asyncio.Lock()

    # Call session
    h.call_session = MagicMock()
    h.call_session.id = uuid.uuid4()
    h.call_session.tenant_id = uuid.uuid4()
    h.call_session.call_sid = "CA_test_humanization"
    h.call_session.call_transcript = []
    h.call_session.call_metadata = {}
    h.call_session.agent_id = uuid.uuid4()
    h.call_session.assistant_phone_number = "+15005550006"
    h.call_session.to_number = "+19999999999"
    h.call_session.from_number = "+19999999998"
    h.call_session.call_type = "inbound"
    h.call_session.customer_name = None
    h.call_session.customer_phone = None
    h.call_session.customer_email = None
    h.call_session.user_id = uuid.uuid4()

    # Agent
    h.agent = MagicMock()
    h.agent.id = h.call_session.agent_id
    h.agent.name = agent_name
    h.agent.system_prompt = system_prompt
    h.agent.first_message = "Hi there! How can I help you today?"
    h.agent.greeting_message = None
    h.agent.language = language
    h.agent.tts_voice = MagicMock()
    h.agent.tts_voice.external_voice_id = "en-US-Chirp3-HD-Achernar"
    h.agent.tts_provider = MagicMock()
    h.agent.tts_provider.slug = "google"
    h.agent.model = MagicMock()
    h.agent.model.name = "gpt-4o-mini"
    h.agent.model.api_key = None
    h.agent.model.max_tokens = 512
    h.agent.model.system_prompt = None
    h.agent.agent_max_tokens = None
    h.agent.provider = MagicMock()
    h.agent.provider.name = "openai"
    h.agent.transfer_route = None

    h.db = MagicMock()
    h.websocket = MagicMock()
    h.stream_sid = "MZ_test_stream"
    h.call_sid = "CA_test_humanization"
    h.agent_id = str(h.agent.id)
    h.call_session_id = str(h.call_session.id)

    # Conversation
    h._conversation_history_cache = []
    h._recent_agent_pairs = []
    h._DUP_USER_TURN_WINDOW_SEC = 15.0
    h._AGENT_LINE_DEDUP_WINDOW_SEC = 25.0
    h._RECENT_AGENT_PAIRS_MAX = 5
    h._llm_last_answered_transcript = ""
    h._llm_last_answered_ts = 0.0
    h._last_quick_ack_user_norm = ""
    h._last_quick_ack_mono = 0.0

    # KB
    h._cached_inbound_kb_block = ""
    h._cached_business_knowledge_block = business_knowledge
    h._kb_cache_ready = True

    # Calendar
    h._last_offered_calendar_slots = []
    h._last_requested_calendar_date = None
    h._last_selected_calendar_slot = None
    h._booking_memory = {}

    # Lifecycle
    h._call_ended = False
    h._stop_event = asyncio.Event()
    h._post_call_orchestration_scheduled = False
    h._pending_resume_screening_qualify = False
    h._auto_greeting_sent = False
    h._recording_started = False
    h._screening_decline_handled = False
    h._in_progress_sent = False
    h._user_picked_up = True
    h._stt_active = True
    h._email_stt_endpointing_upgraded = False
    h._stt_deferred_endpointing_ms = None
    h._twilio_buffer_primed = False

    # Metrics
    h._voice_metrics = MagicMock()
    h._metric_stt_final_ts = 0.0
    h._metric_gen_start_ts = 0.0
    h._metric_first_token_ts = 0.0
    h._metric_first_audio_ts = 0.0
    h._metric_barge_in_ts = 0.0
    h._metric_audio_cut_ts = 0.0

    # Mocked helpers
    h._prefetch_rag_context = AsyncMock(return_value=("", {}))
    h._send_in_progress_status = AsyncMock()
    h._add_to_transcript = AsyncMock()
    h._remember_agent_turn = MagicMock()
    h._update_booking_memory_from_user_turn = MagicMock()
    h._is_booking_intent_turn = MagicMock(return_value=False)
    h._is_booking_context_active = MagicMock(return_value=False)
    h._is_duplicate_agent_line = MagicMock(return_value=False)
    h._is_agent_self_echo = MagicMock(return_value=False)
    h._has_recent_duplicate_reply_for = MagicMock(return_value=False)
    h._schedule_recreate_stt_for_email_collection = MagicMock()
    h._stream_sid_ready = asyncio.Event()
    h._stream_sid_ready.set()

    # Handler class attrs
    h.TTS_FLUSH_MIN_WORDS = VOICE_TUNABLES.tts_flush_min_words
    h.TTS_FLUSH_MAX_WORDS = VOICE_TUNABLES.tts_flush_max_words
    h.HISTORY_MAX_MESSAGES = VOICE_TUNABLES.history_max_messages

    return h


def _async_llm_stream(*chunks: str):
    """Return an async generator that yields text chunks (simulates LLM streaming)."""
    async def _gen(*args, **kwargs):
        for chunk in chunks:
            yield chunk
    return _gen


def _extract_tts_texts(mock_queue_tts: AsyncMock) -> list[str]:
    """Extract all text strings queued to TTS from mock calls."""
    texts = []
    for call in mock_queue_tts.call_args_list:
        arg = call[0][0] if call[0] else call.kwargs.get("text", "")
        text = arg["text"] if isinstance(arg, dict) else str(arg)
        texts.append(text)
    return texts


def _full_tts_output(mock_queue_tts: AsyncMock) -> str:
    """Concatenate all TTS-queued text into one string."""
    return " ".join(t for t in _extract_tts_texts(mock_queue_tts) if t.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PROMPT STRUCTURE — verify humanization directives are present
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPromptHumanization:
    """Verify the assembled system prompt contains all humanization directives."""

    def _capture_system_prompt(self, h: Handler, user_text: str = "hello") -> str:
        """Run generate_and_stream_response and capture the system_prompt sent to LLM."""
        captured = {}

        async def _spy_stream(prompt=None, system_prompt=None, **kwargs):
            captured["system_prompt"] = system_prompt
            captured["prompt"] = prompt
            yield "Sure, I can help."

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy_stream), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response(user_text, 0.9))

        return captured.get("system_prompt", "")

    def test_prompt_contains_voice_first_directive(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "VOICE-FIRST" in prompt, "Prompt must instruct voice-first output"

    def test_prompt_contains_human_phone_style(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "HUMAN:" in prompt
        assert "How may I assist you" in prompt
        assert "contractions" in prompt.lower()

    def test_prompt_contains_natural_fillers_directive(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "NATURAL" in prompt
        assert any(filler in prompt for filler in ["umm", "hmm", "oh", "alright"])

    def test_prompt_contains_no_robot_talk_or_concise(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "short sentences" in prompt.lower() or "concise" in prompt.lower() or "20 words" in prompt

    def test_prompt_contains_text_hygiene_rules(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "TEXT HYGIENE" in prompt or "Avoid" in prompt

    def test_prompt_contains_no_repetition_rule(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "NO REPETITION" in prompt or "REPETITION" in prompt

    def test_prompt_contains_plain_text_only_rule(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "PLAIN TEXT" in prompt or "plain text" in prompt

    def test_prompt_contains_agent_name_as_persona(self):
        h = _base_handler(agent_name="Sarah")
        prompt = self._capture_system_prompt(h)
        assert "Sarah" in prompt

    def test_prompt_contains_custom_instructions(self):
        h = _base_handler(system_prompt="Always greet by name. Ask about their plumbing issue.")
        prompt = self._capture_system_prompt(h)
        assert "plumbing issue" in prompt

    def test_prompt_injects_user_signals_block(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h, "I'm really frustrated with this")
        assert "USER_SIGNALS" in prompt
        assert "inferred_mood" in prompt

    def test_prompt_contains_current_datetime(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "CURRENT DATE & TIME" in prompt

    def test_prompt_contains_grounding_rules(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "GROUNDING RULES" in prompt or "BUSINESS FACTS" in prompt

    def test_prompt_contains_termination_rule(self):
        h = _base_handler()
        prompt = self._capture_system_prompt(h)
        assert "[END_CALL]" in prompt
        assert "TERMINATION" in prompt or "goodbye" in prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOOD DETECTION — comprehensive mood → prompt steering
# ─────────────────────────────────────────────────────────────────────────────

class TestMoodDetection:
    """Verify mood detection across the full emotional spectrum."""

    @pytest.mark.parametrize("text,expected", [
        ("This is an emergency please help me", UserMood.URGENT),
        ("I need help right now immediately", UserMood.URGENT),
        ("I am so angry about this service", UserMood.ANGRY),
        ("This is absolutely unacceptable", UserMood.ANGRY),
        ("I am frustrated, nothing is working", UserMood.FRUSTRATED),
        ("This is ridiculous and a waste of time", UserMood.FRUSTRATED),
        ("Thank you so much, that's wonderful", UserMood.HAPPY),
        ("I really appreciate your help", UserMood.HAPPY),
        ("My father passed away last week", UserMood.SAD),
        ("I can't afford this anymore", UserMood.SAD),
        ("What are your business hours", UserMood.NEUTRAL),
        ("Can I schedule for next Tuesday", UserMood.NEUTRAL),
    ])
    def test_mood_detection_accuracy(self, text, expected):
        assert detect_mood(text, 0.9) == expected

    def test_mood_signals_injected_for_frustrated_caller(self):
        ctx = build_turn_context("This is not working at all, I'm so frustrated", 0.85)
        block = build_user_signals_block(ctx)
        assert "frustrated" in block.lower()
        assert "acknowledge" in block.lower()

    def test_mood_signals_injected_for_sad_caller(self):
        ctx = build_turn_context("Unfortunately my mother passed away", 0.9)
        block = build_user_signals_block(ctx)
        assert "sad" in block.lower()
        assert "warm" in block.lower() or "gentle" in block.lower()

    def test_respond_briefly_for_short_utterance(self):
        ctx = build_turn_context("yes", 0.9)
        assert ctx.respond_briefly is True

    def test_respond_normally_for_longer_utterance(self):
        ctx = build_turn_context(
            "I'd like to schedule an appointment for next Thursday afternoon if possible", 0.9
        )
        assert ctx.respond_briefly is False

    def test_booking_phase_detected(self):
        ctx = build_turn_context("I want to book for tomorrow", 0.9, booking_context_active=True)
        assert ctx.conversation_phase == "booking"

    def test_general_phase_when_not_booking(self):
        ctx = build_turn_context("What services do you offer", 0.9, booking_context_active=False)
        assert ctx.conversation_phase == "general"


# ─────────────────────────────────────────────────────────────────────────────
# 3. TONE ADAPTATION — LLM output is reshaped based on mood
# ─────────────────────────────────────────────────────────────────────────────

class TestToneAdaptation:
    """Verify tone_adapter reshapes LLM output to match caller mood."""

    def test_strips_chipper_opening_for_sad_caller(self):
        ctx = build_turn_context("I feel terrible about this situation", 0.85)
        adapted = tone_adapter("Great! I understand your concern.", ctx, use_ssml=False)
        assert not adapted.startswith("Great!")
        assert "understand" in adapted

    def test_strips_chipper_opening_for_frustrated_caller(self):
        ctx = build_turn_context("This is ridiculous", 0.8)
        adapted = tone_adapter("Awesome! Let me help you with that.", ctx, use_ssml=False)
        assert not adapted.startswith("Awesome!")

    def test_strips_chipper_opening_for_angry_caller(self):
        ctx = build_turn_context("I am so angry about this", 0.85)
        adapted = tone_adapter("Perfect! I'll look into that.", ctx, use_ssml=False)
        assert not adapted.startswith("Perfect!")

    def test_preserves_neutral_response_for_neutral_caller(self):
        ctx = build_turn_context("What are your hours?", 0.9)
        text = "We are open Monday through Friday, 9 to 5."
        assert tone_adapter(text, ctx, use_ssml=False) == text

    def test_preserves_response_for_happy_caller(self):
        ctx = build_turn_context("Thank you so much!", 0.9)
        text = "You're welcome! Have a great day."
        assert tone_adapter(text, ctx, use_ssml=False) == text

    def test_softens_exclamation_for_sad_caller(self):
        ctx = build_turn_context("Unfortunately my mother passed away", 0.8)
        assert ctx.mood == UserMood.SAD
        adapted = tone_adapter("Yay! Let's get started.", ctx, use_ssml=False)
        assert "Yay!" not in adapted

    def test_softens_excitement_markers_for_angry_mood(self):
        ctx = build_turn_context("I am furious about this terrible service", 0.9)
        assert ctx.mood == UserMood.ANGRY
        adapted = tone_adapter("No worries! I'll fix it right away.", ctx, use_ssml=False)
        assert "No worries!" not in adapted
        assert "No worries." in adapted


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUICK-ACK ELIGIBILITY — natural conversational pacing
# ─────────────────────────────────────────────────────────────────────────────

class TestQuickAckHumanization:
    """Quick-acks add natural pacing (Got it, I see, Mm-hmm) — validate eligibility."""

    def test_eligible_for_longer_query(self):
        assert should_send_quick_ack(
            "I'd like to schedule an appointment for next week", VOICE_TUNABLES.quick_ack
        ) is True

    def test_not_eligible_for_short_reply(self):
        assert should_send_quick_ack("yes", VOICE_TUNABLES.quick_ack) is False

    def test_not_eligible_for_empty_text(self):
        assert should_send_quick_ack("", VOICE_TUNABLES.quick_ack) is False

    def test_skipped_for_emotional_content(self):
        """Should NOT quick-ack emotional phrases like 'I need help' or 'this is urgent'."""
        assert should_send_quick_ack(
            "I need help this is urgent please help me", VOICE_TUNABLES.quick_ack
        ) is False

    def test_skipped_for_complaint(self):
        assert should_send_quick_ack(
            "I have a complaint about your service representative", VOICE_TUNABLES.quick_ack
        ) is False

    def test_skipped_for_angry_content(self):
        assert should_send_quick_ack(
            "This is a serious problem and I want it fixed", VOICE_TUNABLES.quick_ack
        ) is False

    def test_eligible_for_normal_booking_request(self):
        assert should_send_quick_ack(
            "Can I book an appointment for Tuesday afternoon", VOICE_TUNABLES.quick_ack
        ) is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. MULTI-TURN CONVERSATION — verify history builds correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurnConversation:
    """Simulate multi-turn calls and verify conversation history handling."""

    def test_history_text_included_in_prompt(self):
        """Conversation history must appear in the system prompt."""
        h = _base_handler()
        h.call_session.call_transcript = [
            {"role": "agent", "content": "Hi, how can I help?", "message_type": "agent_response"},
            {"role": "client", "content": "I need to reschedule", "message_type": "speech"},
            {"role": "agent", "content": "Sure, what day works?", "message_type": "agent_response"},
        ]

        captured = {}

        async def _spy(prompt=None, system_prompt=None, **kwargs):
            captured["system_prompt"] = system_prompt
            yield "Thursday works for me."

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("Thursday please", 0.9))

        prompt = captured.get("system_prompt", "")
        assert "reschedule" in prompt
        assert "what day works" in prompt.lower()

    def test_greeting_messages_excluded_from_history(self):
        """Greeting/system messages should be filtered out of conversation history."""
        h = _base_handler()
        h.call_session.call_transcript = [
            {"role": "agent", "content": "Hi there!", "message_type": "greeting"},
            {"role": "system", "content": "Call started", "message_type": "system"},
            {"role": "client", "content": "I need help", "message_type": "speech"},
        ]

        captured = {}

        async def _spy(prompt=None, system_prompt=None, **kwargs):
            captured["system_prompt"] = system_prompt
            yield "Of course."

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("can you help", 0.9))

        prompt = captured.get("system_prompt", "")
        assert "Call started" not in prompt

    def test_history_bounded_to_max_messages(self):
        """History should not exceed HISTORY_MAX_MESSAGES to keep prompt lean."""
        h = _base_handler()
        h.HISTORY_MAX_MESSAGES = 6
        h.call_session.call_transcript = [
            {"role": "client" if i % 2 == 0 else "agent",
             "content": f"Message {i}",
             "message_type": "speech" if i % 2 == 0 else "agent_response"}
            for i in range(20)
        ]

        captured = {}

        async def _spy(prompt=None, system_prompt=None, **kwargs):
            captured["system_prompt"] = system_prompt
            yield "Okay."

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("continue", 0.9))

        prompt = captured.get("system_prompt", "")
        assert "Message 0" not in prompt, "Old messages should be windowed out"
        assert "Message 19" in prompt, "Recent messages should be retained"


# ─────────────────────────────────────────────────────────────────────────────
# 6. RESPONSE NATURALNESS — LLM output → TTS text quality
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseNaturalness:
    """Verify LLM responses are cleaned and humanized before reaching TTS."""

    def test_control_tokens_never_spoken(self):
        """[END_CALL], [TRANSFER_CALL], [SCREENING_QUALIFIED] must be stripped."""
        h = _base_handler()

        fake_stream = _async_llm_stream(
            "Thanks for calling, have a great day! [END_CALL]"
        )

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=fake_stream), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("goodbye", 0.9))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "[END_CALL]" not in output
        assert "great day" in output.lower()

    def test_booking_tokens_never_spoken(self):
        """[CHECK_SLOTS] and [BOOK_APPOINTMENT] must be stripped from TTS."""
        h = _base_handler()

        fake_stream = _async_llm_stream(
            "Let me check availability. [CHECK_SLOTS:date=2026-05-20]"
        )

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=fake_stream), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("book appointment", 0.9))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "[CHECK_SLOTS" not in output

    def test_outcome_tokens_never_spoken(self):
        """[OUTCOME:...] tokens must be stripped from TTS."""
        h = _base_handler()

        fake_stream = _async_llm_stream(
            "I've noted your request. [OUTCOME:qualified]"
        )

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=fake_stream), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("I'm interested", 0.9))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "[OUTCOME:" not in output

    def test_natural_response_passes_through_unchanged(self):
        """Normal conversational text should not be mangled."""
        h = _base_handler()

        fake_stream = _async_llm_stream(
            "Sure, we have openings on Monday at 2 PM and Wednesday at 10 AM."
        )

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=fake_stream), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("when are you available", 0.9))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "Monday" in output
        assert "Wednesday" in output


# ─────────────────────────────────────────────────────────────────────────────
# 7. GREETING HUMANIZATION — first impression matters
# ─────────────────────────────────────────────────────────────────────────────

class TestGreetingHumanization:
    """Verify greeting flow for natural first impressions."""

    def test_greeting_uses_first_message(self):
        h = _base_handler()
        h.agent.greeting_message = None
        h.agent.first_message = "Hey there! Welcome to Acme Plumbing. What can I do for you?"

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "Acme Plumbing" in output

    def test_greeting_prefers_greeting_message(self):
        h = _base_handler()
        h.agent.greeting_message = "Hi! Thanks for calling Acme. How can I help?"
        h.agent.first_message = "This should NOT be used"

        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "Acme" in output
        assert "should NOT" not in output

    def test_greeting_skips_llm(self):
        """Greeting path must NOT call LLM — it uses pre-configured text."""
        h = _base_handler()
        llm_called = False

        async def _spy(*args, **kwargs):
            nonlocal llm_called
            llm_called = True
            yield "oops"

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy):
            asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))

        assert llm_called is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL CONVERSATION SCENARIO — multi-turn humanization flow
# ─────────────────────────────────────────────────────────────────────────────

class TestFullConversationScenario:
    """Simulate a complete call and verify humanization across turns."""

    def test_frustrated_caller_gets_empathetic_handling(self):
        """
        Scenario: Caller is frustrated → mood detected → prompt steers LLM to
        acknowledge first → tone adapter strips chipper openings from response.
        """
        h = _base_handler()

        # Simulate frustrated caller
        user_text = "I am frustrated, I have been waiting for two hours and nobody called me back"
        ctx = build_turn_context(user_text, 0.9)
        assert ctx.mood == UserMood.FRUSTRATED

        signals = build_user_signals_block(ctx)
        assert "frustrated" in signals.lower()
        assert "acknowledge" in signals.lower()

        # Simulate LLM responding with a chipper opening (bad)
        llm_output = "Great! I'm sorry to hear that. Let me look into this for you right away."
        adapted = tone_adapter(llm_output, ctx, use_ssml=False)
        assert not adapted.startswith("Great!")
        assert "sorry" in adapted.lower()

    def test_happy_caller_gets_natural_response(self):
        """Happy caller: no tone modification needed."""
        user_text = "Thank you so much, that was really helpful"
        ctx = build_turn_context(user_text, 0.9)
        assert ctx.mood == UserMood.HAPPY

        llm_output = "You're welcome! Is there anything else I can help with?"
        adapted = tone_adapter(llm_output, ctx, use_ssml=False)
        assert adapted == llm_output

    def test_urgent_caller_gets_brief_directed_response(self):
        """Urgent mood → respond_briefly + acknowledge-first directive."""
        user_text = "This is an emergency, I need help right now"
        ctx = build_turn_context(user_text, 0.9)
        assert ctx.mood == UserMood.URGENT
        assert ctx.respond_briefly is True

        signals = build_user_signals_block(ctx)
        assert "respond_briefly: yes" in signals
        assert "urgent" in signals.lower()

    def test_multi_turn_booking_flow(self):
        """
        Turn 1: Greeting
        Turn 2: User asks about services → LLM streams natural response
        Turn 3: User asks to book → LLM streams with booking context
        """
        h = _base_handler()

        # Turn 1: Greeting
        asyncio.run(h.generate_and_stream_response("", 1.0, is_greeting=True))
        assert h._tts_pipeline.queue_tts.called
        h._tts_pipeline.queue_tts.reset_mock()

        # Turn 2: User asks about services
        fake_stream2 = _async_llm_stream(
            "We offer plumbing repairs, drain cleaning, and water heater installation."
        )

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=fake_stream2), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("What services do you offer?", 0.9))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "plumbing" in output.lower()
        h._tts_pipeline.queue_tts.reset_mock()

        # Turn 3: Booking intent
        fake_stream3 = _async_llm_stream(
            "I'd be happy to help you schedule. What day works best for you?"
        )

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=fake_stream3), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response(
                "Can I book a drain cleaning for next week?", 0.9
            ))

        output = _full_tts_output(h._tts_pipeline.queue_tts)
        assert "schedule" in output.lower() or "day" in output.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 9. BUSINESS KNOWLEDGE IN PROMPT — grounding prevents hallucination
# ─────────────────────────────────────────────────────────────────────────────

class TestBusinessKnowledgeGrounding:
    """Verify business knowledge is correctly injected and grounding rules present."""

    def test_no_business_knowledge_adds_guard(self):
        """When no BK is loaded, the prompt must contain a 'do not invent' guard."""
        h = _base_handler(business_knowledge="")

        captured = {}

        async def _spy(prompt=None, system_prompt=None, **kwargs):
            captured["system_prompt"] = system_prompt
            yield "I don't have that information."

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response("What's your address?", 0.9))

        prompt = captured.get("system_prompt", "")
        assert "do not invent" in prompt.lower() or "DO NOT invent" in prompt

    def test_business_knowledge_injected_into_prompt(self):
        """Inbound KB cached at call start must appear in the prompt."""
        kb = (
            "# INBOUND KNOWLEDGE\n"
            "Business Name: Northside Drain Co\n"
            "Address: 123 Main St, Dallas, TX\n"
            "Phone: (555) 123-4567\n"
            "Services: Drain cleaning, Water heater repair, Pipe installation"
        )
        h = _base_handler(system_prompt="You are a friendly scheduling assistant.")
        h._cached_inbound_kb_block = kb
        h._kb_cache_ready = True

        captured = {}

        async def _spy(prompt=None, system_prompt=None, **kwargs):
            captured["system_prompt"] = system_prompt
            yield "We're located at 123 Main St."

        with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy), \
             patch("app.voice.rag_context.build_rag_context_block_with_trace",
                   return_value=("", {"status": "skipped"})), \
             patch.object(h, "_add_to_transcript", new=AsyncMock()), \
             patch.object(h, "_send_in_progress_status", new=AsyncMock()):
            asyncio.run(h.generate_and_stream_response(
                "What is your address and what services do you offer at your location?", 0.9
            ))

        prompt = captured.get("system_prompt", "")
        assert "Northside Drain Co" in prompt
        assert "123 Main St" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 10. TTS STABILITY HINTS — mood → prosody mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestTtsStabilityHints:
    """Verify TTS stability hints are set based on mood for prosody control."""

    def test_neutral_stability_is_moderate(self):
        ctx = build_turn_context("What time do you open?", 0.9)
        assert ctx.tts_stability_hint is not None
        assert 0.40 <= ctx.tts_stability_hint <= 0.55

    def test_angry_gets_lower_stability(self):
        ctx = build_turn_context("I am furious about this", 0.9)
        assert ctx.tts_stability_hint is not None
        assert ctx.tts_stability_hint <= 0.50

    def test_sad_gets_calmer_stability(self):
        ctx = build_turn_context("Unfortunately my mother passed away", 0.9)
        assert ctx.tts_stability_hint is not None
        assert ctx.tts_stability_hint >= 0.50

    def test_happy_stability_is_moderate(self):
        ctx = build_turn_context("Thank you that's wonderful!", 0.9)
        assert ctx.tts_stability_hint is not None
        assert 0.45 <= ctx.tts_stability_hint <= 0.55
