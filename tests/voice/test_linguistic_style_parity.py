"""
Linguistic-humanization parity regression between Twilio
(``BidirectionalStreamHandler``) and LiveKit/browser
(``ConversationOrchestrator``).

Prior to this fix, the two transports gave the LLM CONTRADICTORY guidance
on the same axis of linguistic humanization: Twilio's STYLE & TONE block
consistently told the model NOT to add filler words unless genuinely
appropriate to context (matching the "conservative humanization" principle
— see app.voice.humanization_engine's module docstring, which also notes
audio-level filler insertion was disabled upstream due to clicking
artifacts), while LiveKit's custom/model-prompt branches instead
instructed the model to actively insert "umm"/"uhm"/"hmm" fillers
"occasionally". Humanization behavior must not diverge between transports
(see CLAUDE.md's Voice pipeline section) — this was a real violation of
that requirement, not just a style nit: the SAME conversational situation
could produce a chattier, filler-laden response on browser calls and a
clean one on phone calls.

These tests assert the STYLE & TONE guidance text is now byte-identical
across transports for all three prompt-source branches (base / custom /
model), and that the discredited "use uhm/hmm fillers" instruction is gone
entirely.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_call_pipeline import _base_handler

# The exact NATURAL guidance line both transports must share verbatim.
_CONSERVATIVE_NATURAL_LINE = (
    "- NATURAL: Speak naturally and conversationally. Answer directly. Do not "
    "add artificial hesitation, filler words, acknowledgements, or "
    "conversational padding unless they are genuinely appropriate to the "
    "context."
)
_STRUCTURED_INFO_PACING_MARKER = "STRUCTURED INFO PACING"
_TRANSITION_PACING_MARKER = "TRANSITION PACING"
_DISCREDITED_FILLER_PHRASES = ("uhm", "use fillers", "natural fillers/interjections")


def _capture_twilio_system_prompt(h: BidirectionalStreamHandler) -> list:
    captured = []

    async def _spy_stream(prompt=None, system_prompt=None, **kwargs):
        captured.append(system_prompt or "")
        yield "Sure, here is a short reply."

    return captured, _spy_stream


class TestTwilioStyleAndTone:
    def _run(self, h, spy_stream):
        with patch(
            "app.routers.bidirectional_stream.openai_service.stream_text", new=spy_stream
        ), patch.object(h, "_add_to_transcript", new=AsyncMock()), patch.object(
            h, "_send_in_progress_status", new=AsyncMock()
        ):
            asyncio.run(h.generate_and_stream_response("Hello there", 0.9))

    def test_base_prompt_has_conservative_natural_line_and_pacing_rules(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = None
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        assert _CONSERVATIVE_NATURAL_LINE in sp
        assert _STRUCTURED_INFO_PACING_MARKER in sp
        assert _TRANSITION_PACING_MARKER in sp

    def test_custom_prompt_has_conservative_natural_line_and_pacing_rules(self):
        h = _base_handler()
        h.agent.system_prompt = "Custom instructions for this agent."
        h.agent.model.system_prompt = None
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        assert _CONSERVATIVE_NATURAL_LINE in sp
        assert _STRUCTURED_INFO_PACING_MARKER in sp
        assert _TRANSITION_PACING_MARKER in sp

    def test_model_prompt_has_conservative_natural_line_and_pacing_rules(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = "Model instructions for this agent."
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        assert _CONSERVATIVE_NATURAL_LINE in sp
        assert _STRUCTURED_INFO_PACING_MARKER in sp
        assert _TRANSITION_PACING_MARKER in sp


def _fake_livekit_handler(*, custom_system_prompt=None, model_system_prompt=None):
    h = MagicMock()
    h.TTS_FLUSH_MIN_WORDS = 3
    h.TTS_FLUSH_MAX_WORDS = 30
    h._add_to_transcript = AsyncMock()
    h._current_turn_user_text = ""
    h._current_turn_stt_confidence = 0.0
    h._prev_tts_tail = b""
    h._tts_cancel = asyncio.Event()
    h._twilio_buffer_primed = False
    h._use_ssml = False
    h.db = None
    h.call_flow = None

    class _FakeCallSession:
        def __bool__(self) -> bool:
            return False

    h.call_session = _FakeCallSession()

    tts_pipeline = MagicMock()
    tts_pipeline.queue_tts = AsyncMock()
    tts_pipeline.reset_previous_text_continuity = MagicMock()
    h._tts_pipeline = tts_pipeline

    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.name = "TestAgent"
    agent.language = "en"
    agent.system_prompt = custom_system_prompt
    agent.model = MagicMock()
    agent.model.system_prompt = model_system_prompt
    agent.model.temperature = 30
    agent.model.max_tokens = 512
    agent.model.api_key = None
    agent.model.model_name = "gpt-4o-mini"
    agent.agent_temperature = None
    agent.agent_max_tokens = None
    agent.llm_model = None
    agent.provider = MagicMock()
    agent.provider.name = "openai"
    agent.transfer_route = None

    agent.tts_provider_slug = None
    agent.tts_language = None
    agent.tts_voice_external_id = None
    agent.tts_settings_json = {}
    agent.encrypted_elevenlabs_api_key = None
    agent.tts_provider = MagicMock()
    agent.tts_provider.slug = "elevenlabs"
    agent.tts_voice = MagicMock()
    agent.tts_voice.external_voice_id = "voice-1"

    h.agent = agent
    return h


class TestLiveKitStyleAndTone:
    def _run(self, orchestrator, monkeypatch, captured):
        async def _stub_stream(**kwargs):
            captured.append(kwargs.get("system_prompt") or "")
            yield "Sure, here is a short reply."

        stub_service = MagicMock()
        stub_service.stream_text = _stub_stream
        monkeypatch.setattr(
            "app.core.agent_runtime.llm_service_for_provider", lambda slug: stub_service
        )
        asyncio.run(orchestrator.generate_and_stream_response("Hello there", 0.9))

    def test_base_prompt_has_conservative_natural_line_and_pacing_rules(self, monkeypatch):
        h = _fake_livekit_handler()
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        assert _CONSERVATIVE_NATURAL_LINE in sp
        assert _STRUCTURED_INFO_PACING_MARKER in sp
        assert _TRANSITION_PACING_MARKER in sp
        for phrase in _DISCREDITED_FILLER_PHRASES:
            assert phrase not in sp.lower()

    def test_custom_prompt_has_conservative_natural_line_and_pacing_rules(self, monkeypatch):
        h = _fake_livekit_handler(custom_system_prompt="You help schedule appointments.")
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        assert _CONSERVATIVE_NATURAL_LINE in sp
        assert _STRUCTURED_INFO_PACING_MARKER in sp
        assert _TRANSITION_PACING_MARKER in sp
        for phrase in _DISCREDITED_FILLER_PHRASES:
            assert phrase not in sp.lower()

    def test_model_prompt_has_conservative_natural_line_and_pacing_rules(self, monkeypatch):
        h = _fake_livekit_handler(model_system_prompt="Follow the recruiting script.")
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        assert _CONSERVATIVE_NATURAL_LINE in sp
        assert _STRUCTURED_INFO_PACING_MARKER in sp
        assert _TRANSITION_PACING_MARKER in sp
        for phrase in _DISCREDITED_FILLER_PHRASES:
            assert phrase not in sp.lower()
