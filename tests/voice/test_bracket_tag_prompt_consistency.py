"""
Bracket-tag prompt-consistency regression tests (Phase 5-4).

These tests guard against the specific contradiction fixed in this phase:
before the fix, the base/custom/model system-prompt branches on both
transports (Twilio's ``BidirectionalStreamHandler`` in
``app/routers/bidirectional_stream.py`` and LiveKit's
``ConversationOrchestrator`` in ``app/voice/conversation_orchestrator.py``)
could assemble a system prompt containing BOTH a generic
"NO BRACKET TAGS: Never output..." instruction AND the
``build_elevenlabs_audio_tag_prompt_block()`` block saying tags ARE allowed,
whenever the agent used ElevenLabs TTS together with a custom
``system_prompt`` or ``model.system_prompt``.

Only prompt-text assembly is asserted here (never LLM compliance or audio
output — those are untestable in this codebase). External LLM/DB
boundaries are mocked; only the real prompt-building code path runs.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_call_pipeline import _base_handler


ELEVEN_TAG_MARKER = "ELEVENLABS AUDIO TAGS"
GENERIC_NO_TAG_MARKER = "NO BRACKET TAGS"


def _assert_single_tag_source(system_prompt: str) -> None:
    """Exactly one bracket-tag instruction source may ever be present."""
    has_eleven_block = ELEVEN_TAG_MARKER in system_prompt
    has_generic_never = GENERIC_NO_TAG_MARKER in system_prompt
    assert not (has_eleven_block and has_generic_never), (
        "Assembled prompt contains BOTH the ElevenLabs 'tags allowed' block "
        "and the generic 'never use tags' instruction — contradiction "
        "reintroduced.\n---\n" + system_prompt
    )
    # At least one of the two sources must be present so tag behavior is
    # always specified one way or the other.
    assert has_eleven_block or has_generic_never


# ─────────────────────────────────────────────────────────────────────────
# Twilio (BidirectionalStreamHandler) — inline prompt assembly
# ─────────────────────────────────────────────────────────────────────────

def _capture_twilio_system_prompt(h: BidirectionalStreamHandler) -> list:
    captured = []

    async def _spy_stream(prompt=None, system_prompt=None, **kwargs):
        captured.append(system_prompt or "")
        yield "Sure, here is a short reply."

    return captured, _spy_stream


class TestTwilioBracketTagPromptConsistency:
    def _run(self, h, spy_stream):
        with patch(
            "app.routers.bidirectional_stream.openai_service.stream_text", new=spy_stream
        ), patch.object(h, "_add_to_transcript", new=AsyncMock()), patch.object(
            h, "_send_in_progress_status", new=AsyncMock()
        ):
            asyncio.run(h.generate_and_stream_response("Hello there", 0.9))

    def test_base_prompt_elevenlabs_no_contradiction(self):
        h = _base_handler()
        h.agent.system_prompt = None  # force base_prompt branch
        h.agent.model.system_prompt = None
        h.agent.tts_provider.slug = "elevenlabs"
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        assert captured, "LLM must have been invoked"
        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER in sp
        assert GENERIC_NO_TAG_MARKER not in sp

    def test_base_prompt_google_no_contradiction(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = None
        h.agent.tts_provider.slug = "google"
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER not in sp
        assert GENERIC_NO_TAG_MARKER in sp

    def test_custom_system_prompt_elevenlabs_no_contradiction(self):
        h = _base_handler()
        h.agent.system_prompt = "You help schedule appointments."
        h.agent.tts_provider.slug = "elevenlabs"
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER in sp
        assert GENERIC_NO_TAG_MARKER not in sp

    def test_custom_system_prompt_google_no_contradiction(self):
        h = _base_handler()
        h.agent.system_prompt = "You help schedule appointments."
        h.agent.tts_provider.slug = "google"
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER not in sp
        assert GENERIC_NO_TAG_MARKER in sp

    def test_model_system_prompt_elevenlabs_no_contradiction(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = "Follow the recruiting script."
        h.agent.tts_provider.slug = "elevenlabs"
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER in sp
        assert GENERIC_NO_TAG_MARKER not in sp

    def test_model_system_prompt_google_no_contradiction(self):
        h = _base_handler()
        h.agent.system_prompt = None
        h.agent.model.system_prompt = "Follow the recruiting script."
        h.agent.tts_provider.slug = "google"
        captured, spy = _capture_twilio_system_prompt(h)
        self._run(h, spy)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER not in sp
        assert GENERIC_NO_TAG_MARKER in sp


# ─────────────────────────────────────────────────────────────────────────
# LiveKit (ConversationOrchestrator) — shared class, invoked via a minimal
# duck-typed fake handler exposing only the attributes
# generate_and_stream_response() actually reads/writes on self._h.
# ─────────────────────────────────────────────────────────────────────────

class _FakeCallSession:
    """Falsy call_session so every DB-dependent prompt-building branch
    (CRM context, KB, caller memory, HubSpot field mapping, business
    knowledge fetch) is skipped — keeping this test's only real
    dependencies at the LLM-boundary, matching the module docstring above.
    """

    def __bool__(self) -> bool:
        return False


def _fake_livekit_handler(*, tts_slug: str, custom_system_prompt=None, model_system_prompt=None):
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
    h.call_session = _FakeCallSession()
    h.call_flow = None

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

    # Deterministic legacy TTS resolution (bypass ticket-based path).
    agent.tts_provider_slug = None
    agent.tts_language = None
    agent.tts_voice_external_id = None
    agent.tts_settings_json = {}
    agent.encrypted_elevenlabs_api_key = None
    agent.tts_provider = MagicMock()
    agent.tts_provider.slug = tts_slug
    agent.tts_voice = MagicMock()
    agent.tts_voice.external_voice_id = "voice-1"

    h.agent = agent
    return h


class TestLiveKitBracketTagPromptConsistency:
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

    def test_base_prompt_elevenlabs_no_contradiction(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="elevenlabs")
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        assert captured
        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER in sp
        assert GENERIC_NO_TAG_MARKER not in sp

    def test_base_prompt_google_no_generic_tag_line_needed(self, monkeypatch):
        """LiveKit's base path has no generic 'NO BRACKET TAGS' line at all
        (intentional — the elevenlabs_audio_tag_block is simply empty for
        non-ElevenLabs). This must stay true: no contradiction either way."""
        h = _fake_livekit_handler(tts_slug="google")
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        assert ELEVEN_TAG_MARKER not in sp
        # Base LiveKit path intentionally omits the generic line too; just
        # confirm no contradiction is possible (can't have both).
        assert not (ELEVEN_TAG_MARKER in sp and GENERIC_NO_TAG_MARKER in sp)

    def test_custom_system_prompt_elevenlabs_no_contradiction(self, monkeypatch):
        h = _fake_livekit_handler(
            tts_slug="elevenlabs", custom_system_prompt="You help schedule appointments."
        )
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER in sp
        assert GENERIC_NO_TAG_MARKER not in sp

    def test_custom_system_prompt_google_no_contradiction(self, monkeypatch):
        h = _fake_livekit_handler(
            tts_slug="google", custom_system_prompt="You help schedule appointments."
        )
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER not in sp
        assert GENERIC_NO_TAG_MARKER in sp

    def test_model_system_prompt_elevenlabs_no_contradiction(self, monkeypatch):
        h = _fake_livekit_handler(
            tts_slug="elevenlabs", model_system_prompt="Follow the recruiting script."
        )
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER in sp
        assert GENERIC_NO_TAG_MARKER not in sp

    def test_model_system_prompt_google_no_contradiction(self, monkeypatch):
        h = _fake_livekit_handler(
            tts_slug="google", model_system_prompt="Follow the recruiting script."
        )
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        _assert_single_tag_source(sp)
        assert ELEVEN_TAG_MARKER not in sp
        assert GENERIC_NO_TAG_MARKER in sp
