"""
Regression coverage for the structural contact-intake fix: confirmed
name/email/phone/address fields must be rendered into the per-turn system
prompt as an "ALREADY COLLECTED — DO NOT RE-ASK" block on both transports
(Twilio's BidirectionalStreamHandler and LiveKit's ConversationOrchestrator),
and must NOT appear at all when nothing has been confirmed yet.

Only prompt-text assembly is asserted (never LLM compliance) — same
boundary convention as tests/voice/test_bracket_tag_prompt_consistency.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler
from tests.voice.test_call_pipeline import _base_handler

MARKER = "ALREADY COLLECTED"


def _capture_twilio_system_prompt():
    captured = []

    async def _spy_stream(prompt=None, system_prompt=None, **kwargs):
        captured.append(system_prompt or "")
        yield "Sure, here is a short reply."

    return captured, _spy_stream


class TestTwilioContactIntakePromptInjection:
    def _run(self, h, spy_stream):
        with patch(
            "app.routers.bidirectional_stream.openai_service.stream_text", new=spy_stream
        ), patch.object(h, "_add_to_transcript", new=AsyncMock()), patch.object(
            h, "_send_in_progress_status", new=AsyncMock()
        ):
            asyncio.run(h.generate_and_stream_response("Hello there", 0.9))

    def test_no_block_when_nothing_confirmed(self):
        h = _base_handler()
        h.call_session.call_metadata = {}
        captured, spy = _capture_twilio_system_prompt()
        self._run(h, spy)

        assert MARKER not in captured[0]

    def test_block_present_when_fields_confirmed(self):
        h = _base_handler()
        h.call_session.call_metadata = {
            "contact_intake": {
                "name": "Adel",
                "name_confident": True,
                "phone": "555-123-4567",
                "phone_confirmed": True,
            }
        }
        captured, spy = _capture_twilio_system_prompt()
        self._run(h, spy)

        sp = captured[0]
        assert MARKER in sp
        assert "Name: CONFIRMED (Adel)" in sp
        assert "Phone: CONFIRMED (555-123-4567)" in sp

    def test_block_present_with_custom_system_prompt_branch(self):
        h = _base_handler()
        h.agent.system_prompt = "Custom instructions for this agent."
        h.call_session.call_metadata = {
            "contact_intake": {"email": "adel@example.com", "email_validated": True}
        }
        captured, spy = _capture_twilio_system_prompt()
        self._run(h, spy)

        sp = captured[0]
        assert MARKER in sp
        assert "Email: CONFIRMED (adel@example.com)" in sp


class TestLiveKitContactIntakePromptInjection:
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

    def test_no_block_when_call_session_falsy(self, monkeypatch):
        # _fake_livekit_handler's default call_session is a falsy
        # _FakeCallSession() — contact intake lookup must be skipped cleanly.
        h = _fake_livekit_handler(tts_slug="google")
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        assert MARKER not in captured[0]

    def test_block_present_when_fields_confirmed(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="google")
        h.call_session = MagicMock()
        h.call_session.call_transcript = []
        h.call_session.call_metadata = {
            "contact_intake": {
                "address": "123 Main Street",
                "address_confirmed": True,
            }
        }
        h.db = None
        h.call_flow = None
        orchestrator = ConversationOrchestrator(h)
        captured: list = []
        self._run(orchestrator, monkeypatch, captured)

        sp = captured[0]
        assert MARKER in sp
        assert "Address: CONFIRMED (123 Main Street)" in sp
