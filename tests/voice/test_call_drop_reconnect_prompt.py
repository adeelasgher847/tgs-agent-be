"""Coverage for F-11 (call-drop reconnect recognition) wiring added to
ConversationOrchestrator.build_system_prompt() and
generate_and_stream_response()'s is_greeting branch in
app/voice/conversation_orchestrator.py.

Follows the established `_fake_livekit_handler` fixture pattern from
tests/voice/test_bracket_tag_prompt_consistency.py, matching the sibling
F-08 injection tests in tests/voice/test_system_webhook_prompt_injection.py.

The router-side wiring (app/routers/voice.py::handle_incoming_call setting
call_metadata["is_reconnect"]/["reconnect_from_session_id"] +
parent_call_id) is covered separately in
tests/routers/test_voice_dynamic_inbound_routing.py.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler

RECONNECT_MARKER = "RECONNECTING CALLER (CALL DROP)"
RECONNECT_GREETING = (
    "Welcome back! Looks like we got cut off there — let's "
    "pick up right where we left off."
)


class TestBuildSystemPromptReconnectInstruction:
    def test_is_reconnect_true_adds_instruction(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={"is_reconnect": True},
            tenant_id=uuid.uuid4(),
        )
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert RECONNECT_MARKER in prompt

    def test_is_reconnect_absent_no_instruction(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={},
            tenant_id=uuid.uuid4(),
        )
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert RECONNECT_MARKER not in prompt

    def test_is_reconnect_false_no_instruction(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={"is_reconnect": False},
            tenant_id=uuid.uuid4(),
        )
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert RECONNECT_MARKER not in prompt


class TestBuildSystemPromptReconnectTranscriptSnippet:
    def test_dropped_session_transcript_included_as_snippet(self):
        dropped_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        h = _fake_livekit_handler(tts_slug="google")
        h.db = MagicMock()  # truthy — required for the DB-lookup branch to run
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={
                "is_reconnect": True,
                "reconnect_from_session_id": str(dropped_id),
            },
            tenant_id=tenant_id,
        )
        orchestrator = ConversationOrchestrator(h)

        dropped_session = SimpleNamespace(
            call_transcript=[
                {"role": "client", "content": "I need to reschedule my appointment"},
                {"role": "agent", "content": "Sure, what day works for you?"},
            ]
        )

        with patch(
            "app.services.call_session_service.call_session_service"
            ".get_call_session_by_id_and_tenant",
            return_value=dropped_session,
        ) as mock_lookup:
            prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        mock_lookup.assert_called_once_with(h.db, dropped_id, tenant_id)
        assert RECONNECT_MARKER in prompt
        assert "reschedule my appointment" in prompt
        assert "what day works for you" in prompt

    def test_dropped_session_lookup_failure_fails_open(self):
        """DB lookup raising must not break prompt building — base reconnect
        instruction still present, just without the transcript snippet."""
        dropped_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        h = _fake_livekit_handler(tts_slug="google")
        h.db = MagicMock()
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={
                "is_reconnect": True,
                "reconnect_from_session_id": str(dropped_id),
            },
            tenant_id=tenant_id,
        )
        orchestrator = ConversationOrchestrator(h)

        with patch(
            "app.services.call_session_service.call_session_service"
            ".get_call_session_by_id_and_tenant",
            side_effect=RuntimeError("simulated DB error"),
        ):
            prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert RECONNECT_MARKER in prompt
        assert "Context from the dropped call" not in prompt

    def test_no_dropped_session_found_no_snippet(self):
        dropped_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        h = _fake_livekit_handler(tts_slug="google")
        h.db = MagicMock()
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={
                "is_reconnect": True,
                "reconnect_from_session_id": str(dropped_id),
            },
            tenant_id=tenant_id,
        )
        orchestrator = ConversationOrchestrator(h)

        with patch(
            "app.services.call_session_service.call_session_service"
            ".get_call_session_by_id_and_tenant",
            return_value=None,
        ):
            prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert RECONNECT_MARKER in prompt
        assert "Context from the dropped call" not in prompt


class TestReconnectGreetingOverride:
    def _run_greeting(self, h) -> str:
        orchestrator = ConversationOrchestrator(h)
        asyncio.run(
            orchestrator.generate_and_stream_response("", 0.9, is_greeting=True)
        )
        assert h._tts_pipeline.queue_tts.await_count == 1
        return h._tts_pipeline.queue_tts.await_args.args[0]["text"]

    def test_reconnect_true_overrides_agent_greeting(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.agent.greeting_message = "Hi there, thanks for calling!"
        h._voice_orchestrator = None  # bypass Gemini/OpenAI Realtime hand-off branches
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={"is_reconnect": True},
            tenant_id=uuid.uuid4(),
        )

        greeting = self._run_greeting(h)

        assert greeting == RECONNECT_GREETING

    def test_reconnect_absent_uses_agent_greeting(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.agent.greeting_message = "Hi there, thanks for calling!"
        h._voice_orchestrator = None
        h.call_session = SimpleNamespace(
            call_transcript=None,
            call_metadata={},
            tenant_id=uuid.uuid4(),
        )

        greeting = self._run_greeting(h)

        assert greeting == "Hi there, thanks for calling!"
        assert greeting != RECONNECT_GREETING
