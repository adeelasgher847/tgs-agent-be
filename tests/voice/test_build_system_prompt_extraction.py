"""
Regression test for the ConversationOrchestrator.build_system_prompt()
extraction (behavior-preserving refactor pulling prompt-assembly logic out
of generate_and_stream_response so the Gemini Live native-audio session
start path can reuse it — see VoiceOrchestrator._start_gemini_live_session).

Asserts:
  - build_system_prompt(), called directly, produces the same system_prompt
    generate_and_stream_response's LLM-streaming call site used to build
    inline (captured via the existing llm_service_for_provider stub pattern
    used across tests/voice/*_grounding.py).
  - Minimum content-shape guarantees: grounding text present, KB context
    included when flow.knowledge_base_ids is set, conversation history
    included.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.voice.conversation_orchestrator import (
    ConversationOrchestrator,
    _BUSINESS_FACTS_GROUNDING_TEXT,
)
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler


class TestBuildSystemPromptMatchesStreamingPath:
    def test_build_system_prompt_equals_what_generate_and_stream_response_used(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="google")
        orchestrator = ConversationOrchestrator(h)

        # Build directly.
        direct = asyncio.run(orchestrator.build_system_prompt("Hello there", 0.9))

        # Build via the full streaming path (same as pre-refactor behavior) —
        # capture the system_prompt kwarg passed into stream_text.
        captured: list[str] = []

        async def _stub_stream(**kwargs):
            captured.append(kwargs.get("system_prompt") or "")
            yield "Sure, here is a short reply."

        stub_service = MagicMock()
        stub_service.stream_text = _stub_stream
        monkeypatch.setattr(
            "app.core.agent_runtime.llm_service_for_provider", lambda slug: stub_service
        )
        asyncio.run(orchestrator.generate_and_stream_response("Hello there", 0.9))

        assert captured, "LLM must have been invoked"
        assert direct == captured[0], (
            "build_system_prompt() output diverged from the inline prompt "
            "generate_and_stream_response used to build pre-refactor."
        )


class TestBuildSystemPromptShape:
    def test_contains_grounding_text_and_history(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.call_session = MagicMock()
        h.HISTORY_MAX_MESSAGES = 20  # MagicMock auto-attrs would otherwise shadow getattr()'s default
        h.call_session.call_transcript = [
            {"role": "client", "content": "What are your hours?", "message_type": "speech"},
            {"role": "agent", "content": "We're open 9 to 5.", "message_type": "agent_response"},
        ]
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Great, thanks", 0.9))

        assert _BUSINESS_FACTS_GROUNDING_TEXT in prompt
        assert "What are your hours?" in prompt
        assert "We're open 9 to 5." in prompt

    def test_includes_kb_context_when_flow_has_knowledge_base_ids(self):
        kb_id = uuid.uuid4()
        h = _fake_livekit_handler(tts_slug="google")
        h.db = MagicMock()  # truthy — required for the KB-retrieval branch to run
        flow = MagicMock()
        flow.knowledge_base_ids = [str(kb_id)]
        flow.caller_memory_enabled = False
        h.call_flow = flow
        orchestrator = ConversationOrchestrator(h)

        fake_kb_block = (
            "--- KNOWLEDGE BASE CONTEXT ---\nTEST_KB_FACT\n--- END CONTEXT ---"
        )
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=(fake_kb_block, 12.0)),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            prompt = asyncio.run(
                orchestrator.build_system_prompt("What's your refund policy?", 0.9)
            )

        assert "TEST_KB_FACT" in prompt
        assert "KNOWLEDGE BASE CONTEXT" in prompt

    def test_no_kb_configured_produces_no_kb_block(self):
        h = _fake_livekit_handler(tts_slug="google")
        h.db = MagicMock()
        flow = MagicMock()
        flow.knowledge_base_ids = []
        flow.caller_memory_enabled = False
        h.call_flow = flow
        orchestrator = ConversationOrchestrator(h)

        prompt = asyncio.run(orchestrator.build_system_prompt("Hi", 0.9))

        assert "--- KNOWLEDGE BASE CONTEXT ---" not in prompt
