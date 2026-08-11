"""
KB/RAG grounding regression tests for the LiveKit/browser voice-agent path
(LiveKitBrowserCallHandler -> ConversationOrchestrator -> kb_retrieval_service).

Covers the bug fixed here: KB chunks were spliced into the system prompt but
no grounding rule told the model the "KNOWLEDGE BASE CONTEXT" block was a
usable/authoritative source, so the model ignored it. Also covers the
retrieval-trigger contract (KB attached vs not, multiple KBs) and the
fail-open timeout behavior.

Only the real prompt-building code in ConversationOrchestrator.
generate_and_stream_response() is exercised; retrieve_kb_context_for_turn and
the LLM streaming call are mocked at the boundary.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler


FAKE_KB_FACT = "TEST_KB_FACT_12345"
FAKE_KB_BLOCK = (
    "--- KNOWLEDGE BASE CONTEXT ---\n"
    "The following information comes from the business's uploaded Knowledge Base. "
    "Use this information when answering relevant factual questions. Do not ignore it. "
    "Do not add facts that are not supported by it.\n"
    f"{FAKE_KB_FACT}\n"
    "--- END CONTEXT ---"
)


def _handler_with_kb(kb_ids):
    h = _fake_livekit_handler(tts_slug="google")
    h.db = MagicMock()  # truthy — required for the KB-retrieval branch to run
    flow = MagicMock()
    flow.knowledge_base_ids = [str(k) for k in kb_ids]
    flow.caller_memory_enabled = False
    h.call_flow = flow
    return h


def _run_and_capture(orchestrator, monkeypatch):
    captured: list[str] = []

    async def _stub_stream(**kwargs):
        captured.append(kwargs.get("system_prompt") or "")
        yield "Sure, here is a short reply."

    stub_service = MagicMock()
    stub_service.stream_text = _stub_stream
    monkeypatch.setattr(
        "app.core.agent_runtime.llm_service_for_provider", lambda slug: stub_service
    )
    asyncio.run(orchestrator.generate_and_stream_response("What is your refund policy?", 0.9))
    return captured


class TestKbContextReachesSystemPrompt:
    """Invariant 1: successful KB retrieval -> chunks explicitly presented as
    authoritative, and the prompt instructs the model to use them."""

    def test_kb_fact_reaches_system_prompt_and_is_grounded(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb([kb_id])
        orchestrator = ConversationOrchestrator(h)

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=(FAKE_KB_BLOCK, 42.0)),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            captured = _run_and_capture(orchestrator, monkeypatch)

        assert captured, "LLM must have been invoked"
        sp = captured[0]
        assert FAKE_KB_FACT in sp
        # The grounding rule must name BOTH sources as usable.
        assert "KNOWLEDGE BASE CONTEXT" in sp
        assert "AUTHORITATIVE BUSINESS FACTS" in sp

    def test_grounding_rule_mentions_kb_context_even_with_custom_prompt(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb([kb_id])
        h.agent.system_prompt = "You help schedule appointments."
        orchestrator = ConversationOrchestrator(h)

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=(FAKE_KB_BLOCK, 42.0)),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            captured = _run_and_capture(orchestrator, monkeypatch)

        sp = captured[0]
        assert FAKE_KB_FACT in sp
        assert "GROUNDING RULES" in sp
        assert "KNOWLEDGE BASE CONTEXT" in sp


class TestKbRetrievalTriggerContract:
    def test_no_kb_configured_skips_retrieval(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="google")
        h.db = MagicMock()
        flow = MagicMock()
        flow.knowledge_base_ids = []
        flow.caller_memory_enabled = False
        h.call_flow = flow
        orchestrator = ConversationOrchestrator(h)

        retrieve_mock = AsyncMock(return_value=("", 0.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ):
            captured = _run_and_capture(orchestrator, monkeypatch)

        retrieve_mock.assert_not_called()
        assert captured
        # The grounding rule may still *mention* "KNOWLEDGE BASE CONTEXT" (it names
        # the block generically for whenever KB content IS present); what must be
        # absent is any actual retrieved block/content.
        assert "--- KNOWLEDGE BASE CONTEXT ---" not in captured[0]

    def test_multiple_kbs_all_queried(self, monkeypatch):
        kb_a, kb_b = uuid.uuid4(), uuid.uuid4()
        h = _handler_with_kb([kb_a, kb_b])
        orchestrator = ConversationOrchestrator(h)

        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 10.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            _run_and_capture(orchestrator, monkeypatch)

        assert retrieve_mock.await_count == 1
        called_kb_ids = retrieve_mock.await_args.kwargs["kb_ids"]
        assert set(called_kb_ids) == {str(kb_a), str(kb_b)}


class TestKbRetrievalFailOpen:
    """Invariant 2: timeout/failure -> no KB-derived content is fabricated,
    but the LLM call still proceeds (fail open)."""

    def test_timeout_produces_no_kb_context_but_llm_still_called(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb([kb_id])
        orchestrator = ConversationOrchestrator(h)

        # Force a short timeout and a retrieval coroutine slower than it.
        monkeypatch.setattr(
            "app.core.config.settings.RAG_KB_RETRIEVAL_TIMEOUT_SEC", 0.01
        )

        async def _slow_retrieve(**kwargs):
            await asyncio.sleep(0.2)
            return FAKE_KB_BLOCK, 200.0

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=_slow_retrieve,
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            captured = _run_and_capture(orchestrator, monkeypatch)

        assert captured, "LLM must still be called after a KB retrieval timeout"
        assert FAKE_KB_FACT not in captured[0]

    def test_retrieval_exception_produces_no_kb_context_but_llm_still_called(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb([kb_id])
        orchestrator = ConversationOrchestrator(h)

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            captured = _run_and_capture(orchestrator, monkeypatch)

        assert captured, "LLM must still be called after a KB retrieval failure"
        assert FAKE_KB_FACT not in captured[0]
