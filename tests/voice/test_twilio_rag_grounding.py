"""
KB/RAG grounding regression tests for the Twilio/phone voice-agent path
(BidirectionalStreamHandler.generate_and_stream_response -> rag_context.py ->
RagService.retrieve() -> pgvector).

Mirrors tests/voice/test_livekit_kb_grounding.py (the browser/LiveKit path's
equivalent regression suite) but exercises the real, separate Twilio
implementation. The two paths are deliberately NOT unified — see
bidirectional_stream.py / conversation_orchestrator.py module docstrings —
so this suite intentionally duplicates the invariant rather than sharing
fixtures/helpers across the two pipelines.

Only the real prompt-building code in
BidirectionalStreamHandler.generate_and_stream_response() is exercised;
RagService.retrieve() and the LLM streaming call are mocked at the boundary.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.services.rag_service import RagChunkDTO
from app.voice import rag_context as rag_context_module
from tests.voice.test_call_pipeline import _base_handler


TWILIO_KB_TEST_FACT = "TWILIO_KB_TEST_FACT_98765"


def _mock_retrieve_high_confidence(*args, **kwargs):
    return [
        RagChunkDTO(
            text=TWILIO_KB_TEST_FACT,
            source_title="Test FAQ",
            source_ref="test-faq",
            score=0.95,
        )
    ]


def _run_and_capture_system_prompt(h, user_text: str = "What is your pricing for that service?") -> list[str]:
    captured: list[str] = []

    async def _spy_stream(prompt=None, system_prompt=None, messages=None, **kwargs):
        captured.append(system_prompt or "")
        yield "Sure, here is a short reply."

    with patch("app.routers.bidirectional_stream.openai_service.stream_text", new=_spy_stream), \
         patch.object(h, "_add_to_transcript", new=AsyncMock()), \
         patch.object(h, "_send_in_progress_status", new=AsyncMock()):
        asyncio.run(h.generate_and_stream_response(user_text, 0.9))

    return captured


class TestTwilioRagContextReachesSystemPrompt:
    """Invariant: a high-confidence retrieved chunk must demonstrably flow
    into the actual `messages`/system_prompt payload sent toward the LLM."""

    def test_rag_fact_reaches_system_prompt_base_prompt(self, monkeypatch):
        h = _base_handler()
        h.agent.system_prompt = None  # force base_prompt branch

        monkeypatch.setattr(
            rag_context_module.rag_service, "retrieve", _mock_retrieve_high_confidence
        )

        captured = _run_and_capture_system_prompt(h)

        assert captured, "LLM must have been invoked"
        sp = captured[0]
        assert TWILIO_KB_TEST_FACT in sp
        assert "KNOWLEDGE BASE CONTEXT" in sp
        # The grounding rule must name BOTH the KB context and business facts
        # as authoritative — not just the latter (the bug fixed on the
        # browser path was this rule silently ignoring KB context).
        assert "AUTHORITATIVE BUSINESS FACTS and, when present in this prompt, KNOWLEDGE BASE CONTEXT" in sp
        assert "AUTHORITATIVE BUSINESS FACTS" in sp

    def test_rag_fact_reaches_system_prompt_with_custom_agent_prompt(self, monkeypatch):
        h = _base_handler()
        h.agent.system_prompt = "You help schedule appointments."

        monkeypatch.setattr(
            rag_context_module.rag_service, "retrieve", _mock_retrieve_high_confidence
        )

        captured = _run_and_capture_system_prompt(h)

        sp = captured[0]
        assert TWILIO_KB_TEST_FACT in sp
        assert "GROUNDING RULES" in sp
        assert "KNOWLEDGE BASE CONTEXT" in sp

    def test_low_confidence_chunk_does_not_reach_system_prompt(self, monkeypatch):
        h = _base_handler()
        h.agent.system_prompt = None

        def _mock_retrieve_low_confidence(*args, **kwargs):
            return [
                RagChunkDTO(
                    text=TWILIO_KB_TEST_FACT,
                    source_title="Test FAQ",
                    source_ref="test-faq",
                    score=0.1,  # below RAG_SCORE_THRESHOLD (0.4 default)
                )
            ]

        monkeypatch.setattr(
            rag_context_module.rag_service, "retrieve", _mock_retrieve_low_confidence
        )

        captured = _run_and_capture_system_prompt(h)

        assert captured, "LLM must still be called even with no qualifying KB context"
        assert TWILIO_KB_TEST_FACT not in captured[0]

    def test_retrieval_exception_fails_open_llm_still_called(self, monkeypatch):
        h = _base_handler()
        h.agent.system_prompt = None

        def _boom(*args, **kwargs):
            raise RuntimeError("pgvector unavailable")

        monkeypatch.setattr(rag_context_module.rag_service, "retrieve", _boom)

        captured = _run_and_capture_system_prompt(h)

        assert captured, "LLM must still be called after a RAG retrieval failure"
        assert TWILIO_KB_TEST_FACT not in captured[0]
