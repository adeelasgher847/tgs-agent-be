"""
Regression coverage: Twilio's RAG timeout/failure fallback must NOT inject
an explicit "respond that this information is not available" refusal
directive into the prompt -- confirmed via real production logs and a
frame-level trace of a live call where the agent refused to answer a
question its own knowledge base clearly had the answer for.

Root cause: on asyncio.TimeoutError (RAG_RETRIEVAL_TIMEOUT_SEC is only
0.45s, a real production call timed out on its very first substantive
question), both `_prefetch_rag_context` and the synchronous RAG-build
block in `_build_system_prompt_full` called
`build_rag_context_block_with_trace(user_text="", ...)` to synthesize a
"fallback". That function's empty-input branch returns a block that
explicitly instructs the LLM to refuse ("respond that this information is
not available instead of guessing or inventing details") -- appropriate
for a genuine "no KB entries exist for this query" semantic result, but
actively wrong for a purely technical retrieval failure/timeout, where the
KB may well have the answer. The browser/LiveKit path's equivalent
timeout (kb_retrieval_service.py) just proceeds with an empty context
block and no explicit instruction either way -- these tests confirm
Twilio's fallback now matches that behavior.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler

REFUSAL_PHRASE = "respond that this information is not available"


def _handler_for_rag_prefetch() -> Handler:
    h = object.__new__(Handler)
    h.call_session = MagicMock()
    h.call_session.tenant_id = uuid.uuid4()
    h.agent = MagicMock()
    h.agent.id = uuid.uuid4()
    h.agent.is_inbound_agent = False
    return h


@pytest.mark.asyncio
async def test_prefetch_rag_context_timeout_returns_genuinely_empty_block():
    h = _handler_for_rag_prefetch()

    def _slow_build(**kwargs):
        time.sleep(0.2)
        return "# KNOWLEDGE BASE CONTEXT\nirrelevant, never reached in time", {}

    with patch(
        "app.routers.bidirectional_stream.settings.RAG_RETRIEVAL_TIMEOUT_SEC", 0.01
    ), patch(
        "app.routers.bidirectional_stream.build_rag_context_block_with_trace",
        side_effect=_slow_build,
    ):
        block, trace = await h._prefetch_rag_context("What are your business hours?")

    assert block == ""
    assert REFUSAL_PHRASE not in block
    assert trace["status"] == "timeout"
    assert trace["timeout"] is True


@pytest.mark.asyncio
async def test_prefetch_rag_context_exception_returns_genuinely_empty_block():
    h = _handler_for_rag_prefetch()

    def _raises(**kwargs):
        raise RuntimeError("pgvector connection reset")

    with patch(
        "app.routers.bidirectional_stream.build_rag_context_block_with_trace",
        side_effect=_raises,
    ):
        block, trace = await h._prefetch_rag_context("Do you offer AC repair?")

    assert block == ""
    assert REFUSAL_PHRASE not in block
    assert trace["status"] == "failure"
    assert "pgvector connection reset" in trace["error"]


@pytest.mark.asyncio
async def test_prefetch_rag_context_success_still_returns_real_context():
    """Guardrail: the fail-open fix must not accidentally swallow a
    successful, fast retrieval too."""
    h = _handler_for_rag_prefetch()

    expected = ("# KNOWLEDGE BASE CONTEXT\nHours are 9-5.", {"status": "high_confidence"})

    with patch(
        "app.routers.bidirectional_stream.build_rag_context_block_with_trace",
        return_value=expected,
    ):
        result = await h._prefetch_rag_context("What are your hours?")

    assert result == expected
