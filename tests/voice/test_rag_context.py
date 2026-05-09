import uuid
from unittest.mock import MagicMock

import pytest

from app.voice import rag_context as rag_context_module
from app.voice.rag_context import build_rag_context_block
from app.services.rag_service import RagChunkDTO


def test_no_user_text_returns_no_kb_block(monkeypatch):
    # Ensure code path isn't blocked by missing pinecone config.
    monkeypatch.setattr(rag_context_module.settings, "PINECONE_API_KEY", "dummy", raising=False)

    out = build_rag_context_block(
        user_text="   ",
        tenant_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
    )
    assert "No relevant knowledge base entries were found for this query." in out


def test_missing_tenant_returns_no_kb_block(monkeypatch):
    monkeypatch.setattr(rag_context_module.settings, "PINECONE_API_KEY", "dummy", raising=False)

    out = build_rag_context_block(
        user_text="What are your hours?",
        tenant_id=None,
        agent_id=uuid.uuid4(),
    )
    assert "No relevant knowledge base entries were found for this query." in out


def test_low_confidence_chunks_filtered_out(monkeypatch):
    monkeypatch.setattr(rag_context_module.settings, "PINECONE_API_KEY", "dummy", raising=False)

    threshold = getattr(rag_context_module.settings, "RAG_SCORE_THRESHOLD", 0.4)
    low_score = max(0.0, threshold - 0.05)

    def _mock_retrieve(*args, **kwargs):
        return [
            RagChunkDTO(
                text="Hours are 9am to 5pm.",
                source_title="Test FAQ",
                source_ref="test-faq",
                score=low_score,
            )
        ]

    monkeypatch.setattr(rag_context_module.rag_service, "retrieve", _mock_retrieve)

    out = build_rag_context_block(
        user_text="What are your support hours?",
        tenant_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
    )
    assert "No relevant knowledge base entries were found for this query." in out


def test_high_confidence_chunks_render_context(monkeypatch):
    monkeypatch.setattr(rag_context_module.settings, "PINECONE_API_KEY", "dummy", raising=False)

    threshold = getattr(rag_context_module.settings, "RAG_SCORE_THRESHOLD", 0.4)
    high_score = threshold + 0.1

    def _mock_retrieve(*args, **kwargs):
        return [
            RagChunkDTO(
                text="Hours are 9am to 5pm, Monday to Friday.",
                source_title="Support FAQ",
                source_ref="support-faq",
                score=high_score,
            )
        ]

    monkeypatch.setattr(rag_context_module.rag_service, "retrieve", _mock_retrieve)

    out = build_rag_context_block(
        user_text="What are your support hours?",
        tenant_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
    )

    assert "You have access to company knowledge retrieved" in out
    assert "[1]" in out  # chunk numbering / citation ids
    assert "Hours are 9am to 5pm" in out
    assert "cite the chunk number like [1]" in out

