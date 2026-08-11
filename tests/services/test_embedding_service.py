"""
Tests for app.services.embedding_service.embed_text_for_rag.

Covers the embedding-provider-mismatch fix: when OPENAI_API_KEY is configured,
an OpenAI embedding failure must raise (not silently fall back to Gemini),
since kbchunk.embedding rows are OpenAI-embedded and a Gemini vector would be
a different embedding space entirely — producing semantically meaningless
nearest-neighbor results instead of no results.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.embedding_service import embed_text_for_rag


def test_openai_success_returns_vector():
    fake_resp = MagicMock()
    fake_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = fake_resp

    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch("app.core.openai_client.get_openai_client", return_value=fake_client),
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.RAG_EMBEDDING_MODEL = "text-embedding-3-small"
        result = embed_text_for_rag("hello world")

    assert result == [0.1, 0.2, 0.3]


def test_openai_failure_raises_instead_of_falling_back_to_gemini():
    """When OPENAI_API_KEY is set but the call fails, this must raise — not
    silently switch to Gemini and return a vector in a different embedding
    space than the stored kbchunk.embedding column."""
    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch(
            "app.core.openai_client.get_openai_client",
            side_effect=RuntimeError("OpenAI outage"),
        ),
        patch("app.services.gemini_service.gemini_service") as mock_gemini,
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.GEMINI_API_KEY = "gemini-key"
        mock_settings.RAG_EMBEDDING_MODEL = "text-embedding-3-small"

        with pytest.raises(RuntimeError, match="OpenAI outage"):
            embed_text_for_rag("hello world")

    mock_gemini.embed_text.assert_not_called()


def test_gemini_used_only_when_openai_not_configured_at_all():
    """Gemini remains a valid deliberate-deployment fallback when there is no
    OpenAI key configured — this is a different case from a mid-call OpenAI
    failure and must keep working."""
    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch("app.services.gemini_service.gemini_service") as mock_gemini,
    ):
        mock_settings.OPENAI_API_KEY = ""
        mock_settings.GEMINI_API_KEY = "gemini-key"
        mock_settings.RAG_FALLBACK_EMBEDDING_MODEL = "gemini-embedding-002"
        mock_settings.VECTOR_DIMENSION = 1536
        mock_gemini.embed_text.return_value = [0.4, 0.5]

        result = embed_text_for_rag("hello world")

    assert result == [0.4, 0.5]
    mock_gemini.embed_text.assert_called_once()


def test_no_provider_configured_raises():
    with patch("app.services.embedding_service.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = ""
        mock_settings.GEMINI_API_KEY = ""

        with pytest.raises(RuntimeError, match="No embedding provider available"):
            embed_text_for_rag("hello world")
