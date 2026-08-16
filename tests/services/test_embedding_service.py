"""
Tests for app.services.embedding_service.embed_text_for_rag.

Covers the embedding-provider-mismatch fix: when OPENAI_API_KEY is configured,
an OpenAI embedding failure must raise (not silently fall back to Gemini),
since kbchunk.embedding rows are OpenAI-embedded and a Gemini vector would be
a different embedding space entirely — producing semantically meaningless
nearest-neighbor results instead of no results.

Also covers the cold-start-latency fix: _embed_openai_ada002 must route
through openai_service's cached-by-api-key client (OpenAIService.get_client)
rather than constructing a brand-new OpenAI client (and therefore a brand-new
httpx connection pool) on every call, plus the fire-and-forget warm_embedding_client
startup hook.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.embedding_service import embed_text_for_rag, warm_embedding_client


def test_openai_success_returns_vector():
    fake_resp = MagicMock()
    fake_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = fake_resp

    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch("app.services.openai_service.openai_service.get_client", return_value=fake_client),
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
            "app.services.openai_service.openai_service.get_client",
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


def test_embed_reuses_cached_client_across_calls():
    """The whole point of the cold-start fix: two embedding calls must reuse
    the SAME client instance (i.e. the same underlying httpx connection pool)
    rather than each constructing a fresh OpenAI client. This is what lets a
    warm call reuse an already-established TCP/TLS connection instead of
    paying a fresh handshake every single time."""
    from app.services.openai_service import openai_service

    fake_resp = MagicMock()
    fake_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = fake_resp

    # Reset the singleton's client cache so this test is order-independent.
    openai_service._clients = {}

    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch("app.services.openai_service.settings") as mock_openai_settings,
        patch("app.services.openai_service.get_openai_client", return_value=fake_client) as mock_factory,
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.RAG_EMBEDDING_MODEL = "text-embedding-3-small"
        mock_openai_settings.OPENAI_API_KEY = "test-key"

        embed_text_for_rag("first call")
        embed_text_for_rag("second call")

    # The underlying client factory must only be invoked once — the second
    # call must be served from openai_service's cache, not a fresh client.
    mock_factory.assert_called_once()
    assert fake_client.embeddings.create.call_count == 2


def test_warm_embedding_client_swallows_errors_and_never_raises():
    """warm_embedding_client is scheduled as a fire-and-forget background task
    from app startup — it must never raise, even if the embedding call fails,
    so a startup-time exception can never crash the worker process."""
    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch(
            "app.services.embedding_service.embed_text_for_rag",
            side_effect=RuntimeError("boom"),
        ),
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        asyncio.run(warm_embedding_client())  # must not raise


def test_warm_embedding_client_noop_without_openai_key():
    """No OPENAI_API_KEY configured (e.g. Gemini-only or on-prem deployment
    without RAG) → warm-up is a no-op rather than raising or performing an
    unnecessary call."""
    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch("app.services.embedding_service.embed_text_for_rag") as mock_embed,
    ):
        mock_settings.OPENAI_API_KEY = ""
        asyncio.run(warm_embedding_client())

    mock_embed.assert_not_called()


def test_warm_embedding_client_calls_embed_with_openai_key():
    with (
        patch("app.services.embedding_service.settings") as mock_settings,
        patch("app.services.embedding_service.embed_text_for_rag") as mock_embed,
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        asyncio.run(warm_embedding_client())

    mock_embed.assert_called_once()


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
