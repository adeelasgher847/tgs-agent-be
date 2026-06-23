"""
Tests for the real-time KB retrieval layer (kb_retrieval_service).

Covers:
  - retrieve_kb_context_for_turn: KB attached to flow → context block in output
  - retrieve_kb_context_for_turn: Redis cache hit on second call (embedding not re-fetched)
  - retrieve_kb_context_for_turn: partial KB failure logged, other KBs still returned
  - retrieve_kb_context_for_turn: no kb_ids → empty string returned
  - format_kb_context_block: correct spec format
  - GET /{kb_id}/search: returns [{content, score, metadata}]
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.kb_retrieval_service import (
    RetrievedChunk,
    format_kb_context_block,
    retrieve_kb_context_for_turn,
)


FAKE_EMBEDDING = [0.1] * 1536


# ── format_kb_context_block ───────────────────────────────────────────────────

def test_format_kb_context_block_spec_format():
    chunks = [
        RetrievedChunk(content="First chunk text", score=0.9, metadata={}),
        RetrievedChunk(content="Second chunk text", score=0.8, metadata={}),
    ]
    block = format_kb_context_block(chunks)
    assert block.startswith("--- KNOWLEDGE BASE CONTEXT ---")
    assert "First chunk text" in block
    assert "Second chunk text" in block
    assert block.endswith("--- END CONTEXT ---")
    assert "\n---\n" in block


def test_format_kb_context_block_empty():
    assert format_kb_context_block([]) == ""


def test_format_kb_context_block_single_chunk():
    chunks = [RetrievedChunk(content="Only chunk", score=0.95, metadata={})]
    block = format_kb_context_block(chunks)
    lines = block.splitlines()
    assert lines[0] == "--- KNOWLEDGE BASE CONTEXT ---"
    assert "Only chunk" in block
    assert lines[-1] == "--- END CONTEXT ---"


# ── Helper: build a mock DB that returns the given chunk rows ─────────────────

def _fake_db_rows(chunks: list[dict]):
    rows = []
    for c in chunks:
        row = MagicMock()
        row.content = c["content"]
        row.score = c["score"]
        row.chunk_metadata = c.get("metadata", {})
        rows.append(row)
    result = MagicMock()
    result.fetchall.return_value = rows
    db = MagicMock()
    db.execute.return_value = result
    return db


# ── retrieve_kb_context_for_turn ──────────────────────────────────────────────

def test_retrieve_returns_block_when_kb_attached():
    """KB attached → context block contains the retrieved chunk text."""
    kb_id = uuid.uuid4()

    async def fake_query(kb_id, vec_str, top_k):
        return [RetrievedChunk(content="Company refund policy is 30 days.", score=0.92, metadata={})]

    with (
        patch(
            "app.services.kb_retrieval_service._get_embedding_cached",
            new=AsyncMock(return_value=FAKE_EMBEDDING),
        ),
        patch(
            "app.services.kb_retrieval_service._query_single_kb",
            side_effect=fake_query,
        ),
    ):
        context_block, latency_ms = asyncio.run(
            retrieve_kb_context_for_turn(
                transcript="What is your refund policy?",
                kb_ids=[kb_id],
                redis_client=None,
            )
        )

    assert "--- KNOWLEDGE BASE CONTEXT ---" in context_block
    assert "Company refund policy is 30 days." in context_block
    assert "--- END CONTEXT ---" in context_block
    assert latency_ms >= 0


def test_retrieve_redis_cache_hit_skips_embedding():
    """Cache hit: embedding function must NOT be called on a cache hit."""
    kb_id = uuid.uuid4()
    cached_block = "--- KNOWLEDGE BASE CONTEXT ---\nCached chunk\n--- END CONTEXT ---"
    transcript = "What is your return policy?"
    kb_ids = [kb_id]

    import hashlib

    cache_key = (
        "kb:ctx:"
        + hashlib.sha256(
            (transcript + ":" + ":".join(sorted(str(k) for k in kb_ids))).encode()
        ).hexdigest()
    )

    redis_client = AsyncMock()
    redis_client.get = AsyncMock(
        side_effect=lambda key: json.dumps(cached_block) if key == cache_key else None
    )

    embed_mock = AsyncMock(return_value=FAKE_EMBEDDING)

    with patch("app.services.kb_retrieval_service._get_embedding_cached", new=embed_mock):
        context_block, _ = asyncio.run(
            retrieve_kb_context_for_turn(
                transcript=transcript,
                kb_ids=kb_ids,
                redis_client=redis_client,
            )
        )

    assert context_block == cached_block
    embed_mock.assert_not_called()


def test_retrieve_no_kb_ids_returns_empty():
    context_block, latency_ms = asyncio.run(
        retrieve_kb_context_for_turn(
            transcript="Hello",
            kb_ids=[],
            redis_client=None,
        )
    )
    assert context_block == ""
    assert latency_ms == 0.0


def test_retrieve_partial_failure_continues():
    """If one KB errors, the other KB's results are still returned."""
    good_kb_id = uuid.uuid4()
    bad_kb_id = uuid.uuid4()

    async def fake_query(kb_id, vec_str, top_k):
        if kb_id == bad_kb_id:
            raise RuntimeError("DB failure")
        return [RetrievedChunk(content="Good KB chunk.", score=0.88, metadata={})]

    with (
        patch(
            "app.services.kb_retrieval_service._get_embedding_cached",
            new=AsyncMock(return_value=FAKE_EMBEDDING),
        ),
        patch(
            "app.services.kb_retrieval_service._query_single_kb",
            side_effect=fake_query,
        ),
    ):
        context_block, _ = asyncio.run(
            retrieve_kb_context_for_turn(
                transcript="Test",
                kb_ids=[good_kb_id, bad_kb_id],
                redis_client=None,
            )
        )

    assert "Good KB chunk." in context_block


def test_retrieve_embedding_failure_returns_empty():
    """Embedding failure → empty string, call is not blocked."""
    with patch(
        "app.services.kb_retrieval_service._get_embedding_cached",
        new=AsyncMock(side_effect=RuntimeError("OpenAI down")),
    ):
        context_block, _ = asyncio.run(
            retrieve_kb_context_for_turn(
                transcript="Any query",
                kb_ids=[uuid.uuid4()],
                redis_client=None,
            )
        )
    assert context_block == ""


def test_retrieve_top5_merge_across_kbs():
    """Results from multiple KBs are merged and the global top 5 are kept."""
    kb_a = uuid.uuid4()
    kb_b = uuid.uuid4()

    chunks_a = [RetrievedChunk(content=f"A{i}", score=0.50 + i * 0.05, metadata={}) for i in range(3)]
    chunks_b = [RetrievedChunk(content=f"B{i}", score=0.70 + i * 0.02, metadata={}) for i in range(4)]

    async def fake_query(kb_id, vec_str, top_k):
        return chunks_a if kb_id == kb_a else chunks_b

    with (
        patch(
            "app.services.kb_retrieval_service._get_embedding_cached",
            new=AsyncMock(return_value=FAKE_EMBEDDING),
        ),
        patch(
            "app.services.kb_retrieval_service._query_single_kb",
            side_effect=fake_query,
        ),
    ):
        context_block, _ = asyncio.run(
            retrieve_kb_context_for_turn(
                transcript="query",
                kb_ids=[kb_a, kb_b],
                redis_client=None,
            )
        )

    # Top 5 should come from merged + sorted (chunks_b scores are higher)
    all_chunks = sorted(chunks_a + chunks_b, key=lambda c: c.score, reverse=True)
    for chunk in all_chunks[:5]:
        assert chunk.content in context_block


# ── GET /{kb_id}/search endpoint ─────────────────────────────────────────────

@pytest.fixture()
def workspace_id(db) -> uuid.UUID:
    from app.models.tenant import Tenant

    return db.query(Tenant).first().id


def _mock_tenant(wid: uuid.UUID):
    user = MagicMock()
    user.current_tenant_id = wid
    return user


@contextmanager
def _auth_ctx(wid: uuid.UUID):
    from app.api.deps import require_tenant, require_readonly_or_api_key, require_config_or_api_key
    import app.middleware.api_key_middleware as auth_mw
    from app.main import app as _app

    mock_user = _mock_tenant(wid)
    overridden = [require_tenant, require_readonly_or_api_key, require_config_or_api_key]
    for dep in overridden:
        _app.dependency_overrides[dep] = lambda: mock_user
    with patch.object(auth_mw, "_try_jwt_auth", new=AsyncMock(return_value=True)):
        try:
            yield
        finally:
            for dep in overridden:
                _app.dependency_overrides.pop(dep, None)


def test_search_endpoint_returns_results(client, db, workspace_id):
    """GET /kb/{kb_id}/search returns [{content, score, metadata}]."""
    from app.models.knowledge_base_document import KnowledgeBase
    from app.api.deps import get_db
    from app.main import app as _app

    kb = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="Search Test KB")
    db.add(kb)
    db.commit()

    # Build a mock row for the pgvector result
    mock_row = MagicMock()
    mock_row.content = "Product warranty is 12 months."
    mock_row.score = 0.91
    mock_row.chunk_metadata = {"source": "manual.pdf"}
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [mock_row]

    # Wrap the real session so ORM queries still work but raw SQL is intercepted
    class _PgvectorMockSession:
        def __init__(self, real_db):
            self._real = real_db

        def query(self, *a, **kw):
            return self._real.query(*a, **kw)

        def execute(self, stmt, params=None, **kw):
            return mock_result

        def __getattr__(self, name):
            return getattr(self._real, name)

    # Save the existing test override so we can restore it afterwards
    from tests.conftest import override_get_db  # type: ignore[import]

    original_override = _app.dependency_overrides.get(get_db, override_get_db)
    _app.dependency_overrides[get_db] = lambda: _PgvectorMockSession(db)

    try:
        with _auth_ctx(workspace_id):
            with (
                patch("app.routers.knowledge_base.settings") as mock_settings,
                patch(
                    "app.routers.knowledge_base.embed_text_for_rag",
                    return_value=FAKE_EMBEDDING,
                ),
            ):
                mock_settings.OPENAI_API_KEY = "test-key"
                mock_settings.RAG_TOP_K = 5
                resp = client.get(f"/api/v1/kb/{kb.id}/search?q=warranty&limit=5")
    finally:
        _app.dependency_overrides[get_db] = original_override

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_endpoint_404_unknown_kb(client, db, workspace_id):
    with _auth_ctx(workspace_id):
        with patch("app.routers.knowledge_base.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            resp = client.get(f"/api/v1/kb/{uuid.uuid4()}/search?q=test")
    assert resp.status_code == 404


def test_search_endpoint_400_no_openai_key(client, db, workspace_id):
    fake_kb_id = uuid.uuid4()

    with _auth_ctx(workspace_id):
        with (
            patch("app.routers.knowledge_base.settings") as mock_settings,
            # Bypass the 404 check so we reach the OPENAI_API_KEY guard
            patch(
                "app.routers.knowledge_base._get_kb_or_404",
                return_value=MagicMock(),
            ),
        ):
            mock_settings.OPENAI_API_KEY = ""
            resp = client.get(f"/api/v1/kb/{fake_kb_id}/search?q=test")
    assert resp.status_code == 400
