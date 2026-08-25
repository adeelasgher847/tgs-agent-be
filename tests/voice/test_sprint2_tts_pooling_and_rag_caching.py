"""
Sprint 2 Tests:
  1. ElevenLabs AsyncClient Connection Pooling & Lifecycle
  2. Robust RAG Cache Keys & Explicit KB Revision Invalidation Invariants
"""
from __future__ import annotations

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.elevenlabs_service import ElevenLabsService
from app.services.kb_retrieval_service import (
    build_embedding_cache_key,
    build_retrieval_cache_key,
    get_kb_revision,
    invalidate_kb_cache,
    retrieve_kb_context_for_turn,
)


# ── Step 4 Tests: ElevenLabs Connection Pooling ───────────────────────────────

@pytest.mark.asyncio
async def test_elevenlabs_async_client_reuse_on_same_loop():
    """Verify that multiple TTS requests on the same event loop reuse the exact same AsyncClient."""
    service = ElevenLabsService()
    try:
        client1 = service._get_async_client(timeout_sec=10.0)
        client2 = service._get_async_client(timeout_sec=10.0)
        assert client1 is client2
        assert not client1.is_closed
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_elevenlabs_async_client_lifecycle_aclose():
    """Verify that aclose() closes all cached AsyncClients and empties the pool."""
    service = ElevenLabsService()
    client = service._get_async_client(timeout_sec=10.0)
    assert not client.is_closed
    assert len(service._async_clients) == 1

    await service.aclose()
    assert client.is_closed
    assert len(service._async_clients) == 0


@pytest.mark.asyncio
async def test_elevenlabs_streaming_cancellation_releases_cleanly():
    """Verify that cancelling an in-flight stream does not corrupt the connection pool."""
    service = ElevenLabsService()
    try:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def _mock_aiter_bytes(chunk_size):
            yield b"chunk_1"
            await asyncio.sleep(0.5)  # Simulate slow network chunk
            yield b"chunk_2"

        mock_response.aiter_bytes = _mock_aiter_bytes

        # Mock the stream context manager on the client
        client = service._get_async_client()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)
        client.stream = MagicMock(return_value=mock_stream_ctx)

        async def _consume():
            chunks = []
            async for chunk in service.async_stream_text_to_speech(
                text="Hello world",
                voice_id="voice-123",
                api_key_override="test-key",
            ):
                chunks.append(chunk)
                if len(chunks) == 1:
                    break
            return chunks

        res = await asyncio.wait_for(_consume(), timeout=1.0)
        assert res == [b"chunk_1"]
        # Client remains open and reusable in pool
        assert not client.is_closed
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_elevenlabs_client_and_transport_reuse_across_sequential_requests():
    """Verify that multiple sequential TTS streaming calls execute through ElevenLabsService._get_async_client()."""
    import httpx

    service = ElevenLabsService()
    init_call_count = 0
    created_clients = []

    real_async_client_cls = httpx.AsyncClient

    def _mock_async_client_factory(*args, **kwargs):
        nonlocal init_call_count
        init_call_count += 1
        # Create real AsyncClient with mock transport so streams succeed
        def _handle_request(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"audio_chunk_data")

        client = real_async_client_cls(transport=httpx.MockTransport(_handle_request))
        created_clients.append(client)
        return client

    try:
        with patch("httpx.AsyncClient", side_effect=_mock_async_client_factory):
            # First request: _get_async_client() must instantiate a client
            chunks1 = []
            async for chunk in service.async_stream_text_to_speech(
                text="First sentence.", voice_id="v1", api_key_override="key1"
            ):
                chunks1.append(chunk)

            # Second request: _get_async_client() must reuse the existing cached client
            chunks2 = []
            async for chunk in service.async_stream_text_to_speech(
                text="Second sentence.", voice_id="v1", api_key_override="key1"
            ):
                chunks2.append(chunk)

            # Third request: _get_async_client() must reuse the existing cached client
            chunks3 = []
            async for chunk in service.async_stream_text_to_speech(
                text="Third sentence.", voice_id="v1", api_key_override="key1"
            ):
                chunks3.append(chunk)

        # httpx.AsyncClient was instantiated EXACTLY once across all 3 streaming requests
        assert init_call_count == 1
        assert len(created_clients) == 1
        assert chunks1 == [b"audio_chunk_data"]
        assert chunks2 == [b"audio_chunk_data"]
        assert chunks3 == [b"audio_chunk_data"]

        # Pool still holds the single open client
        loop = asyncio.get_running_loop()
        assert service._async_clients[id(loop)] is created_clients[0]
        assert not created_clients[0].is_closed
    finally:
        await service.aclose()


# ── Step 5 Tests: RAG Cache Isolation & Invalidation ──────────────────────────

def test_embedding_and_retrieval_cache_keys_are_separate():
    """Verify embedding cache keys (kb:emb:) and retrieval cache keys (kb:ctx:) are separated."""
    text = "What are your business hours?"
    emb_key = build_embedding_cache_key(text)
    ctx_key = build_retrieval_cache_key(text, [uuid.uuid4()])

    assert emb_key.startswith("kb:emb:")
    assert ctx_key.startswith("kb:ctx:")
    assert emb_key != ctx_key


def test_retrieval_cache_isolation_across_parameters():
    """Verify cache keys change when any retrieval-affecting parameter changes."""
    text = "pricing plans"
    kb1 = uuid.uuid4()
    kb2 = uuid.uuid4()

    base_key = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=5,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_1",
    )

    # 1. Tenant isolation
    diff_tenant = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=5,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_B",
        agent_id="agent_1",
    )
    assert base_key != diff_tenant

    # 2. Agent isolation
    diff_agent = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=5,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_2",
    )
    assert base_key != diff_agent

    # 3. KB set change
    diff_kbs = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1, kb2],
        top_k=5,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1", str(kb2): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_1",
    )
    assert base_key != diff_kbs

    # 4. Top-K change
    diff_topk = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=3,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_1",
    )
    assert base_key != diff_topk

    # 5. Score threshold change
    diff_threshold = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=5,
        score_threshold=0.75,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_1",
    )
    assert base_key != diff_threshold

    # 6. Embedding Model change
    diff_model = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=5,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_1",
        model_id="text-embedding-3-large",
    )
    assert base_key != diff_model

    # 7. Embedding Provider change
    diff_provider = build_retrieval_cache_key(
        transcript=text,
        kb_ids=[kb1],
        top_k=5,
        score_threshold=0.65,
        kb_revisions={str(kb1): "v1"},
        tenant_id="tenant_A",
        agent_id="agent_1",
        embedding_provider="gemini",
    )
    assert base_key != diff_provider


@pytest.mark.asyncio
async def test_kb_revision_cache_invalidation_invariant():
    """Verify that updating a KB revision atomically changes the cache key, invalidating old results."""
    mock_redis = MagicMock()
    redis_store = {}

    async def _mock_get(k):
        return redis_store.get(k)

    async def _mock_set(k, v, ex=None):
        redis_store[k] = v
        return True

    mock_redis.get = _mock_get
    mock_redis.set = _mock_set

    kb_id = uuid.uuid4()
    # 1. Initial revision
    rev1 = await get_kb_revision(kb_id, mock_redis)
    assert rev1 == "v1"

    key1 = build_retrieval_cache_key(
        transcript="test query",
        kb_ids=[kb_id],
        kb_revisions={str(kb_id): rev1},
    )

    # 2. Invalidate KB (document added/edited/deleted)
    rev2 = await invalidate_kb_cache(kb_id, mock_redis)
    assert rev2 != "v1"

    key2 = build_retrieval_cache_key(
        transcript="test query",
        kb_ids=[kb_id],
        kb_revisions={str(kb_id): rev2},
    )

    # Cache keys MUST differ, ensuring stale cached answers are never served after KB edit
    assert key1 != key2


@pytest.mark.asyncio
async def test_concurrent_retrieval_cache_hit_and_miss():
    """Verify concurrent requests with mocked Redis client handle hits and misses correctly."""
    mock_redis = MagicMock()
    cache_data = {}

    async def _mock_get(k):
        return cache_data.get(k)

    async def _mock_set(k, v, ex=None):
        cache_data[k] = v
        return True

    mock_redis.get = _mock_get
    mock_redis.set = _mock_set

    kb_id = uuid.uuid4()

    with patch("app.services.kb_retrieval_service._get_embedding_cached", new=AsyncMock(return_value=[0.1, 0.2, 0.3])), \
         patch("app.services.kb_retrieval_service._query_single_kb", new=AsyncMock(return_value=[])):
        
        # 1. Initial retrieval (cache miss)
        ctx1, lat1 = await retrieve_kb_context_for_turn(
            transcript="hours",
            kb_ids=[kb_id],
            redis_client=mock_redis,
        )
        
        # 2. Populate cache
        cache_key = build_retrieval_cache_key(
            transcript="hours",
            kb_ids=[kb_id],
            kb_revisions={str(kb_id): "v1"},
        )
        cache_data[cache_key] = '"Cached context for business hours"'

        # 3. Concurrent requests should hit cache immediately
        results = await asyncio.gather(
            retrieve_kb_context_for_turn(transcript="hours", kb_ids=[kb_id], redis_client=mock_redis),
            retrieve_kb_context_for_turn(transcript="hours", kb_ids=[kb_id], redis_client=mock_redis),
            retrieve_kb_context_for_turn(transcript="hours", kb_ids=[kb_id], redis_client=mock_redis),
        )
        for block, lat in results:
            assert block == "Cached context for business hours"


@pytest.mark.asyncio
async def test_kb_revision_redis_failure_fails_closed_and_bypasses_cache():
    """Verify that when Redis is unavailable or fails, get_kb_revision returns None and retrieval bypasses cache."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(side_effect=Exception("Redis connection refused"))
    mock_redis.set = AsyncMock(side_effect=Exception("Redis connection refused"))

    kb_id = uuid.uuid4()

    # 1. get_kb_revision must return None rather than a static "v1"
    rev = await get_kb_revision(kb_id, mock_redis)
    assert rev is None

    # 2. When Redis is unavailable (or None is passed), get_kb_revision returns None
    rev_none = await get_kb_revision(kb_id, None)
    assert rev_none is None

    # 3. retrieve_kb_context_for_turn must bypass cache and query DB directly (fail-closed for caching, fail-open for service)
    mock_query = AsyncMock(return_value=[])
    with patch("app.services.kb_retrieval_service._get_embedding_cached", new=AsyncMock(return_value=[0.1, 0.2, 0.3])), \
         patch("app.services.kb_retrieval_service._query_single_kb", new=mock_query):

        ctx, lat = await retrieve_kb_context_for_turn(
            transcript="query during redis outage",
            kb_ids=[kb_id],
            redis_client=mock_redis,
        )
        # Database query was executed directly because caching was safely bypassed
        mock_query.assert_called_once()
        assert ctx == ""


@pytest.mark.asyncio
async def test_invalidate_kb_cache_reports_failure_on_redis_error():
    """Verify that invalidate_kb_cache returns None when Redis is unavailable or errors, instead of returning an unpersisted revision."""
    kb_id = uuid.uuid4()

    # 1. Redis is None
    res_none = await invalidate_kb_cache(kb_id, None)
    assert res_none is None

    # 2. Redis set errors out
    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(side_effect=Exception("Redis timeout"))
    res_err = await invalidate_kb_cache(kb_id, mock_redis)
    assert res_err is None


@pytest.mark.asyncio
async def test_stale_cache_never_served_after_invalidation():
    """End-to-end invariant test: modifying a KB invalidates the cache so stale chunk text is never served."""
    mock_redis = MagicMock()
    redis_store = {}

    async def _mock_get(k):
        return redis_store.get(k)

    async def _mock_set(k, v, ex=None):
        redis_store[k] = v
        return True

    mock_redis.get = _mock_get
    mock_redis.set = _mock_set

    kb_id = uuid.uuid4()

    # 1. Initial retrieval populates cache with version 1 content
    from app.services.kb_retrieval_service import RetrievedChunk
    chunk_v1 = [RetrievedChunk(content="Original 30-day refund policy", score=0.9, metadata={})]
    chunk_v2 = [RetrievedChunk(content="Updated 60-day refund policy", score=0.95, metadata={})]

    with patch("app.services.kb_retrieval_service._get_embedding_cached", new=AsyncMock(return_value=[0.1, 0.2, 0.3])):
        with patch("app.services.kb_retrieval_service._query_single_kb", new=AsyncMock(return_value=chunk_v1)):
            ctx1, _ = await retrieve_kb_context_for_turn(
                transcript="refund policy",
                kb_ids=[kb_id],
                redis_client=mock_redis,
            )
            assert "Original 30-day refund policy" in ctx1

        # Second call hits cache (returns v1 content without querying DB)
        with patch("app.services.kb_retrieval_service._query_single_kb", new=AsyncMock(side_effect=AssertionError("DB should not be called"))):
            ctx1_cached, _ = await retrieve_kb_context_for_turn(
                transcript="refund policy",
                kb_ids=[kb_id],
                redis_client=mock_redis,
            )
            assert "Original 30-day refund policy" in ctx1_cached

        # 2. KB document is updated -> invalidate_kb_cache is triggered
        new_rev = await invalidate_kb_cache(kb_id, mock_redis)
        assert new_rev is not None

        # 3. Subsequent retrieval with new DB content MUST NOT return old cached content
        with patch("app.services.kb_retrieval_service._query_single_kb", new=AsyncMock(return_value=chunk_v2)):
            ctx2, _ = await retrieve_kb_context_for_turn(
                transcript="refund policy",
                kb_ids=[kb_id],
                redis_client=mock_redis,
            )
            assert "Updated 60-day refund policy" in ctx2
            assert "Original 30-day refund policy" not in ctx2

