import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.services.embedding_service import embed_text_for_rag_async
from app.services.kb_retrieval_service import retrieve_kb_context_for_turn, _get_embedding_cached
from app.voice.conversation_orchestrator import ConversationOrchestrator


@pytest.mark.asyncio
async def test_deepgram_endpointing_config_defaults():
    """Verify Deepgram endpointing defaults and configuration."""
    assert settings.DEEPGRAM_STT_ENDPOINTING_MS == 350
    assert settings.DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED == 500
    assert settings.DEEPGRAM_STT_UTTERANCE_END_MS == 1000


@pytest.mark.asyncio
async def test_prompt_history_max_messages_default():
    """Verify VOICE_HISTORY_MAX_MESSAGES default is trimmed to 12."""
    assert settings.VOICE_HISTORY_MAX_MESSAGES == 12


@pytest.mark.asyncio
async def test_embed_text_for_rag_async_openai():
    """Verify embed_text_for_rag_async uses AsyncOpenAI client directly without ThreadPoolExecutor."""
    fake_embedding = [0.1] * 1536
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_embedding)]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_resp)

    with patch("app.core.config.settings.OPENAI_API_KEY", "test-key"):
        with patch("app.core.openai_client.get_async_openai_client", return_value=mock_client):
            result = await embed_text_for_rag_async("What are your business hours?")
            assert result == fake_embedding
            mock_client.embeddings.create.assert_awaited_once_with(
                model=settings.RAG_EMBEDDING_MODEL,
                input="What are your business hours?",
            )


@pytest.mark.asyncio
async def test_kb_retrieval_service_cache_hit():
    """Verify retrieve_kb_context_for_turn hits Redis cache when available."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value='"--- KNOWLEDGE BASE CONTEXT ---\\nCached context\\n--- END CONTEXT ---"')

    kb_id = uuid.uuid4()
    context_block, latency_ms = await retrieve_kb_context_for_turn(
        transcript="business hours",
        kb_ids=[kb_id],
        redis_client=mock_redis,
    )
    assert "Cached context" in context_block
    assert latency_ms >= 0.0
    mock_redis.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_interim_rag_prefetch_trigger():
    """Verify maybe_prefetch_rag kicks off background task when transcript is stable."""
    handler = MagicMock()
    handler.call_flow = MagicMock(knowledge_base_ids=[uuid.uuid4()])
    handler.db = MagicMock()
    handler._rag_prefetch_task = None

    orchestrator = ConversationOrchestrator(handler)

    with patch("app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = ("Context", 5.0)
        await orchestrator.maybe_prefetch_rag("What are the pricing plans for standard tier?", confidence=0.85)

        assert handler._rag_prefetch_task is not None
        await handler._rag_prefetch_task  # await background task completion
        mock_retrieve.assert_awaited_once()


@pytest.mark.asyncio
async def test_barge_in_cancels_rag_prefetch_task():
    """Verify barge-in cancels any in-flight RAG prefetch task."""
    from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler

    handler = MagicMock()
    handler._llm_response_task = None
    handler._tts_pipeline = None

    # Create a dummy running task
    async def long_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(long_task())
    handler._rag_prefetch_task = task

    # Invoke method bound to handler mock instance
    await LiveKitBrowserCallHandler._cancel_inflight_llm_response(handler)
    await asyncio.sleep(0)
    assert task.cancelled() or task.done() or task.cancelling() > 0
    assert getattr(handler, "_rag_prefetch_task", None) is None
