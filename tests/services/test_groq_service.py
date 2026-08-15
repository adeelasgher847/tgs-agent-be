"""Unit tests for groq_service.py — `stream_text` async-client regression.

Mirrors tests/services/test_openai_service.py's approach: mock the OpenAI-
compatible SDK client at the boundary and prove `stream_text` (a) still
yields chunks in the correct order and (b) no longer blocks the event loop
while a stream is "in flight" (old bug: sync client iterated inside
`async def`). Groq has no gpt-5-style reasoning-model parameter branching —
only the async-client fix applies here.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.groq_service import GroqService


def _mock_chat_response(content="hello world", model="llama-3.3-70b-versatile"):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content), finish_reason="stop")]
    resp.model = model
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


class _FakeAsyncStreamChunks:
    """Async iterator mimicking a Groq (OpenAI-compatible) streaming
    response, with a small `await asyncio.sleep(...)` between chunks to
    simulate real network I/O."""

    def __init__(self, texts, delay=0.01):
        self._texts = texts
        self._delay = delay

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for text in self._texts:
            await asyncio.sleep(self._delay)
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=text))]
            yield chunk


def test_generate_text_unaffected_uses_sync_client():
    """Non-streaming call sites still use the sync client, untouched."""
    service = GroqService()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_chat_response()

    with patch.object(service, "get_client", return_value=mock_client):
        result = service.generate_text("hi", model_name="llama-3.3-70b-versatile", api_key="k")

    assert result["content"] == "hello world"
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_stream_text_yields_chunks_in_order():
    service = GroqService()
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_FakeAsyncStreamChunks(["Hel", "lo ", "world"], delay=0)
    )

    with patch.object(service, "get_async_client", return_value=mock_async_client):
        chunks = [
            c
            async for c in service.stream_text(
                "hi", model_name="llama-3.3-70b-versatile", api_key="k"
            )
        ]

    assert chunks == ["Hel", "lo ", "world"]
    _, kwargs = mock_async_client.chat.completions.create.call_args
    assert kwargs["stream"] is True
    assert kwargs["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_stream_text_does_not_block_event_loop():
    """A concurrently scheduled task must make progress while the (simulated
    network) stream is in flight between chunks — proves stream_text no
    longer iterates a blocking sync stream inside an async def."""
    service = GroqService()
    mock_async_client = MagicMock()
    # Wider per-chunk / ticker delays than a first pass at this test used
    # (0.05s / 0.01s) — that left only ~10ms of margin between the
    # "concurrent" and "blocked" timing lines, tight enough to flake under
    # CI scheduling jitter. Widened here for ~100ms of margin on both sides.
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_FakeAsyncStreamChunks(["a", "b", "c"], delay=0.1)
    )

    progress_marks = []

    async def ticker():
        for i in range(5):
            await asyncio.sleep(0.03)
            progress_marks.append(i)

    async def consume_stream():
        result = []
        with patch.object(service, "get_async_client", return_value=mock_async_client):
            async for chunk in service.stream_text("hi", model_name="llama-3.3-70b-versatile"):
                result.append(chunk)
        return result

    loop = asyncio.get_event_loop()
    start = loop.time()
    stream_result, _ = await asyncio.gather(consume_stream(), ticker())
    elapsed = loop.time() - start

    assert stream_result == ["a", "b", "c"]
    assert len(progress_marks) == 5
    # 3 chunks * 0.1s = 0.3s simulated stream I/O; ticker = 5 * 0.03s = 0.15s.
    # Sequential (blocked) execution would take >= 0.45s; concurrent
    # execution stays close to max(0.3, 0.15) = 0.3s. The 0.4s threshold
    # sits at the midpoint, giving ~100ms slack on both sides.
    assert elapsed < 0.4, f"expected concurrent interleaving, got elapsed={elapsed}"
