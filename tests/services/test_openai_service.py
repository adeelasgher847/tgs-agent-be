"""Unit tests for openai_service.py.

Covers three related changes:
  1. `stream_text` now uses an async client (`get_async_client` /
     `AsyncOpenAI`) instead of iterating a sync stream inside `async def` —
     regression test proves the event loop is not blocked while a stream is
     "in flight".
  2. `_is_reasoning_model` / `_completion_params` — the gpt-5.x parameter
     branch (no `temperature`, uses `max_completion_tokens`) vs. the
     unchanged branch for every other model.
  3. Boundary-level proof (mocking `client.chat.completions.create`) that
     `generate_text` / `chat_completion` / `stream_text` pass the right
     kwargs for both branches.

External HTTP is never hit — the OpenAI SDK client is mocked at the
boundary throughout, per repo convention.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.openai_service import (
    OpenAIService,
    _completion_params,
    _is_reasoning_model,
)


# ─────────────────────────────────── _is_reasoning_model / _completion_params ──

@pytest.mark.parametrize(
    "model_name",
    ["gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4", "GPT-5-MINI"],
)
def test_is_reasoning_model_true_for_gpt5_family(model_name):
    assert _is_reasoning_model(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    ["gpt-4.1", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-mini", "", None],
)
def test_is_reasoning_model_false_for_existing_models(model_name):
    assert _is_reasoning_model(model_name) is False


@pytest.mark.parametrize(
    "model_name", ["gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4"]
)
def test_completion_params_reasoning_model_omits_temperature(model_name):
    # max_tokens above the reasoning-model floor passes through unchanged.
    params = _completion_params(model_name, temperature=0.7, max_tokens=2500)
    assert params == {"model": model_name, "max_completion_tokens": 2500}
    assert "temperature" not in params
    assert "max_tokens" not in params


@pytest.mark.parametrize(
    "model_name", ["gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4"]
)
def test_completion_params_reasoning_model_floors_low_max_tokens(model_name):
    """
    Regression test for the silent-dead-air bug: gpt-5.x's max_completion_tokens
    is a SHARED budget covering invisible reasoning tokens + the visible
    completion. Voice agents commonly configure a small conversational
    max_tokens (e.g. 100, see agent_runtime.resolve_llm_runtime's default) —
    on a reasoning model that can be entirely consumed by reasoning, yielding
    content="" + finish_reason="length" with no exception raised. The floor
    below must always win for a low conversational setting.
    """
    params = _completion_params(model_name, temperature=0.7, max_tokens=100)
    assert params["max_completion_tokens"] == 1500
    assert "temperature" not in params
    assert "max_tokens" not in params


@pytest.mark.parametrize(
    "model_name", ["gpt-4.1", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
)
def test_completion_params_existing_model_unchanged(model_name):
    params = _completion_params(model_name, temperature=0.55, max_tokens=333)
    assert params == {"model": model_name, "temperature": 0.55, "max_tokens": 333}


@pytest.mark.parametrize(
    "model_name", ["gpt-4.1", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
)
def test_completion_params_existing_model_not_floored_at_low_max_tokens(model_name):
    """The gpt-5.x floor must NEVER apply to non-reasoning models — a low
    conversational max_tokens (e.g. 100) is passed straight through as-is."""
    params = _completion_params(model_name, temperature=0.55, max_tokens=100)
    assert params["max_tokens"] == 100
    assert "max_completion_tokens" not in params


# ───────────────────────────────────────────────── helpers ───────────────────

def _mock_chat_response(content="hello world", model="gpt-4.1"):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content), finish_reason="stop")]
    resp.model = model
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


class _FakeAsyncStreamChunks:
    """Async iterator mimicking an OpenAI streaming response, with a small
    `await asyncio.sleep(...)` between chunks to simulate real network I/O —
    used to prove the event loop isn't blocked while iterating."""

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


# ───────────────────────────────────────── generate_text / chat_completion ────

def test_generate_text_gpt5_omits_temperature_uses_max_completion_tokens():
    service = OpenAIService()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_chat_response(model="gpt-5")

    with patch.object(service, "get_client", return_value=mock_client):
        result = service.generate_text(
            "hi", model_name="gpt-5", temperature=0.9, max_tokens=200, api_key="k"
        )

    assert result["content"] == "hello world"
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-5"
    # max_tokens=200 is below the reasoning-model floor (1500) — floored so a
    # low conversational value can't starve the response of headroom.
    assert kwargs["max_completion_tokens"] == 1500
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_generate_text_existing_model_uses_temperature_and_max_tokens():
    service = OpenAIService()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_chat_response(model="gpt-4.1")

    with patch.object(service, "get_client", return_value=mock_client):
        result = service.generate_text(
            "hi", model_name="gpt-4.1", temperature=0.4, max_tokens=150, api_key="k"
        )

    assert result["content"] == "hello world"
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4.1"
    assert kwargs["temperature"] == 0.4
    assert kwargs["max_tokens"] == 150
    assert "max_completion_tokens" not in kwargs


def test_chat_completion_gpt5_omits_temperature():
    service = OpenAIService()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_chat_response(model="gpt-5.1")

    with patch.object(service, "get_client", return_value=mock_client):
        result = service.chat_completion(
            [{"role": "user", "content": "hi"}],
            model_name="gpt-5.1",
            temperature=0.9,
            max_tokens=64,
            api_key="k",
        )

    assert result["content"] == "hello world"
    _, kwargs = mock_client.chat.completions.create.call_args
    # max_tokens=64 is below the reasoning-model floor (1500) — floored.
    assert kwargs["max_completion_tokens"] == 1500
    assert "temperature" not in kwargs


def test_chat_completion_existing_model_unchanged():
    service = OpenAIService()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_chat_response(model="gpt-4o")

    with patch.object(service, "get_client", return_value=mock_client):
        service.chat_completion(
            [{"role": "user", "content": "hi"}],
            model_name="gpt-4o",
            temperature=0.6,
            max_tokens=80,
            api_key="k",
        )

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.6
    assert kwargs["max_tokens"] == 80
    assert "max_completion_tokens" not in kwargs


# ─────────────────────────────────────────────────────── stream_text ──────────

@pytest.mark.asyncio
async def test_stream_text_yields_chunks_in_order():
    service = OpenAIService()
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_FakeAsyncStreamChunks(["Hel", "lo ", "world"], delay=0)
    )

    with patch.object(service, "get_async_client", return_value=mock_async_client):
        chunks = [c async for c in service.stream_text("hi", model_name="gpt-4o-mini")]

    assert chunks == ["Hel", "lo ", "world"]


@pytest.mark.asyncio
async def test_stream_text_gpt5_omits_temperature_uses_max_completion_tokens():
    service = OpenAIService()
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_FakeAsyncStreamChunks(["ok"], delay=0)
    )

    with patch.object(service, "get_async_client", return_value=mock_async_client):
        chunks = [
            c
            async for c in service.stream_text(
                "hi", model_name="gpt-5", temperature=0.8, max_tokens=99, api_key="k"
            )
        ]

    assert chunks == ["ok"]
    _, kwargs = mock_async_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-5"
    # max_tokens=99 is below the reasoning-model floor (1500) — floored so
    # the streaming hot path never starves gpt-5's reasoning budget either.
    assert kwargs["max_completion_tokens"] == 1500
    assert kwargs["stream"] is True
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_stream_text_existing_model_uses_temperature_and_max_tokens():
    service = OpenAIService()
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_FakeAsyncStreamChunks(["ok"], delay=0)
    )

    with patch.object(service, "get_async_client", return_value=mock_async_client):
        [c async for c in service.stream_text("hi", model_name="gpt-4.1", temperature=0.3, max_tokens=42)]

    _, kwargs = mock_async_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 42
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_stream_text_does_not_block_event_loop():
    """Proves stream_text uses a real async iteration, not a blocking sync
    call — a concurrently scheduled task must make progress *while* the
    (simulated network) stream is "in flight" between chunks."""
    service = OpenAIService()
    mock_async_client = MagicMock()
    # Wider per-chunk delay + wider ticker delay than a first pass at this
    # test used (0.05s / 0.01s) — that gave only ~10ms of margin between the
    # "concurrent" and "blocked" timing lines, which is tight enough to flake
    # under CI scheduling jitter. Both are widened here to give a comfortable
    # ~100ms margin on both sides of the threshold below.
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
            async for chunk in service.stream_text("hi", model_name="gpt-4o-mini"):
                result.append(chunk)
        return result

    loop = asyncio.get_event_loop()
    start = loop.time()
    stream_result, _ = await asyncio.gather(consume_stream(), ticker())
    elapsed = loop.time() - start

    assert stream_result == ["a", "b", "c"]
    assert len(progress_marks) == 5
    # Stream has 3 chunks * 0.1s = 0.3s of simulated I/O; ticker has
    # 5 * 0.03s = 0.15s. If the two ran back-to-back (i.e. stream_text
    # blocked the loop while "in flight", the old sync-client bug), total
    # elapsed would be >= 0.45s. Running concurrently, elapsed stays close
    # to max(0.3, 0.15) = 0.3s. The 0.4s threshold below sits at the
    # midpoint, giving ~100ms of slack on both sides for CI scheduling
    # jitter while still failing hard if the old blocking bug regresses.
    assert elapsed < 0.4, f"expected concurrent interleaving, got elapsed={elapsed}"
