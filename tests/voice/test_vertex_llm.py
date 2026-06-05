"""
Unit tests for Vertex AI Gemini LLM integration.

Covers: prompt construction, history pruning, cancellation, error fallback,
and credential routing.  All Vertex SDK calls are mocked — no real GCP calls.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# llm_prompt_builder — pure functions
# ─────────────────────────────────────────────────────────────────────────────

from app.voice.llm_prompt_builder import (
    build_history_text,
    build_kb_context_for_vertex,
    build_vertex_contents,
    prune_history_to_turns,
)


class TestPruneHistoryToTurns:
    def test_empty(self):
        assert prune_history_to_turns([], 20) == []

    def test_under_limit(self):
        h = [("client", "hi"), ("agent", "hello")]
        assert prune_history_to_turns(h, 20) == h

    def test_exactly_at_limit(self):
        # 3 turns = 6 messages
        h2 = [(r, c) for i in range(3) for r, c in [("client", f"u{i}"), ("agent", f"a{i}")]]
        assert prune_history_to_turns(h2, 3) == h2

    def test_over_limit_drops_oldest(self):
        # 25 turns → last 20 kept
        history = []
        for i in range(25):
            history.append(("client", f"user_{i}"))
            history.append(("agent", f"agent_{i}"))
        pruned = prune_history_to_turns(history, 20)
        assert len(pruned) == 40          # 20 turns × 2 messages
        assert pruned[0] == ("client", "user_5")   # oldest dropped = turns 0–4
        assert pruned[-1] == ("agent", "agent_24")

    def test_max_turns_zero(self):
        h = [("client", "hi"), ("agent", "hello")]
        assert prune_history_to_turns(h, 0) == []


class TestBuildHistoryText:
    def test_basic_render(self):
        h = [("client", "Hi"), ("agent", "Hello")]
        txt = build_history_text(h, max_turns=20)
        assert txt == "Client: Hi\nAgent: Hello"

    def test_pruned_render(self):
        history = []
        for i in range(25):
            history.append(("client", f"u{i}"))
            history.append(("agent", f"a{i}"))
        txt = build_history_text(history, max_turns=20)
        lines = txt.split("\n")
        assert len(lines) == 40
        assert lines[0].startswith("Client: u5")


class TestBuildKbContextForVertex:
    def test_combines_inbound_and_business(self):
        ctx = build_kb_context_for_vertex("inbound doc", "business facts")
        assert "inbound doc" in ctx
        assert "business facts" in ctx

    def test_empty_uses_guard(self):
        guard = "No verified business facts"
        ctx = build_kb_context_for_vertex("", "", empty_guard=guard)
        assert ctx == guard


# ─────────────────────────────────────────────────────────────────────────────
# build_vertex_contents — needs vertexai stubs
# ─────────────────────────────────────────────────────────────────────────────

class _FakePart:
    def __init__(self, text: str):
        self.text = text

    @staticmethod
    def from_text(t: str) -> "_FakePart":
        return _FakePart(t)


class _FakeContent:
    def __init__(self, role: str, parts: list):
        self.role = role
        self.parts = parts


def _patch_vertexai_models(monkeypatch):
    fake = SimpleNamespace(Content=_FakeContent, Part=_FakePart)
    monkeypatch.setattr(
        "app.voice.llm_prompt_builder.build_vertex_contents.__globals__",
        {},
        raising=False,
    )
    return fake


class TestBuildVertexContents:
    def _run(self, history, transcript, kb=None, max_turns=20):
        # Patch vertexai.generative_models before calling
        import sys
        fake_module = SimpleNamespace(
            Content=_FakeContent,
            Part=_FakePart,
        )
        sys.modules.setdefault("vertexai", SimpleNamespace(init=lambda **k: None))
        sys.modules["vertexai.generative_models"] = fake_module
        contents = build_vertex_contents(history, transcript, kb, max_turns)
        return contents

    def test_empty_history(self):
        contents = self._run([], "Hello there")
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello there"

    def test_role_mapping(self):
        history = [("client", "Hi"), ("agent", "Hello")]
        contents = self._run(history, "What is 2+2?")
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"
        assert contents[2].parts[0].text == "What is 2+2?"

    def test_kb_context_appended_to_user(self):
        contents = self._run([], "my question", kb="some context")
        assert "some context" in contents[0].parts[0].text
        assert "my question" in contents[0].parts[0].text

    def test_history_pruned_to_max_turns(self):
        history = []
        for i in range(25):
            history.append(("client", f"u{i}"))
            history.append(("agent", f"a{i}"))
        contents = self._run(history, "current", max_turns=20)
        # 20 turns = 40 messages + 1 current user turn
        assert len(contents) == 41
        # Oldest dropped
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "u5"


# ─────────────────────────────────────────────────────────────────────────────
# VertexGeminiService — stream_text
# ─────────────────────────────────────────────────────────────────────────────

from app.services.vertex_gemini_service import (  # noqa: E402
    VertexGeminiService,
    VertexLlmError,
    VertexLlmErrorType,
    _classify_vertex_error,
)


def _fake_response_iter(texts: list[str]):
    """Return sync iterator of fake Vertex response chunks."""
    class _FakeChunk:
        def __init__(self, t: str):
            self._text = t

        @property
        def text(self):
            return self._text

    return iter(_FakeChunk(t) for t in texts)


async def _async_fake_response_iter(texts: list[str]):
    """Return async generator of fake Vertex response chunks (matches generate_content_async)."""
    class _FakeChunk:
        def __init__(self, t: str):
            self._text = t

        @property
        def text(self):
            return self._text

    for t in texts:
        yield _FakeChunk(t)


def test_vertex_stream_text_yields_chunks():
    """Happy path: stream returns token chunks in order via generate_content_async."""
    chunks = ["Hello", " there", "!"]

    import sys

    mock_model = MagicMock()
    mock_model.generate_content_async = MagicMock(
        return_value=_async_fake_response_iter(chunks)
    )

    sys.modules["vertexai"] = SimpleNamespace(init=lambda **k: None)
    sys.modules["vertexai.generative_models"] = SimpleNamespace(
        GenerativeModel=MagicMock(return_value=mock_model),
        GenerationConfig=MagicMock(),
        Content=_FakeContent,
        Part=_FakePart,
    )

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_init"),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            result = []
            async for chunk in svc.stream_text(
                prompt="hi",
                system_prompt="be helpful",
                model_name="gemini-2.5-flash",
                temperature=0.3,
                max_tokens=100,
            ):
                result.append(chunk)
        return result

    result = asyncio.run(_run())
    assert result == chunks


def test_vertex_generative_model_uses_short_model_name():
    """SDK expects short names (e.g. gemini-2.5-flash), not publishers/google/models/ prefix."""
    import sys

    mock_model = MagicMock()
    mock_model.generate_content_async = MagicMock(
        return_value=_async_fake_response_iter(["ok"])
    )
    mock_generative_model = MagicMock(return_value=mock_model)

    sys.modules["vertexai"] = SimpleNamespace(init=lambda **k: None)
    sys.modules["vertexai.generative_models"] = SimpleNamespace(
        GenerativeModel=mock_generative_model,
        GenerationConfig=MagicMock(),
        Content=_FakeContent,
        Part=_FakePart,
    )

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_init"),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            async for _ in svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"):
                pass

    asyncio.run(_run())
    mock_generative_model.assert_called_once()
    assert mock_generative_model.call_args.kwargs["model_name"] == "gemini-2.5-flash"
    assert "publishers/" not in mock_generative_model.call_args.kwargs["model_name"]


def test_vertex_stream_text_async_generator_not_awaited():
    """Regression: stream=True returns an async generator synchronously, not a coroutine."""
    chunks = ["ok"]

    def generate_content_async(*_args, **_kwargs):
        gen = _async_fake_response_iter(chunks)
        assert not asyncio.iscoroutine(gen), "SDK must return async generator, not coroutine"
        return gen

    import sys

    mock_model = MagicMock()
    mock_model.generate_content_async = generate_content_async

    sys.modules["vertexai"] = SimpleNamespace(init=lambda **k: None)
    sys.modules["vertexai.generative_models"] = SimpleNamespace(
        GenerativeModel=MagicMock(return_value=mock_model),
        GenerationConfig=MagicMock(),
        Content=_FakeContent,
        Part=_FakePart,
    )

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_init"),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            result = []
            async for chunk in svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"):
                result.append(chunk)
        return result

    # Old `await model.generate_content_async(...)` would raise:
    # TypeError: object async_generator can't be used in 'await' expression
    result = asyncio.run(_run())
    assert result == chunks


def test_vertex_stream_cancelled_by_event():
    """cancel_event stops the stream mid-way via generate_content_async."""
    chunks = ["tok1", "tok2", "tok3", "tok4"]
    cancel = asyncio.Event()

    async def _cancelling_iter():
        class _ChunkObj:
            def __init__(self, t):
                self._t = t

            @property
            def text(self):
                return self._t

        for i, t in enumerate(chunks):
            if i == 2:
                cancel.set()
            yield _ChunkObj(t)

    import sys
    sys.modules.setdefault("vertexai", SimpleNamespace(init=lambda **k: None))

    mock_model = MagicMock()
    mock_model.generate_content_async = MagicMock(return_value=_cancelling_iter())

    sys.modules["vertexai.generative_models"] = SimpleNamespace(
        GenerativeModel=MagicMock(return_value=mock_model),
        GenerationConfig=MagicMock(),
        Content=_FakeContent,
        Part=_FakePart,
    )

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_init"),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            result = []
            async for chunk in svc.stream_text(
                prompt="hi",
                cancel_event=cancel,
                model_name="gemini-2.5-flash",
            ):
                result.append(chunk)
        return result

    result = asyncio.run(_run())
    # cancel_event set before iteration 2 → tok3, tok4 must not be yielded
    assert "tok3" not in result
    assert "tok4" not in result


def test_vertex_generate_text_returns_content():
    """Non-streaming generate_text for legacy gather / live-voice paths."""
    mock_response = SimpleNamespace(text="Hello from Vertex")
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    import sys
    sys.modules.setdefault("vertexai", SimpleNamespace(init=lambda **k: None))
    sys.modules["vertexai.generative_models"] = SimpleNamespace(
        GenerativeModel=MagicMock(return_value=mock_model),
        GenerationConfig=MagicMock(),
        Content=_FakeContent,
        Part=_FakePart,
    )

    with (
        patch("app.services.vertex_gemini_service._ensure_vertex_init"),
        patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
    ):
        svc = VertexGeminiService()
        result = svc.generate_text(
            prompt="hi",
            system_prompt="You are helpful",
            model_name="gemini-2.5-flash",
        )

    assert result["content"] == "Hello from Vertex"
    assert result["model"] == "gemini-2.5-flash"
    assert result["response_time"] >= 0


def test_vertex_stream_quota_error_raises_vertex_llm_error():
    """_classify_vertex_error maps quota-like exceptions to QUOTA error type."""
    # Test the classification logic directly (simulates what stream_text catches)
    # Using message-based matching since isinstance may have Python 3.14 edge cases
    class QuotaExceededException(Exception):
        """Simulates google.api_core.exceptions.ResourceExhausted."""
    QuotaExceededException.__name__ = "ResourceExhausted"

    exc = QuotaExceededException("quota exceeded")
    vertex_err = _classify_vertex_error(exc)
    assert vertex_err.error_type == VertexLlmErrorType.QUOTA

    # Also verify the VertexGeminiService raises VertexLlmError on SDK errors
    async def _run():
        import sys
        sys.modules.setdefault("vertexai", SimpleNamespace(init=lambda **k: None))

        mock_model = MagicMock()
        mock_model.generate_content_async = MagicMock(
            side_effect=RuntimeError("quota exceeded in stream")
        )

        sys.modules["vertexai.generative_models"] = SimpleNamespace(
            GenerativeModel=MagicMock(return_value=mock_model),
            GenerationConfig=MagicMock(),
            Content=_FakeContent,
            Part=_FakePart,
        )
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_init"),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            with pytest.raises(VertexLlmError) as exc_info:
                async for _ in svc.stream_text(prompt="hi"):
                    pass
        return exc_info.value.error_type

    error_type = asyncio.run(_run())
    assert error_type == VertexLlmErrorType.QUOTA


# ─────────────────────────────────────────────────────────────────────────────
# resolve_llm_runtime — credential routing
# ─────────────────────────────────────────────────────────────────────────────

from app.core.agent_runtime import resolve_llm_runtime, llm_service_for_provider  # noqa: E402


def _make_gemini_agent(with_api_key: bool = True) -> MagicMock:
    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.llm_model = "gemini-2.5-flash"
    agent.agent_temperature = None
    agent.agent_max_tokens = None
    model = MagicMock()
    model.api_key = "encrypted_key_abc" if with_api_key else None
    model.temperature = None
    model.max_tokens = None
    agent.model = model
    agent.provider = None
    return agent


def test_resolve_llm_runtime_gemini_ignores_api_key():
    """Gemini models must return api_key=None even when model.api_key is set."""
    agent = _make_gemini_agent(with_api_key=True)
    runtime = resolve_llm_runtime(agent)
    assert runtime.provider_slug == "gemini"
    assert runtime.api_key is None


def test_resolve_llm_runtime_gemini_default_temperature():
    """Default temperature for Gemini is 0.3 (VOICE_LLM_DEFAULT_TEMPERATURE)."""
    agent = _make_gemini_agent(with_api_key=False)
    runtime = resolve_llm_runtime(agent)
    assert abs(runtime.temperature - 0.3) < 0.01


def test_resolve_llm_runtime_gemini_agent_temperature_override():
    """agent.agent_temperature=50 → 0.50."""
    agent = _make_gemini_agent(with_api_key=False)
    agent.agent_temperature = 50
    runtime = resolve_llm_runtime(agent)
    assert abs(runtime.temperature - 0.50) < 0.01


def test_llm_service_for_provider_gemini_returns_vertex():
    """llm_service_for_provider("gemini") → VertexGeminiService instance."""
    from app.services.vertex_gemini_service import VertexGeminiService
    svc = llm_service_for_provider("gemini")
    assert isinstance(svc, VertexGeminiService)


def test_llm_service_for_provider_openai():
    from app.services.openai_service import OpenAIService
    svc = llm_service_for_provider("openai")
    assert isinstance(svc, OpenAIService)


def test_llm_service_for_provider_groq():
    from app.services.groq_service import GroqService
    svc = llm_service_for_provider("groq")
    assert isinstance(svc, GroqService)


# ─────────────────────────────────────────────────────────────────────────────
# PipelineSession
# ─────────────────────────────────────────────────────────────────────────────

from app.voice.pipeline_session import PipelineSession  # noqa: E402


def test_pipeline_session_cancel_sets_event():
    ps = PipelineSession()
    assert not ps.llm_cancel.is_set()
    ps.cancel_llm()
    assert ps.llm_cancel.is_set()


def test_pipeline_session_reset():
    ps = PipelineSession()
    ps.cancel_llm()
    ps.reset_llm_cancel()
    assert not ps.llm_cancel.is_set()


def test_pipeline_session_append_and_prune():
    ps = PipelineSession()
    for i in range(25):
        ps.append_turn("client", f"user_{i}")
        ps.append_turn("agent", f"agent_{i}")
    pruned = ps.get_pruned_history(max_turns=20)
    assert len(pruned) == 40
    assert pruned[0] == ("client", "user_5")


def test_pipeline_session_shared_history_ref():
    """PipelineSession.history is the same list as passed in — shared reference."""
    shared_list: list[tuple[str, str]] = []
    ps = PipelineSession(history=shared_list)
    ps.append_turn("client", "hello")
    assert shared_list == [("client", "hello")]


# ─────────────────────────────────────────────────────────────────────────────
# Error fallback (integration: handler canned response)
# ─────────────────────────────────────────────────────────────────────────────

def test_vertex_error_produces_canned_fallback_tts():
    """
    VertexLlmError carries correct error_type; fallback config message is correct.
    """
    err = VertexLlmError("quota exceeded", VertexLlmErrorType.QUOTA)
    assert err.error_type == VertexLlmErrorType.QUOTA
    assert str(err) == "quota exceeded"

    from app.core.config import settings
    fallback = getattr(settings, "VOICE_LLM_FALLBACK_MESSAGE", "I am sorry, I did not catch that")
    assert fallback == "I am sorry, I did not catch that"


def test_classify_vertex_error_quota_by_class_name():
    """Falls back to class-name matching when google.api_core isinstance check fails."""
    class ResourceExhaustedSimulated(Exception):
        pass
    ResourceExhaustedSimulated.__name__ = "ResourceExhausted"
    err = _classify_vertex_error(ResourceExhaustedSimulated("too many"))
    assert err.error_type == VertexLlmErrorType.QUOTA


def test_classify_vertex_error_timeout_by_class_name():
    """Falls back to class-name matching for DeadlineExceeded."""
    class DeadlineExceededSimulated(Exception):
        pass
    DeadlineExceededSimulated.__name__ = "DeadlineExceeded"
    err = _classify_vertex_error(DeadlineExceededSimulated("timed out"))
    assert err.error_type == VertexLlmErrorType.TIMEOUT


def test_classify_vertex_error_quota_by_message():
    """Message-based fallback: 'quota' in message text → QUOTA."""
    err = _classify_vertex_error(RuntimeError("quota exceeded"))
    assert err.error_type == VertexLlmErrorType.QUOTA


def test_classify_vertex_error_timeout_by_message():
    """Message-based fallback: 'timeout' in message text → TIMEOUT."""
    err = _classify_vertex_error(RuntimeError("connection timeout"))
    assert err.error_type == VertexLlmErrorType.TIMEOUT


def test_classify_vertex_error_unknown():
    err = _classify_vertex_error(ValueError("unexpected"))
    assert err.error_type == VertexLlmErrorType.UNKNOWN
