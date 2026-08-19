"""
Unit tests for Vertex AI Gemini LLM integration.

Covers: prompt construction, history pruning, cancellation, error fallback,
and credential routing.  All Vertex SDK calls are mocked — no real GCP calls.

Note on the google-genai SDK and tests/conftest.py: conftest stubs
``sys.modules["google.genai"]`` with a bare ``MagicMock()`` for the whole
test session so unrelated unit tests never need real GCP credentials or the
SDK installed. That's fine for tests that only need to control the shape of
values returned from mocked calls (stream_text/generate_text/
generate_with_tools tests below configure their own fake client). But a
couple of tests here intentionally want to exercise the *real* installed
``google-genai==2.3.0`` package's actual behavior (``Part.from_text`` being
keyword-only, real ``google.genai.errors.ClientError``/``ServerError``
construction and their ``isinstance`` relationship to ``APIError`` as
consumed by ``_classify_vertex_error``) — see ``_real_google_genai()``
below, which temporarily swaps out the conftest stub for those cases only.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.llm_prompt_builder import (
    build_history_text,
    build_kb_context_for_vertex,
    build_vertex_contents,
    prune_history_to_turns,
)


@contextlib.contextmanager
def _real_google_genai():
    """
    Temporarily undo conftest's ``google`` / ``google.genai`` MagicMock stub
    so ``import google.genai...`` resolves to the real installed package for
    the duration of the ``with`` block, then restore the stub so it doesn't
    leak into other tests/files.
    """
    saved = {k: v for k, v in sys.modules.items() if k == "google" or k.startswith("google.")}
    for k in list(saved):
        del sys.modules[k]
    try:
        import google.genai.types as real_types
        import google.genai.errors as real_errors

        yield real_types, real_errors
    finally:
        for k in list(sys.modules):
            if k == "google" or k.startswith("google."):
                del sys.modules[k]
        sys.modules.update(saved)

# ─────────────────────────────────────────────────────────────────────────────
# llm_prompt_builder — pure functions
# ─────────────────────────────────────────────────────────────────────────────


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
# build_vertex_contents — needs google.genai.types stubs
# ─────────────────────────────────────────────────────────────────────────────

class _FakePart:
    """
    Mirrors the real ``google.genai.types.Part.from_text(*, text: str)``
    signature — keyword-only, confirmed against the installed
    google-genai==2.3.0 package. ``llm_prompt_builder.build_vertex_contents``
    calls it as ``Part.from_text(text=text)``; if that ever regressed to a
    positional call, this fake would raise TypeError just like the real SDK.
    """

    def __init__(self, text: str):
        self.text = text

    @staticmethod
    def from_text(*, text: str) -> "_FakePart":
        return _FakePart(text)


class _FakeContent:
    def __init__(self, role: str, parts: list):
        self.role = role
        self.parts = parts


class TestBuildVertexContents:
    def _run(self, history, transcript, kb=None, max_turns=20, monkeypatch=None):
        import app.voice.llm_prompt_builder as prompt_builder_module

        fake_types = SimpleNamespace(Content=_FakeContent, Part=_FakePart)
        # monkeypatch.setitem restores the original (conftest-stubbed)
        # sys.modules entry after this test — must not leak into other
        # tests, several of which need google.genai.types.GenerateContentConfig.
        monkeypatch.setitem(sys.modules, "google.genai", MagicMock(types=fake_types))
        # Force _load_vertex_models() to re-resolve Content/Part against our
        # fake types module rather than reusing whatever a prior test cached.
        monkeypatch.setattr(prompt_builder_module, "_Content", None)
        monkeypatch.setattr(prompt_builder_module, "_Part", None)
        return build_vertex_contents(history, transcript, kb, max_turns)

    def test_empty_history(self, monkeypatch):
        contents = self._run([], "Hello there", monkeypatch=monkeypatch)
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello there"

    def test_role_mapping(self, monkeypatch):
        history = [("client", "Hi"), ("agent", "Hello")]
        contents = self._run(history, "What is 2+2?", monkeypatch=monkeypatch)
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"
        assert contents[2].parts[0].text == "What is 2+2?"

    def test_kb_context_appended_to_user(self, monkeypatch):
        contents = self._run([], "my question", kb="some context", monkeypatch=monkeypatch)
        assert "some context" in contents[0].parts[0].text
        assert "my question" in contents[0].parts[0].text

    def test_history_pruned_to_max_turns(self, monkeypatch):
        history = []
        for i in range(25):
            history.append(("client", f"u{i}"))
            history.append(("agent", f"a{i}"))
        contents = self._run(history, "current", max_turns=20, monkeypatch=monkeypatch)
        # 20 turns = 40 messages + 1 current user turn
        assert len(contents) == 41
        # Oldest dropped
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "u5"

    def test_part_from_text_is_keyword_only(self):
        """Regression guard for the real SDK's actual signature difference
        from the deprecated vertexai SDK: Part.from_text(text) positionally
        must fail, only Part.from_text(text=...) is valid."""
        with pytest.raises(TypeError):
            _FakePart.from_text("positional not allowed")
        assert _FakePart.from_text(text="ok").text == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# VertexGeminiService — stream_text
# ─────────────────────────────────────────────────────────────────────────────

from app.services.vertex_gemini_service import (  # noqa: E402
    VertexGeminiService,
    VertexLlmError,
    VertexLlmErrorType,
    _classify_vertex_error,
    _ensure_vertex_client,
    _location_for_model,
    _vertex_clients,
)


# ─────────────────────────────────────────────────────────────────────────────
# _location_for_model — pure routing function
# ─────────────────────────────────────────────────────────────────────────────


class TestLocationForModel:
    @pytest.mark.parametrize(
        "model_name", ["gemini-3-flash-preview", "gemini-3.1-flash-lite"]
    )
    def test_global_only_models_route_to_global(self, model_name):
        assert _location_for_model(model_name) == "global"

    @pytest.mark.parametrize(
        "model_name",
        ["gemini-2.5-flash", "gemini-2.0-flash", "some-future-model-id"],
    )
    def test_other_models_route_to_configured_location(self, model_name, monkeypatch):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.VERTEX_AI_LOCATION",
            "us-central1",
        )
        assert _location_for_model(model_name) == "us-central1"


# ─────────────────────────────────────────────────────────────────────────────
# _ensure_vertex_client — per-location caching
# ─────────────────────────────────────────────────────────────────────────────


class TestEnsureVertexClientCaching:
    """
    ``_ensure_vertex_client`` does ``from google import genai`` then
    ``genai.Client(...)``. Because tests/conftest.py stubs
    ``sys.modules["google"]`` as a bare ``MagicMock()`` (not a real package),
    ``from google import genai`` resolves via attribute access on that mock
    object (``sys.modules["google"].genai``), NOT via ``sys.modules["google.genai"]``
    — those are two distinct mock objects. So the patch target here has to be
    the attribute actually consulted at call time: ``sys.modules["google"].genai``.
    ``unittest.mock.patch("google.genai.Client", ...)`` looks unintuitive but
    would silently patch the wrong object and never observe a call.
    """

    @pytest.fixture(autouse=True)
    def _clear_client_cache(self):
        """_vertex_clients is a module-level cache — clear before/after each
        test so tests don't leak state into each other."""
        _vertex_clients.clear()
        yield
        _vertex_clients.clear()

    def _patch_genai_client(self, monkeypatch, mock_client_cls):
        fake_genai = SimpleNamespace(Client=mock_client_cls)
        monkeypatch.setattr(sys.modules["google"], "genai", fake_genai, raising=False)

    def test_same_location_returns_cached_client(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.GOOGLE_CLOUD_PROJECT_ID",
            "test-project",
        )
        mock_client_cls = MagicMock(side_effect=lambda **_kwargs: MagicMock())
        self._patch_genai_client(monkeypatch, mock_client_cls)

        client1 = _ensure_vertex_client("us-central1")
        client2 = _ensure_vertex_client("us-central1")

        assert client1 is client2
        mock_client_cls.assert_called_once()

    def test_different_locations_each_get_own_client(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.GOOGLE_CLOUD_PROJECT_ID",
            "test-project",
        )
        mock_client_cls = MagicMock(side_effect=lambda **_kwargs: MagicMock())
        self._patch_genai_client(monkeypatch, mock_client_cls)

        global_client = _ensure_vertex_client("global")
        regional_client = _ensure_vertex_client("us-central1")
        # Requesting each location again should hit the cache, not
        # construct a new client.
        global_client_again = _ensure_vertex_client("global")
        regional_client_again = _ensure_vertex_client("us-central1")

        assert global_client is not regional_client
        assert global_client is global_client_again
        assert regional_client is regional_client_again
        assert mock_client_cls.call_count == 2
        call_locations = {c.kwargs["location"] for c in mock_client_cls.call_args_list}
        assert call_locations == {"global", "us-central1"}

    def test_no_project_configured_raises_runtime_error_for_any_location(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.GOOGLE_CLOUD_PROJECT_ID", None
        )
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.GCP_PROJECT_ID", None
        )
        with pytest.raises(RuntimeError, match="Vertex AI requires"):
            _ensure_vertex_client("global")
        with pytest.raises(RuntimeError, match="Vertex AI requires"):
            _ensure_vertex_client("us-central1")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: model_name → location routing through the public API
# ─────────────────────────────────────────────────────────────────────────────


class TestModelToLocationRoutingEndToEnd:
    """Confirms stream_text/generate_text/generate_with_tools call
    _ensure_vertex_client with the correct location for a given model_name,
    by patching _ensure_vertex_client itself and inspecting the location it
    was invoked with (rather than going through the real genai.Client)."""

    @pytest.mark.parametrize(
        "model_name,expected_location",
        [
            ("gemini-3-flash-preview", "global"),
            ("gemini-3.1-flash-lite", "global"),
            ("gemini-2.5-flash", "us-central1"),
        ],
    )
    def test_stream_text_routes_location(self, model_name, expected_location, monkeypatch):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.VERTEX_AI_LOCATION",
            "us-central1",
        )
        mock_stream = AsyncMock(return_value=_async_fake_response_iter(["ok"]))
        client = _make_mock_genai_client(stream=mock_stream)
        mock_ensure = MagicMock(return_value=client)

        async def _run():
            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    mock_ensure,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=[],
                ),
            ):
                svc = VertexGeminiService()
                async for _ in svc.stream_text(prompt="hi", model_name=model_name):
                    pass

        asyncio.run(_run())
        mock_ensure.assert_called_once_with(expected_location)

    @pytest.mark.parametrize(
        "model_name,expected_location",
        [
            ("gemini-3-flash-preview", "global"),
            ("gemini-3.1-flash-lite", "global"),
            ("gemini-2.5-flash", "us-central1"),
        ],
    )
    def test_generate_text_routes_location(self, model_name, expected_location, monkeypatch):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.VERTEX_AI_LOCATION",
            "us-central1",
        )
        mock_response = SimpleNamespace(text="Hello")
        mock_generate = MagicMock(return_value=mock_response)
        client = _make_mock_genai_client(generate=mock_generate)
        mock_ensure = MagicMock(return_value=client)

        with (
            patch(
                "app.services.vertex_gemini_service._ensure_vertex_client",
                mock_ensure,
            ),
            patch(
                "app.services.vertex_gemini_service.build_vertex_contents",
                return_value=[],
            ),
        ):
            svc = VertexGeminiService()
            svc.generate_text(prompt="hi", model_name=model_name)

        mock_ensure.assert_called_once_with(expected_location)

    @pytest.mark.parametrize(
        "model_name,expected_location",
        [
            ("gemini-3-flash-preview", "global"),
            ("gemini-3.1-flash-lite", "global"),
            ("gemini-2.5-flash", "us-central1"),
        ],
    )
    def test_generate_with_tools_routes_location(self, model_name, expected_location, monkeypatch):
        monkeypatch.setattr(
            "app.services.vertex_gemini_service.settings.VERTEX_AI_LOCATION",
            "us-central1",
        )
        response = SimpleNamespace(text="Hi there", candidates=[])
        async_generate = AsyncMock(return_value=response)
        client = _make_mock_genai_client(async_generate=async_generate)
        mock_ensure = MagicMock(return_value=client)
        tool_executor = AsyncMock()

        async def _run():
            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    mock_ensure,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=[],
                ),
            ):
                svc = VertexGeminiService()
                return await svc.generate_with_tools(
                    prompt="hi", model_name=model_name, tool_executor=tool_executor
                )

        asyncio.run(_run())
        mock_ensure.assert_called_once_with(expected_location)


class TestThinkingConfigRouting:
    """
    _THINKING_LEVEL_BY_MODEL pins an explicit ``thinking_config`` for the
    Gemini 3 family (latency-sensitive real-time voice). Confirms the
    ``config`` object actually passed to the underlying
    ``generate_content``/``generate_content_stream`` call carries the right
    ``thinking_config.thinking_level`` for mapped models, and — the more
    important regression guard — that unmapped models (e.g.
    ``gemini-2.5-flash``) get no ``thinking_config`` at all, i.e.
    ``config.thinking_config is None``.

    Uses ``_real_google_genai()`` (see top of file) rather than the
    conftest-stubbed MagicMock ``google.genai``, because a MagicMock
    ``types.GenerateContentConfig(**kwargs)`` call wouldn't actually store
    ``kwargs`` as inspectable attributes on its return value — only the real
    ``GenerateContentConfig``/``ThinkingConfig`` objects let us assert on the
    real field values the SDK will see.
    """

    @pytest.mark.parametrize(
        "model_name,expected_level",
        [
            ("gemini-3-flash-preview", "MINIMAL"),
            ("gemini-3.1-flash-lite", "MINIMAL"),
        ],
    )
    def test_stream_text_sets_thinking_config_for_mapped_models(self, model_name, expected_level):
        with _real_google_genai():
            mock_stream = AsyncMock(return_value=_async_fake_response_iter(["ok"]))
            client = _make_mock_genai_client(stream=mock_stream)

            async def _run():
                with (
                    patch(
                        "app.services.vertex_gemini_service._ensure_vertex_client",
                        return_value=client,
                    ),
                    patch(
                        "app.services.vertex_gemini_service.build_vertex_contents",
                        return_value=[],
                    ),
                ):
                    svc = VertexGeminiService()
                    async for _ in svc.stream_text(prompt="hi", model_name=model_name):
                        pass

            asyncio.run(_run())

        config = mock_stream.call_args.kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == expected_level

    def test_stream_text_no_thinking_config_for_unmapped_model(self):
        """gemini-2.5-flash is not in _THINKING_LEVEL_BY_MODEL — config must
        stay byte-for-byte unchanged (no thinking_config key at all)."""
        with _real_google_genai():
            mock_stream = AsyncMock(return_value=_async_fake_response_iter(["ok"]))
            client = _make_mock_genai_client(stream=mock_stream)

            async def _run():
                with (
                    patch(
                        "app.services.vertex_gemini_service._ensure_vertex_client",
                        return_value=client,
                    ),
                    patch(
                        "app.services.vertex_gemini_service.build_vertex_contents",
                        return_value=[],
                    ),
                ):
                    svc = VertexGeminiService()
                    async for _ in svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"):
                        pass

            asyncio.run(_run())

        config = mock_stream.call_args.kwargs["config"]
        assert config.thinking_config is None

    @pytest.mark.parametrize(
        "model_name,expected_level",
        [
            ("gemini-3-flash-preview", "MINIMAL"),
            ("gemini-3.1-flash-lite", "MINIMAL"),
        ],
    )
    def test_generate_text_sets_thinking_config_for_mapped_models(self, model_name, expected_level):
        with _real_google_genai():
            mock_response = SimpleNamespace(text="Hello")
            mock_generate = MagicMock(return_value=mock_response)
            client = _make_mock_genai_client(generate=mock_generate)

            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    return_value=client,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=[],
                ),
            ):
                svc = VertexGeminiService()
                svc.generate_text(prompt="hi", model_name=model_name)

        config = mock_generate.call_args.kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == expected_level

    def test_generate_text_no_thinking_config_for_unmapped_model(self):
        with _real_google_genai():
            mock_response = SimpleNamespace(text="Hello")
            mock_generate = MagicMock(return_value=mock_response)
            client = _make_mock_genai_client(generate=mock_generate)

            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    return_value=client,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=[],
                ),
            ):
                svc = VertexGeminiService()
                svc.generate_text(prompt="hi", model_name="gemini-2.5-flash")

        config = mock_generate.call_args.kwargs["config"]
        assert config.thinking_config is None

    @pytest.mark.parametrize(
        "model_name,expected_level",
        [
            ("gemini-3-flash-preview", "MINIMAL"),
            ("gemini-3.1-flash-lite", "MINIMAL"),
        ],
    )
    def test_generate_with_tools_sets_thinking_config_for_mapped_models(
        self, model_name, expected_level
    ):
        with _real_google_genai():
            response = SimpleNamespace(text="Hi there", candidates=[])
            async_generate = AsyncMock(return_value=response)
            client = _make_mock_genai_client(async_generate=async_generate)
            tool_executor = AsyncMock()

            async def _run():
                with (
                    patch(
                        "app.services.vertex_gemini_service._ensure_vertex_client",
                        return_value=client,
                    ),
                    patch(
                        "app.services.vertex_gemini_service.build_vertex_contents",
                        return_value=[],
                    ),
                ):
                    svc = VertexGeminiService()
                    return await svc.generate_with_tools(
                        prompt="hi", model_name=model_name, tool_executor=tool_executor
                    )

            asyncio.run(_run())

        config = async_generate.call_args.kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == expected_level

    def test_generate_with_tools_no_thinking_config_for_unmapped_model(self):
        with _real_google_genai():
            response = SimpleNamespace(text="Hi there", candidates=[])
            async_generate = AsyncMock(return_value=response)
            client = _make_mock_genai_client(async_generate=async_generate)
            tool_executor = AsyncMock()

            async def _run():
                with (
                    patch(
                        "app.services.vertex_gemini_service._ensure_vertex_client",
                        return_value=client,
                    ),
                    patch(
                        "app.services.vertex_gemini_service.build_vertex_contents",
                        return_value=[],
                    ),
                ):
                    svc = VertexGeminiService()
                    return await svc.generate_with_tools(
                        prompt="hi", model_name="gemini-2.5-flash", tool_executor=tool_executor
                    )

            asyncio.run(_run())

        config = async_generate.call_args.kwargs["config"]
        assert config.thinking_config is None


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


def _make_mock_genai_client(*, stream=None, generate=None, async_generate=None):
    """
    Build a MagicMock standing in for the google.genai.Client returned by
    _ensure_vertex_client(). Only the pieces stream_text/generate_text/
    generate_with_tools actually touch are configured:
      - client.aio.models.generate_content_stream — async def, returns an
        AsyncIterator when awaited (real SDK shape, confirmed against the
        installed google-genai==2.3.0 source).
      - client.models.generate_content — plain sync method.
      - client.aio.models.generate_content — async method (used by
        generate_with_tools).
    """
    client = MagicMock()
    if stream is not None:
        client.aio.models.generate_content_stream = stream
    if generate is not None:
        client.models.generate_content = generate
    if async_generate is not None:
        client.aio.models.generate_content = async_generate
    return client


@pytest.mark.parametrize(
    "model_name", ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]
)
def test_vertex_stream_text_yields_chunks(model_name):
    """Happy path: stream returns token chunks in order via
    client.aio.models.generate_content_stream, and model_name is forwarded
    verbatim to the model= kwarg (covers both existing and new Gemini 3
    model IDs)."""
    chunks = ["Hello", " there", "!"]

    mock_stream = AsyncMock(return_value=_async_fake_response_iter(chunks))
    client = _make_mock_genai_client(stream=mock_stream)

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_client", return_value=client),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            result = []
            async for chunk in svc.stream_text(
                prompt="hi",
                system_prompt="be helpful",
                model_name=model_name,
                temperature=0.3,
                max_tokens=100,
            ):
                result.append(chunk)
        return result

    result = asyncio.run(_run())
    assert result == chunks
    assert mock_stream.call_args.kwargs["model"] == model_name


def test_vertex_generative_model_uses_short_model_name():
    """SDK expects short names (e.g. gemini-2.5-flash), not publishers/google/models/ prefix."""
    mock_stream = AsyncMock(return_value=_async_fake_response_iter(["ok"]))
    client = _make_mock_genai_client(stream=mock_stream)

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_client", return_value=client),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            async for _ in svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"):
                pass

    asyncio.run(_run())
    mock_stream.assert_called_once()
    assert mock_stream.call_args.kwargs["model"] == "gemini-2.5-flash"
    assert "publishers/" not in mock_stream.call_args.kwargs["model"]


def test_vertex_stream_text_awaits_coroutine_from_generate_content_async():
    """Documents the new SDK's actual async-streaming shape (confirmed against
    the installed google-genai==2.3.0 source:
    `AsyncModels.generate_content_stream` is itself declared `async def ... ->
    AsyncIterator[...]`, i.e. calling it returns a coroutine that must be
    awaited to get the actual async-iterable, exactly as documented in the
    SDK's own usage example: `async for chunk in await
    client.aio.models.generate_content_stream(...)`.

    This test builds a hand-written `async def` (not an AsyncMock) for
    `generate_content_stream` to prove `stream_text` really does `await` the
    call before iterating, rather than iterating a bare coroutine directly
    (which would raise `TypeError: 'async for' requires an object with
    __aiter__ method, got coroutine`)."""
    chunks = ["ok"]

    async def generate_content_stream(*_args, **_kwargs):
        return _async_fake_response_iter(chunks)

    client = _make_mock_genai_client(stream=generate_content_stream)

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_client", return_value=client),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            result = []
            async for chunk in svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"):
                result.append(chunk)
        return result

    result = asyncio.run(_run())
    assert result == chunks


def test_vertex_stream_cancelled_by_event():
    """cancel_event stops the stream mid-way via generate_content_stream."""
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

    mock_stream = AsyncMock(return_value=_cancelling_iter())
    client = _make_mock_genai_client(stream=mock_stream)

    async def _run():
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_client", return_value=client),
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


@pytest.mark.parametrize(
    "model_name", ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]
)
def test_vertex_generate_text_returns_content(model_name):
    """Non-streaming generate_text for legacy gather / live-voice paths,
    via client.models.generate_content (plain sync call, no .aio)."""
    mock_response = SimpleNamespace(text="Hello from Vertex")
    mock_generate = MagicMock(return_value=mock_response)
    client = _make_mock_genai_client(generate=mock_generate)

    with (
        patch("app.services.vertex_gemini_service._ensure_vertex_client", return_value=client),
        patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
    ):
        svc = VertexGeminiService()
        result = svc.generate_text(
            prompt="hi",
            system_prompt="You are helpful",
            model_name=model_name,
        )

    assert result["content"] == "Hello from Vertex"
    assert result["model"] == model_name
    assert result["response_time"] >= 0
    assert mock_generate.call_args.kwargs["model"] == model_name


# ─────────────────────────────────────────────────────────────────────────────
# VertexGeminiService — generate_with_tools
# ─────────────────────────────────────────────────────────────────────────────


class _FakeToolPart:
    """Mirrors ``google.genai.types.Part.from_function_response(*, name, response)``
    closely enough to assert on what generate_with_tools builds."""

    def __init__(self, name, response):
        self.name = name
        self.response = response

    @staticmethod
    def from_function_response(*, name, response):
        return _FakeToolPart(name, response)


class _FakeToolContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


def _install_fake_genai_types(monkeypatch):
    """
    Swap ``sys.modules["google.genai"]`` for a controlled fake ``types``
    namespace (same technique as ``TestBuildVertexContents._run`` above) so
    ``types.Content(role=..., parts=...)`` / ``types.Part.from_function_response``
    calls inside ``generate_with_tools`` build real, inspectable objects
    instead of opaque MagicMock auto-attrs. ``monkeypatch`` restores the
    original (conftest-stubbed) sys.modules entry after the test.
    Returns the fake ``types`` namespace for assertions.
    """
    fake_types = SimpleNamespace(
        Content=_FakeToolContent,
        Part=_FakeToolPart,
        FunctionDeclaration=MagicMock(),
        Tool=MagicMock(),
        GenerateContentConfig=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "google.genai", MagicMock(types=fake_types))
    return fake_types


def _make_fc_part(name: str, args):
    """A fake response Part carrying a function_call, matching the shape
    ``generate_with_tools`` reads: ``part.function_call.name`` / ``.args``."""
    return SimpleNamespace(function_call=SimpleNamespace(name=name, args=args))


class TestGenerateWithTools:
    def test_no_function_call_returns_text_directly(self):
        """No function_call part in the response → returns response.text.strip(),
        tool_executor is never called, and there is no follow-up generate_content
        call (client.aio.models.generate_content is called exactly once)."""
        response = SimpleNamespace(text="  Hello, how can I help?  ", candidates=[])
        async_generate = AsyncMock(return_value=response)
        client = _make_mock_genai_client(async_generate=async_generate)
        tool_executor = AsyncMock()

        async def _run():
            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    return_value=client,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=[],
                ),
            ):
                svc = VertexGeminiService()
                return await svc.generate_with_tools(
                    prompt="hi", tool_executor=tool_executor
                )

        result = asyncio.run(_run())

        assert result == "Hello, how can I help?"
        tool_executor.assert_not_called()
        async_generate.assert_awaited_once()

    def test_function_call_invokes_tool_executor_and_wraps_response_in_content(self, monkeypatch):
        """Function_call present → tool_executor is awaited with (name, args); the
        follow-up generate_content call's contents include a
        types.Content(role="user", parts=[Part.from_function_response(...)]) turn —
        this is the regression guard for the role="function" → role="user" bug fix."""
        _install_fake_genai_types(monkeypatch)

        fc_part = _make_fc_part("check_availability", {"date": "2026-08-17"})
        candidate_content = SimpleNamespace(parts=[fc_part])
        first_response = SimpleNamespace(candidates=[SimpleNamespace(content=candidate_content)])
        follow_up_response = SimpleNamespace(text="  You're all set for 10am.  ")

        async_generate = AsyncMock(side_effect=[first_response, follow_up_response])
        client = _make_mock_genai_client(async_generate=async_generate)

        tool_result = {"slots": ["2026-08-17T10:00:00Z"]}
        tool_executor = AsyncMock(return_value=tool_result)

        base_contents: list = []

        async def _run():
            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    return_value=client,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=base_contents,
                ),
            ):
                svc = VertexGeminiService()
                return await svc.generate_with_tools(
                    prompt="What's available tomorrow?", tool_executor=tool_executor
                )

        result = asyncio.run(_run())

        # tool_executor awaited with the resolved (name, args)
        tool_executor.assert_awaited_once_with("check_availability", {"date": "2026-08-17"})

        # Two generate_content calls: the initial turn + the follow-up after
        # the tool result is fed back.
        assert async_generate.await_count == 2

        # The function-response turn must be wrapped in Content(role="user", ...)
        # — NOT role="function" (the old vertexai SDK convention this was fixed
        # away from).
        follow_up_contents = async_generate.call_args_list[1].kwargs["contents"]
        function_response_turns = [
            c for c in follow_up_contents if isinstance(c, _FakeToolContent)
        ]
        assert len(function_response_turns) == 1
        wrapped_turn = function_response_turns[0]
        assert wrapped_turn.role == "user"

        # Part.from_function_response called with the right name/response.
        assert len(wrapped_turn.parts) == 1
        wrapped_part = wrapped_turn.parts[0]
        assert isinstance(wrapped_part, _FakeToolPart)
        assert wrapped_part.name == "check_availability"
        assert wrapped_part.response == tool_result

        # The model's own function_call turn is also appended onto the same
        # contents list used for the follow-up call.
        assert candidate_content in follow_up_contents

        # Final text comes from the follow-up call's .text, stripped.
        assert result == "You're all set for 10am."

    def test_function_call_with_no_args_defaults_to_empty_dict(self, monkeypatch):
        """fc.args falsy (None) → fc_args passed to tool_executor is {} (matches
        `dict(fc.args) if fc.args else {}` in the source)."""
        _install_fake_genai_types(monkeypatch)

        fc_part = _make_fc_part("book_appointment", None)
        candidate_content = SimpleNamespace(parts=[fc_part])
        first_response = SimpleNamespace(candidates=[SimpleNamespace(content=candidate_content)])
        follow_up_response = SimpleNamespace(text="Booked.")

        async_generate = AsyncMock(side_effect=[first_response, follow_up_response])
        client = _make_mock_genai_client(async_generate=async_generate)
        tool_executor = AsyncMock(return_value={"booked": True})

        async def _run():
            with (
                patch(
                    "app.services.vertex_gemini_service._ensure_vertex_client",
                    return_value=client,
                ),
                patch(
                    "app.services.vertex_gemini_service.build_vertex_contents",
                    return_value=[],
                ),
            ):
                svc = VertexGeminiService()
                return await svc.generate_with_tools(
                    prompt="Book it", tool_executor=tool_executor
                )

        asyncio.run(_run())

        tool_executor.assert_awaited_once_with("book_appointment", {})


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
        mock_stream = AsyncMock(side_effect=RuntimeError("quota exceeded in stream"))
        client = _make_mock_genai_client(stream=mock_stream)
        with (
            patch("app.services.vertex_gemini_service._ensure_vertex_client", return_value=client),
            patch("app.services.vertex_gemini_service.build_vertex_contents", return_value=[]),
        ):
            svc = VertexGeminiService()
            with pytest.raises(VertexLlmError) as exc_info:
                async for _ in svc.stream_text(prompt="hi"):
                    pass
        return exc_info.value.error_type

    error_type = asyncio.run(_run())
    assert error_type == VertexLlmErrorType.QUOTA


class TestClassifyVertexErrorGenaiTypes:
    """
    _classify_vertex_error's primary branch matches real
    google.genai.errors.APIError/ClientError/ServerError by .code/.status.
    tests/conftest.py stubs google.genai as a MagicMock for the whole
    session, which would make `isinstance(exc, genai_errors.APIError)`
    raise TypeError (caught, falls through to the message-based fallback)
    rather than exercising this branch — so these two tests use
    _real_google_genai() to temporarily restore the real installed package.
    """

    def test_client_error_429_resource_exhausted_is_quota(self):
        with _real_google_genai() as (_types, real_errors):
            # Real APIError.__init__(code, response_json, response=None) signature,
            # confirmed against the installed google-genai==2.3.0 source.
            exc = real_errors.ClientError(
                429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}}
            )
            assert exc.code == 429
            assert exc.status == "RESOURCE_EXHAUSTED"

            vertex_err = _classify_vertex_error(exc)
        assert vertex_err.error_type == VertexLlmErrorType.QUOTA

    def test_client_error_504_deadline_exceeded_is_timeout(self):
        with _real_google_genai() as (_types, real_errors):
            exc = real_errors.ServerError(
                504, {"error": {"status": "DEADLINE_EXCEEDED", "message": "deadline exceeded"}}
            )
            assert exc.code == 504
            assert exc.status == "DEADLINE_EXCEEDED"

            vertex_err = _classify_vertex_error(exc)
        assert vertex_err.error_type == VertexLlmErrorType.TIMEOUT


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
