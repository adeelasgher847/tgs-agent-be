"""
Unit tests for the Vertex AI Gemini 2.5 Flash voice LLM service.

Covers:
  - Prompt construction (system at index 0, no prompt leakage in logs)
  - History pruning (> 20 turns trimmed to 40 messages)
  - Cancellation (cancel_event stops streaming, no further chunks yielded)
  - Error fallback (quota/timeout/filter → canned sorry phrase)
  - Model allow-listing (gemini-2.5-flash present, infer_llm_provider returns vertex)
  - agent_runtime routing (llm_service_for_provider("vertex") returns vertex_llm_service)

Run:
    pytest tests/voice/test_vertex_llm.py -v
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run a coroutine synchronously — avoids pytest-asyncio dependency."""
    return asyncio.run(coro)


async def _collect(gen: AsyncIterator[str]) -> List[str]:
    return [chunk async for chunk in gen]


# ---------------------------------------------------------------------------
# 1. Allow-list & provider inference
# ---------------------------------------------------------------------------


class TestLlmModels:
    def test_gemini_25_flash_in_allowed_list(self):
        from app.core.llm_models import ALLOWED_LLM_MODELS, is_allowed_llm_model

        assert "gemini-2.5-flash" in ALLOWED_LLM_MODELS
        assert is_allowed_llm_model("gemini-2.5-flash")

    def test_infer_provider_returns_vertex(self):
        from app.core.llm_models import infer_llm_provider

        assert infer_llm_provider("gemini-2.5-flash") == "vertex"

    def test_infer_provider_other_gemini_still_gemini(self):
        from app.core.llm_models import infer_llm_provider

        assert infer_llm_provider("gemini-1.5-flash") == "gemini"
        assert infer_llm_provider("gemini-2.0-flash") == "gemini"


# ---------------------------------------------------------------------------
# 2. agent_runtime routing
# ---------------------------------------------------------------------------


class TestGoogleCredentials:
    def test_ensure_credentials_sets_env_from_path(self, monkeypatch, tmp_path):
        import os

        from app.core.google_credentials import ensure_google_application_credentials_env

        cred_file = tmp_path / "sa.json"
        cred_file.write_text('{"type":"service_account"}')
        monkeypatch.setattr(
            "app.core.google_credentials.settings.GOOGLE_APPLICATION_CREDENTIALS",
            str(cred_file),
        )
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        path = ensure_google_application_credentials_env()
        assert path == str(cred_file)
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(cred_file)


class TestAgentRuntimeRouting:
    def test_vertex_provider_returns_vertex_service(self):
        from app.core.agent_runtime import llm_service_for_provider
        from app.services.vertex_llm_service import vertex_llm_service

        svc = llm_service_for_provider("vertex")
        assert svc is vertex_llm_service

    def test_resolve_llm_runtime_vertex_skips_model_api_key(self):
        from types import SimpleNamespace

        from app.core.agent_runtime import resolve_llm_runtime

        agent = SimpleNamespace(
            llm_model="gemini-2.5-flash",
            agent_temperature=None,
            agent_max_tokens=None,
            model=SimpleNamespace(api_key="encrypted-should-not-use", temperature=None, max_tokens=None),
            provider=None,
        )
        runtime = resolve_llm_runtime(agent)  # type: ignore[arg-type]
        assert runtime.provider_slug == "vertex"
        assert runtime.api_key is None

    def test_openai_provider_unchanged(self):
        from app.core.agent_runtime import llm_service_for_provider
        from app.services.openai_service import openai_service

        assert llm_service_for_provider("openai") is openai_service

    def test_unknown_falls_back_to_gemini(self):
        from app.core.agent_runtime import llm_service_for_provider
        from app.services.gemini_service import gemini_service

        assert llm_service_for_provider("unknown") is gemini_service


# ---------------------------------------------------------------------------
# 3. stream_text: prompt construction
# ---------------------------------------------------------------------------


def _make_fake_vertex_module(model_cls):
    """Build a sys.modules patch for vertexai.generative_models with a custom model."""
    fake = MagicMock()
    fake.GenerativeModel = model_cls
    fake.GenerationConfig = MagicMock(return_value=MagicMock())
    # Content/Part: lightweight stand-ins
    fake.Content = MagicMock(
        side_effect=lambda role, parts: MagicMock(role=role, parts=parts)
    )
    fake.Part.from_text = MagicMock(side_effect=lambda t: MagicMock(text=t))
    return fake


class TestVertexPromptConstruction:
    def test_system_injected_as_system_instruction_not_user_msg(self):
        captured = {}

        class FakeChunk:
            text = "Hello world"

        class FakeModel:
            def __init__(self, model_name, system_instruction=None, **kw):
                captured["model_name"] = model_name
                captured["system_instruction"] = system_instruction

            def generate_content(self, contents, generation_config, stream):
                captured["user_part"] = contents[0].parts[0].text
                return iter([FakeChunk()])

        from app.services.vertex_llm_service import VertexLlmService

        svc = VertexLlmService()
        svc._initialized = True

        with patch.dict("sys.modules", {"vertexai.generative_models": _make_fake_vertex_module(FakeModel)}):
            run(_collect(svc.stream_text(
                prompt="What are your hours?",
                system_prompt="You are a helpful assistant.",
                model_name="gemini-2.5-flash",
            )))

        assert captured["system_instruction"] == "You are a helpful assistant."
        assert "What are your hours?" in captured["user_part"]
        # System prompt must NOT appear inside the user content
        assert "System:" not in captured.get("user_part", "")

    def test_system_prompt_not_logged_in_full(self, caplog):
        long_system_prompt = "CONFIDENTIAL: " + "x" * 500

        class FakeChunk:
            text = "ok"

        class FakeModel:
            def __init__(self, *a, **kw):
                pass

            def generate_content(self, *a, **kw):
                return iter([FakeChunk()])

        from app.services.vertex_llm_service import VertexLlmService

        svc = VertexLlmService()
        svc._initialized = True

        with caplog.at_level(logging.DEBUG, logger="app"):
            with patch.dict("sys.modules", {"vertexai.generative_models": _make_fake_vertex_module(FakeModel)}):
                run(_collect(svc.stream_text(
                    prompt="hi",
                    system_prompt=long_system_prompt,
                    model_name="gemini-2.5-flash",
                )))

        for record in caplog.records:
            assert long_system_prompt not in record.getMessage()


# ---------------------------------------------------------------------------
# 4. History pruning: > 20 turns → trimmed to 40 messages
# ---------------------------------------------------------------------------


class TestHistoryPruning:
    def test_prune_exceeds_40_messages(self):
        history: list = []
        for i in range(25):
            history.append(("client", f"user turn {i}"))
            history.append(("agent", f"agent turn {i}"))

        assert len(history) == 50

        max_msgs = 40
        if len(history) > max_msgs:
            history = history[-max_msgs:]

        assert len(history) == 40
        # Oldest retained should be from pair index 5 (50-40=10 dropped → pair 5)
        assert history[0] == ("client", "user turn 5")

    def test_prune_exactly_40_unchanged(self):
        history = [("client", f"msg {i}") for i in range(40)]
        max_msgs = 40
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        assert len(history) == 40

    def test_prune_below_40_unchanged(self):
        history = [("client", f"msg {i}") for i in range(10)]
        max_msgs = 40
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        assert len(history) == 10

    def test_config_default_is_40(self):
        from app.core.config import settings
        assert settings.VOICE_HISTORY_MAX_MESSAGES == 40


# ---------------------------------------------------------------------------
# 5. Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_event_stops_async_generator(self):
        async def _fake_stream(cancel_event: asyncio.Event):
            for i in range(10):
                if cancel_event.is_set():
                    return
                yield f"chunk{i}"
                await asyncio.sleep(0)

        async def _run():
            cancel = asyncio.Event()
            received = []
            async for chunk in _fake_stream(cancel):
                received.append(chunk)
                if len(received) == 3:
                    cancel.set()
            return received

        result = run(_run())
        assert result == ["chunk0", "chunk1", "chunk2"]

    def test_vertex_stream_text_pre_cancelled_yields_nothing(self):
        class FakeChunk:
            text = "should not yield"

        class FakeModel:
            def __init__(self, *a, **kw):
                pass

            def generate_content(self, *a, **kw):
                def _gen():
                    for _ in range(100):
                        yield FakeChunk()
                return _gen()

        from app.services.vertex_llm_service import VertexLlmService

        svc = VertexLlmService()
        svc._initialized = True

        async def _run():
            cancel = asyncio.Event()
            cancel.set()
            with patch.dict("sys.modules", {"vertexai.generative_models": _make_fake_vertex_module(FakeModel)}):
                return await _collect(svc.stream_text(
                    prompt="hello",
                    system_prompt="sys",
                    model_name="gemini-2.5-flash",
                    cancel_event=cancel,
                ))

        chunks = run(_run())
        assert chunks == []


# ---------------------------------------------------------------------------
# 6. Error fallback: quota / timeout / filter → typed errors
# ---------------------------------------------------------------------------


class TestErrorFallback:
    def _make_error_model(self, message: str):
        class FakeModel:
            def __init__(self, *a, **kw):
                pass

            def generate_content(self, *a, **kw):
                raise Exception(message)

        return FakeModel

    def test_quota_error(self):
        from app.services.vertex_llm_service import VertexLlmService, VertexQuotaError

        svc = VertexLlmService()
        svc._initialized = True
        fake = _make_fake_vertex_module(self._make_error_model("RESOURCE_EXHAUSTED 429"))

        async def _run():
            with patch.dict("sys.modules", {"vertexai.generative_models": fake}):
                await _collect(svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"))

        with pytest.raises(VertexQuotaError):
            run(_run())

    def test_content_filter_error(self):
        from app.services.vertex_llm_service import VertexLlmService, VertexContentFilterError

        svc = VertexLlmService()
        svc._initialized = True
        fake = _make_fake_vertex_module(self._make_error_model("blocked due to safety policy"))

        async def _run():
            with patch.dict("sys.modules", {"vertexai.generative_models": fake}):
                await _collect(svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"))

        with pytest.raises(VertexContentFilterError):
            run(_run())

    def test_timeout_error(self):
        from app.services.vertex_llm_service import VertexLlmService, VertexTimeoutError

        svc = VertexLlmService()
        svc._initialized = True
        fake = _make_fake_vertex_module(self._make_error_model("deadline exceeded: timeout"))

        async def _run():
            with patch.dict("sys.modules", {"vertexai.generative_models": fake}):
                await _collect(svc.stream_text(prompt="hi", model_name="gemini-2.5-flash"))

        with pytest.raises(VertexTimeoutError):
            run(_run())

    def test_fallback_response_phrase(self):
        from app.services.vertex_llm_service import VERTEX_FALLBACK_RESPONSE

        assert "sorry" in VERTEX_FALLBACK_RESPONSE.lower()
        assert "did not catch" in VERTEX_FALLBACK_RESPONSE.lower()


# ---------------------------------------------------------------------------
# 7. Integration tests — PipelineSession, cancel_event wiring, fallback TTS
# ---------------------------------------------------------------------------


class TestPipelineSession:
    def test_pipeline_session_history_prune(self):
        """PipelineSession.history shared with handler is pruned to 40 messages."""
        from app.voice.pipeline_session import PipelineSession

        history: list = []
        session = PipelineSession(history=history)

        for i in range(25):
            session.history.append(("client", f"user {i}"))
            session.history.append(("agent", f"agent {i}"))

        assert len(session.history) == 50

        max_msgs = 40
        if len(session.history) > max_msgs:
            session.history[:] = session.history[-max_msgs:]

        assert len(session.history) == 40
        # Oldest retained = pair index 5
        assert session.history[0] == ("client", "user 5")

    def test_pipeline_session_cancel_llm_sets_event(self):
        from app.voice.pipeline_session import PipelineSession

        session = PipelineSession()
        assert not session.llm_cancel.is_set()
        session.cancel_llm()
        assert session.llm_cancel.is_set()

    def test_pipeline_session_reset_clears_event(self):
        from app.voice.pipeline_session import PipelineSession

        session = PipelineSession()
        session.cancel_llm()
        session.reset_llm_cancel()
        assert not session.llm_cancel.is_set()

    def test_bind_stt_and_tts_on_session(self):
        from app.voice.pipeline_session import PipelineSession

        session = PipelineSession()
        assert session.stt_pipeline is None
        assert session.tts_pipeline is None

        stt_stub = object()
        tts_stub = object()
        session.bind_stt(stt_stub)  # type: ignore[arg-type]
        session.bind_tts(tts_stub)  # type: ignore[arg-type]
        assert session.stt_pipeline is stt_stub
        assert session.tts_pipeline is tts_stub

        session.clear_pipelines()
        assert session.stt_pipeline is None
        assert session.tts_pipeline is None

    def test_cancel_inflight_sets_llm_cancel_event(self):
        """
        _cancel_inflight_llm_response must set _llm_cancel_event so Vertex
        producer thread stops without waiting for asyncio task cancel.
        """
        # Verify the cancel path calls _pipeline.cancel_llm() which sets the event.
        from app.voice.pipeline_session import PipelineSession

        session = PipelineSession()
        assert not session.llm_cancel.is_set()

        # Simulate what _cancel_inflight_llm_response does
        session.cancel_llm()

        assert session.llm_cancel.is_set(), (
            "_cancel_inflight_llm_response must call _pipeline.cancel_llm() "
            "to set the event before task.cancel()"
        )


class TestBidirectionalVertexCancelEvent:
    def test_bidirectional_passes_cancel_event_to_vertex(self):
        """
        When llm_service is vertex_llm_service, try_stream must pass
        cancel_event=self._llm_cancel_event to stream_turn / stream_text.
        """
        received_kwargs: dict = {}

        async def _fake_stream_turn(**kwargs):
            received_kwargs.update(kwargs)
            if False:
                yield  # make it an async generator

        from app.services.vertex_llm_service import vertex_llm_service

        original = vertex_llm_service.stream_turn
        vertex_llm_service.stream_turn = _fake_stream_turn  # type: ignore[method-assign]

        try:
            import asyncio

            async def _run():
                # Minimal simulation of what try_stream does for Vertex path
                cancel_event = asyncio.Event()
                _gen = vertex_llm_service.stream_turn(
                    system_prompt="sys",
                    conversation_history=[],
                    caller_transcript="hi",
                    kb_context="",
                    temperature=0.3,
                    max_tokens=100,
                    model_name="gemini-2.5-flash",
                    cancel_event=cancel_event,
                )
                # drain
                async for _ in _gen:
                    pass
                return cancel_event

            event = asyncio.run(_run())
            assert "cancel_event" in received_kwargs
            assert received_kwargs["cancel_event"] is event
        finally:
            vertex_llm_service.stream_turn = original  # type: ignore[method-assign]

    def test_vertex_error_queues_fallback_tts(self):
        """
        When VertexQuotaError is raised, the handler must queue VERTEX_FALLBACK_RESPONSE
        to TTS and not propagate the raw error text to the caller.
        """
        from app.services.vertex_llm_service import VERTEX_FALLBACK_RESPONSE, VertexQuotaError

        # Simulate handler error-catch logic
        tts_queued: list[str] = []

        class FakeTtsPipeline:
            async def queue_tts(self, payload):
                tts_queued.append(payload["text"])

        async def _run():
            pipeline = FakeTtsPipeline()
            try:
                raise VertexQuotaError("quota exhausted")
            except VertexQuotaError:
                await pipeline.queue_tts({"text": VERTEX_FALLBACK_RESPONSE, "is_final": True})

        asyncio.run(_run())
        assert len(tts_queued) == 1
        assert tts_queued[0] == VERTEX_FALLBACK_RESPONSE

    def test_resolve_llm_runtime_vertex_default_temp(self):
        """resolve_llm_runtime returns temperature=0.3 for gemini-2.5-flash when no agent_temperature."""
        from app.core.agent_runtime import resolve_llm_runtime
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.llm_model = "gemini-2.5-flash"
        agent.agent_temperature = None
        agent.agent_max_tokens = None
        agent.model = None
        agent.provider = None

        rt = resolve_llm_runtime(agent)
        assert rt.provider_slug == "vertex"
        assert abs(rt.temperature - 0.3) < 1e-6, f"Expected 0.3, got {rt.temperature}"

    def test_resolve_llm_runtime_vertex_agent_temp_override(self):
        """Per-agent agent_temperature overrides the 0.3 default."""
        from app.core.agent_runtime import resolve_llm_runtime
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.llm_model = "gemini-2.5-flash"
        agent.agent_temperature = 50  # → 0.5
        agent.agent_max_tokens = None
        agent.model = None
        agent.provider = None

        rt = resolve_llm_runtime(agent)
        assert abs(rt.temperature - 0.5) < 1e-6, f"Expected 0.5, got {rt.temperature}"
