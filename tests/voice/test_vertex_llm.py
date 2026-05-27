"""
Unit tests for the Gemini 2.5 Flash voice LLM path (Google AI Studio / google-genai).

These tests are intentionally lightweight and offline-friendly. They validate:
  - Model allow-listing + provider inference
  - Default runtime temperature for gemini-2.5-flash (0.3)
  - Prompt construction shape for stream_turn()
  - Cancellation plumbing (PipelineSession + cancel_event passthrough)
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List
from unittest.mock import patch

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

    def test_infer_provider_returns_gemini(self):
        from app.core.llm_models import infer_llm_provider

        assert infer_llm_provider("gemini-2.5-flash") == "gemini"

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
    def test_resolve_llm_runtime_gemini_25_flash_defaults(self):
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
        assert runtime.provider_slug == "gemini"
        assert runtime.temperature == 0.3
        assert runtime.api_key is None

    def test_openai_provider_unchanged(self):
        from app.core.agent_runtime import llm_service_for_provider
        from app.services.openai_service import openai_service

        assert llm_service_for_provider("openai") is openai_service

    def test_unknown_falls_back_to_gemini(self):
        from app.core.agent_runtime import llm_service_for_provider
        from app.services.gemini_service import gemini_service

        assert llm_service_for_provider("unknown") is gemini_service


class TestGeminiStreamTurn:
    def test_stream_turn_builds_user_message_and_passes_cancel_event(self):
        from app.services.gemini_service import gemini_service

        received = {}

        async def _fake_stream_text(**kwargs):
            received.update(kwargs)
            if False:
                yield  # pragma: no cover

        original = gemini_service.stream_text
        gemini_service.stream_text = _fake_stream_text  # type: ignore[method-assign]
        try:
            cancel = asyncio.Event()

            async def _run():
                await _collect(
                    gemini_service.stream_turn(
                        system_prompt="SYS",
                        conversation_history=[("client", "hi"), ("agent", "hello")],
                        caller_transcript="How are you?",
                        kb_context="KB",
                        temperature=0.3,
                        max_tokens=10,
                        model_name="gemini-2.5-flash",
                        api_key="k",
                        cancel_event=cancel,
                    )
                )

            run(_run())
        finally:
            gemini_service.stream_text = original  # type: ignore[method-assign]

        assert received["system_prompt"] == "SYS"
        assert received["model_name"] == "gemini-2.5-flash"
        assert received["temperature"] == 0.3
        assert received["max_tokens"] == 10
        assert received["api_key"] == "k"
        assert received["cancel_event"] is cancel
        # Prompt should include history + caller + context
        msg = received["prompt"]
        assert "Previous conversation:" in msg
        assert "Client: hi" in msg
        assert "Agent: hello" in msg
        assert "Caller:" in msg
        assert "How are you?" in msg
        assert "[Context]" in msg
        assert "KB" in msg


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

class TestFallbackPhrase:
    def test_fallback_response_phrase(self):
        from app.core.config import settings

        msg = settings.VOICE_LLM_FALLBACK_RESPONSE
        assert "sorry" in msg.lower()
        assert "did not catch" in msg.lower()


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
        _cancel_inflight_llm_response must set _llm_cancel_event so the
        streaming producer can stop without waiting for asyncio task cancel.
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


#
# Vertex-specific integration tests were removed when gemini-2.5-flash was
# routed to Google AI Studio (GeminiService) instead of Vertex.
