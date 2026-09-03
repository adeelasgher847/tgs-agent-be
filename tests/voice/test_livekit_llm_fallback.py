"""
A-05 follow-up: LiveKit browser-call LLM-failure fallback regression test.

``ConversationOrchestrator.generate_and_stream_response()`` (used by the
LiveKit browser-call transport, ``LiveKitBrowserCallHandler``) previously
left ``final_text = ""`` and produced no TTS output at all when the LLM
stream call raised — i.e. silent dead air for the caller, unlike the Twilio
transport (``BidirectionalStreamHandler``) which always speaks a canned
fallback message on LLM failure.

This covers the fix: on an LLM streaming exception, the orchestrator now
queues a non-empty, ``is_final=True`` fallback TTS chunk (built from
``settings.VOICE_LLM_FALLBACK_MESSAGE``) instead of staying silent, and
still records that fallback text to the transcript.

Uses the same duck-typed fake-handler harness as
``tests/voice/test_bracket_tag_prompt_consistency.py`` — reused here rather
than re-invented, per that module's own docstring about
``generate_and_stream_response()`` only reading/writing attributes on
``self._h``.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.voice.conversation_orchestrator import ConversationOrchestrator
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler


def _patched_circuit_breaker():
    """A-05's ``llm_circuit_breaker`` is Redis-backed (real Redis in this
    env, per ``settings.REDIS_URL`` — not a local fake). Always patch it out
    here: leaving it live would (a) make these tests reach a real external
    service, forbidden regardless of outcome, and (b) let repeated
    LLM-failure tests actually trip the shared "openai" circuit OPEN,
    polluting unrelated tests that assume the primary provider is always
    called (e.g. ``test_bracket_tag_prompt_consistency.py``). Always report
    "can execute" (primary path, not the breaker's secondary-swap branch) —
    this test only cares about the fallback-on-exception behavior, which
    fires identically regardless of which provider raised.
    """
    return patch.multiple(
        "app.voice.conversation_orchestrator.llm_circuit_breaker",
        can_execute=AsyncMock(return_value=True),
        record_success=AsyncMock(),
        record_failure=AsyncMock(),
    )


def _raising_stub_service():
    """A fake LLM service whose stream_text() is an async generator that
    raises before yielding anything — simulates the LLM call failing
    (whether as a primary-provider failure or after the circuit breaker's
    secondary-provider swap also fails; the fallback logic doesn't care
    which)."""

    class _StubService:
        async def stream_text(self, **kwargs):
            if False:  # pragma: no cover - keeps this an async generator
                yield ""
            raise RuntimeError("simulated LLM streaming failure")

    return _StubService()


class TestLiveKitLLMFallbackOnStreamFailure:
    def _run(self, orchestrator, monkeypatch):
        stub_service = _raising_stub_service()
        monkeypatch.setattr(
            "app.core.agent_runtime.llm_service_for_provider", lambda slug: stub_service
        )
        with _patched_circuit_breaker():
            asyncio.run(orchestrator.generate_and_stream_response("Hello there", 0.9))

    def test_llm_failure_queues_nonempty_final_fallback_tts(self, monkeypatch):
        h = _fake_livekit_handler(tts_slug="elevenlabs")
        orchestrator = ConversationOrchestrator(h)

        self._run(orchestrator, monkeypatch)

        # No silent dead air: queue_tts must have been called at least once.
        assert h._tts_pipeline.queue_tts.await_count >= 1

        # The (only, in this failure path) queued chunk is the fallback:
        # non-empty text and is_final=True, matching the Twilio transport's
        # "always end with something spoken" convention.
        call_args = h._tts_pipeline.queue_tts.await_args_list[-1]
        queued_task = call_args.args[0]
        assert queued_task["is_final"] is True
        assert queued_task["text"]
        assert queued_task["text"].strip() != ""

        expected_fallback = (
            settings.VOICE_LLM_FALLBACK_MESSAGE
            or "I am sorry, I did not catch that. Could you please repeat that?"
        )
        assert queued_task["text"] == expected_fallback

    def test_llm_failure_still_records_fallback_to_transcript(self, monkeypatch):
        """final_text is set to the fallback text so downstream transcript
        recording (``if final_text:``) still runs instead of being skipped."""
        h = _fake_livekit_handler(tts_slug="elevenlabs")
        orchestrator = ConversationOrchestrator(h)

        self._run(orchestrator, monkeypatch)

        h._add_to_transcript.assert_awaited()
        last_call = h._add_to_transcript.await_args_list[-1]
        role, text = last_call.args[0], last_call.args[1]
        assert role == "agent"
        assert text.strip() != ""

    def test_llm_failure_skips_fallback_tts_when_barge_in_cancelled(self, monkeypatch):
        """Matches the same guard used everywhere else in this method: if
        the caller barge-in-cancels the turn (setting ``_tts_cancel``)
        concurrently with the LLM failing, the fallback chunk must not be
        queued into an already-cancelled turn.

        ``_tts_cancel`` is cleared unconditionally at the top of
        ``generate_and_stream_response`` (fresh turn), so the cancel has to
        be set from inside the failing stream call itself to land after
        that reset and still be observed by the ``except`` block's guard.
        """
        h = _fake_livekit_handler(tts_slug="elevenlabs")
        orchestrator = ConversationOrchestrator(h)

        class _CancellingStubService:
            async def stream_text(self, **kwargs):
                h._tts_cancel.set()
                if False:  # pragma: no cover - keeps this an async generator
                    yield ""
                raise RuntimeError("simulated LLM streaming failure")

        monkeypatch.setattr(
            "app.core.agent_runtime.llm_service_for_provider",
            lambda slug: _CancellingStubService(),
        )
        with _patched_circuit_breaker():
            asyncio.run(orchestrator.generate_and_stream_response("Hello there", 0.9))

        h._tts_pipeline.queue_tts.assert_not_awaited()
