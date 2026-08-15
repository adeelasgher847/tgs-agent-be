"""
RAG interim-prefetch tests for the LiveKit/browser voice-agent path.

Covers the browser-transport port of bidirectional_stream.py's
`_prefetch_rag_context` pattern (fire KB retrieval in the background on the
first qualifying STT interim, consume-once on STT final) — adapted to this
transport's own retrieval call (`kb_retrieval_service.retrieve_kb_context_for_turn`)
via:
  - LiveKitBrowserCallHandler._maybe_start_rag_prefetch /
    LiveKitBrowserCallHandler._prefetch_kb_context (fire + fail-open wrapper)
  - LiveKitBrowserCallHandler._cancel_inflight_llm_response (discard on barge-in)
  - ConversationOrchestrator.generate_and_stream_response (consume-once,
    await-in-flight, staleness check via `rag_prefetch_matches_final`)

Two layers are exercised:
  1. Handler-level: does the right thing start/reset the prefetch task.
  2. Orchestrator-level: does the right thing happen when generate_and_stream_
     response consumes whatever prefetch state the handler is in.

External HTTP/Redis/vector-DB calls are always mocked at the boundary
(`kb_retrieval_service.retrieve_kb_context_for_turn`, `redis_client.get_redis`)
per repo convention — never hit real infra.
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.conversation_orchestrator import (
    ConversationOrchestrator,
    rag_prefetch_matches_final,
)
from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler
from tests.voice.test_bracket_tag_prompt_consistency import _fake_livekit_handler


FAKE_KB_FACT = "TEST_KB_FACT_PREFETCH_98765"
FAKE_KB_BLOCK = (
    "--- KNOWLEDGE BASE CONTEXT ---\n"
    f"{FAKE_KB_FACT}\n"
    "--- END CONTEXT ---"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _real_handler_with_kb(kb_ids) -> LiveKitBrowserCallHandler:
    """A real (non-mock) handler instance — needed for tests that exercise
    the actual `_maybe_process_interim` / `_cancel_inflight_llm_response`
    state machine, not just prompt-building."""
    db = MagicMock()
    call_session = MagicMock()
    call_session.id = uuid.uuid4()
    call_session.tenant_id = uuid.uuid4()
    call_session.call_metadata = {}
    call_session.call_transcript = []

    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.name = "Demo Agent"
    agent.language = "en"

    flow = MagicMock()
    flow.knowledge_base_ids = [str(k) for k in kb_ids]

    return LiveKitBrowserCallHandler(db=db, call_session=call_session, agent=agent, call_flow=flow)


def _handler_with_kb_for_prompt(kb_ids):
    """Mock-based handler for orchestrator/prompt-building tests — mirrors
    tests/voice/test_livekit_kb_grounding.py's `_handler_with_kb`."""
    h = _fake_livekit_handler(tts_slug="google")
    h.db = MagicMock()  # truthy — required for the KB-retrieval branch to run
    flow = MagicMock()
    flow.knowledge_base_ids = [str(k) for k in kb_ids]
    flow.caller_memory_enabled = False
    h.call_flow = flow
    h._rag_prefetch_task = None
    h._rag_prefetch_source_text = ""
    return h


async def _run_and_capture(orchestrator, monkeypatch, user_text: str) -> list[str]:
    """Async (same-event-loop) variant of test_livekit_kb_grounding.py's
    `_run_and_capture` — must run in the SAME loop as any prefetch task the
    test creates, so it awaits directly instead of calling asyncio.run()."""
    captured: list[str] = []

    async def _stub_stream(**kwargs):
        captured.append(kwargs.get("system_prompt") or "")
        yield "Sure, here is a short reply."

    stub_service = MagicMock()
    stub_service.stream_text = _stub_stream
    monkeypatch.setattr(
        "app.core.agent_runtime.llm_service_for_provider", lambda slug: stub_service
    )
    await orchestrator.generate_and_stream_response(user_text, 0.9)
    return captured


# ─────────────────────────────────────────────────────────────────────────────
# rag_prefetch_matches_final — the staleness-check helper
# ─────────────────────────────────────────────────────────────────────────────

class TestRagPrefetchMatchesFinal:
    def test_identical_text_matches(self):
        assert rag_prefetch_matches_final("what are your hours", "what are your hours")

    def test_final_extends_interim_matches(self):
        # Common case: interim is a truncated prefix of the eventual final.
        assert rag_prefetch_matches_final("what are your", "what are your hours today")

    def test_materially_different_text_does_not_match(self):
        assert not rag_prefetch_matches_final(
            "what is your refund policy", "actually cancel my subscription entirely"
        )

    def test_empty_source_or_final_does_not_match(self):
        assert not rag_prefetch_matches_final("", "hello")
        assert not rag_prefetch_matches_final("hello", "")


# ─────────────────────────────────────────────────────────────────────────────
# Handler-level: firing / gating / reset
# ─────────────────────────────────────────────────────────────────────────────

class TestHandlerPrefetchTrigger:
    @pytest.mark.asyncio
    async def test_interim_starts_rag_prefetch_when_kb_configured(self):
        kb_id = uuid.uuid4()
        h = _real_handler_with_kb([kb_id])

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0)),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            await h._maybe_process_interim("what are your hours today", 0.9)
            assert h._rag_prefetch_task is not None
            assert isinstance(h._rag_prefetch_task, asyncio.Task)
            await h._rag_prefetch_task  # drain so the test doesn't leak a task

    @pytest.mark.asyncio
    async def test_no_prefetch_when_kb_not_configured(self):
        """RAG disabled for this call flow (no knowledge_base_ids) — the
        interim handler must never fire a background retrieval at all."""
        h = _real_handler_with_kb([])  # empty KB list == RAG disabled
        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ):
            await h._maybe_process_interim("what are your hours today", 0.9)

        assert h._rag_prefetch_task is None
        retrieve_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prefetch_when_call_flow_missing(self):
        h = _real_handler_with_kb([uuid.uuid4()])
        h.call_flow = None  # no flow at all
        await h._maybe_process_interim("what are your hours today", 0.9)
        assert h._rag_prefetch_task is None

    @pytest.mark.asyncio
    async def test_second_interim_does_not_fire_duplicate_prefetch(self):
        """Guards the 'never create duplicate RAG requests for the same
        utterance' requirement — multiple qualifying interims in one turn
        must only trigger ONE retrieval call."""
        kb_id = uuid.uuid4()
        h = _real_handler_with_kb([kb_id])
        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            await h._maybe_process_interim("what are your hours", 0.9)
            first_task = h._rag_prefetch_task
            await h._maybe_process_interim("what are your hours today please", 0.92)
            second_task = h._rag_prefetch_task

            assert first_task is second_task  # same task instance — no re-fire
            await first_task

        retrieve_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_low_confidence_or_short_interim_does_not_fire_prefetch(self):
        kb_id = uuid.uuid4()
        h = _real_handler_with_kb([kb_id])
        h._rag_prefetch_min_words = 3
        h._rag_prefetch_min_confidence = 0.5
        await h._maybe_process_interim("uh", 0.9)  # below min_words
        assert h._rag_prefetch_task is None
        await h._maybe_process_interim("what are your hours", 0.1)  # below min_confidence
        assert h._rag_prefetch_task is None

    @pytest.mark.asyncio
    async def test_barge_in_cancels_and_resets_pending_prefetch(self):
        """Cancellation: an in-flight prefetch must be torn down (not just
        abandoned) when the turn is barge-in-cancelled, so the next turn
        always starts from a clean slate — mirrors bidirectional_stream.py's
        _cancel_inflight_llm_response discarding _rag_prefetch_task."""
        h = _real_handler_with_kb([uuid.uuid4()])
        pending = asyncio.create_task(asyncio.sleep(5))
        h._rag_prefetch_task = pending
        h._rag_prefetch_source_text = "some interim text"

        await h._cancel_inflight_llm_response()
        await asyncio.sleep(0)  # let the cancellation propagate

        assert h._rag_prefetch_task is None
        assert h._rag_prefetch_source_text == ""
        assert pending.cancelled()

    @pytest.mark.asyncio
    async def test_barge_in_qualifying_interim_never_wastefully_fires_prefetch(self):
        """
        Regression test for the ordering bug: firing the prefetch BEFORE
        resolving barge-in meant every barge-in-initiated turn fired a
        prefetch on the very same call that immediately cancelled it via
        `_cancel_inflight_llm_response()` — wasted embedding/vector-DB work,
        and the highest-value case (a user interrupting the agent) got zero
        prefetch benefit. Drives `_maybe_process_interim` end-to-end (not
        pre-seeding `_rag_prefetch_task`) with `_is_tts_playing=True` and
        barge-in-qualifying text, and asserts the prefetch retrieval was
        NEVER called on that call at all — not "fired then cancelled".
        """
        kb_id = uuid.uuid4()
        h = _real_handler_with_kb([kb_id])
        h._is_tts_playing = True
        h._cancel_inflight_llm_response = AsyncMock(wraps=h._cancel_inflight_llm_response)

        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            # Qualifies for barge-in: word_count >= _barge_in_min_words (2)
            # and confidence >= _barge_in_min_conf (~0.26) by a wide margin —
            # and, crucially, ALSO qualifies for the (much weaker) RAG
            # prefetch gate, which is exactly the scenario the bug required.
            await h._maybe_process_interim("please stop talking now", 0.9)

        h._cancel_inflight_llm_response.assert_awaited_once()
        retrieve_mock.assert_not_called()
        assert h._rag_prefetch_task is None

        # On a LATER interim, once barge-in has already been resolved for
        # this call (TTS no longer playing after the cancel above), the
        # prefetch trigger must still work normally and survive to be
        # consumed — proves the fix doesn't just suppress prefetch outright,
        # only reorders when it's allowed to fire.
        h._is_tts_playing = False
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            await h._maybe_process_interim("actually never mind continue", 0.9)
            assert h._rag_prefetch_task is not None
            # Await the task INSIDE the patch context — the task body's own
            # local import + retrieval call must still see the mock; awaiting
            # after the patch has been reverted would spuriously hit the real
            # (unpatched) retrieval function instead.
            await h._rag_prefetch_task

        retrieve_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_completed_prefetch_also_reset_on_cancel(self):
        """Even an already-completed (not just pending) prefetch must be
        cleared on cancel — the barge-in reset path is unconditional."""
        h = _real_handler_with_kb([uuid.uuid4()])

        async def _done():
            return FAKE_KB_BLOCK, 1.0

        done_task = asyncio.create_task(_done())
        await done_task
        h._rag_prefetch_task = done_task
        h._rag_prefetch_source_text = "already finished"

        await h._cancel_inflight_llm_response()

        assert h._rag_prefetch_task is None
        assert h._rag_prefetch_source_text == ""


class TestPrefetchKbContextFailsOpen:
    """`_prefetch_kb_context` (the background coroutine itself) must never
    raise — timeouts and retrieval errors both degrade to an empty result,
    mirroring `_prefetch_rag_context`'s fail-open contract on Twilio."""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_result(self, monkeypatch):
        h = _real_handler_with_kb([uuid.uuid4()])
        monkeypatch.setattr("app.core.config.settings.RAG_KB_RETRIEVAL_TIMEOUT_SEC", 0.01)

        async def _slow(**kwargs):
            await asyncio.sleep(0.2)
            return FAKE_KB_BLOCK, 200.0

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=_slow
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            result = await h._prefetch_kb_context("hours?", [str(uuid.uuid4())])

        assert result == ("", 0.0)

    @pytest.mark.asyncio
    async def test_retrieval_exception_returns_empty_result(self):
        h = _real_handler_with_kb([uuid.uuid4()])
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            result = await h._prefetch_kb_context("hours?", [str(uuid.uuid4())])

        assert result == ("", 0.0)

    @pytest.mark.asyncio
    async def test_no_kb_results_returns_empty_context(self):
        """RAG ran successfully but found nothing relevant — a legitimate
        empty result, distinct from an error/timeout."""
        h = _real_handler_with_kb([uuid.uuid4()])
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn",
            new=AsyncMock(return_value=("", 12.0)),
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            result = await h._prefetch_kb_context("hours?", [str(uuid.uuid4())])

        assert result == ("", 12.0)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-level: consume-once / await-in-flight / staleness / no-prefetch
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorConsumesPrefetch:
    @pytest.mark.asyncio
    async def test_final_reuses_completed_prefetch_without_new_call(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb_for_prompt([kb_id])
        orchestrator = ConversationOrchestrator(h)

        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            task = asyncio.create_task(
                retrieve_mock(transcript="what are your hours", kb_ids=[str(kb_id)], redis_client=None)
            )
            await task  # already finished by the time "final" arrives
            h._rag_prefetch_task = task
            h._rag_prefetch_source_text = "what are your hours"

            captured = await _run_and_capture(orchestrator, monkeypatch, "what are your hours")

        assert FAKE_KB_FACT in captured[0]
        # One call to seed the "prefetch", zero additional calls at final time.
        retrieve_mock.assert_awaited_once()
        assert h._rag_prefetch_task is None  # consumed exactly once
        assert h._rag_prefetch_source_text == ""

    @pytest.mark.asyncio
    async def test_final_awaits_inflight_prefetch_without_duplicate_call(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb_for_prompt([kb_id])
        orchestrator = ConversationOrchestrator(h)

        call_count = 0

        async def _slow_retrieve(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return FAKE_KB_BLOCK, 50.0

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=_slow_retrieve
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            task = asyncio.create_task(
                LiveKitBrowserCallHandler._prefetch_kb_context(h, "what are your hours", [str(kb_id)])
            )
            h._rag_prefetch_task = task
            h._rag_prefetch_source_text = "what are your hours"

            captured = await _run_and_capture(orchestrator, monkeypatch, "what are your hours")

        assert call_count == 1, "must await the SAME in-flight task, never start a second parallel retrieval"
        assert FAKE_KB_FACT in captured[0]

    @pytest.mark.asyncio
    async def test_stale_prefetch_not_reused_for_materially_different_final(self, monkeypatch):
        kb_id = uuid.uuid4()
        h = _handler_with_kb_for_prompt([kb_id])
        orchestrator = ConversationOrchestrator(h)

        stale_block = "--- KNOWLEDGE BASE CONTEXT ---\nSTALE_MARKER_SHOULD_NOT_APPEAR\n--- END CONTEXT ---"

        async def _stale_coro():
            return stale_block, 5.0

        stale_task = asyncio.create_task(_stale_coro())
        await stale_task  # completed — but for the WRONG (interim) text

        h._rag_prefetch_task = stale_task
        h._rag_prefetch_source_text = "what is your refund policy"  # interim seed text

        fresh_calls = []

        async def _fresh_retrieve(**kwargs):
            fresh_calls.append(kwargs.get("transcript"))
            return FAKE_KB_BLOCK, 5.0

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=_fresh_retrieve
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            # Final utterance diverges materially from the interim above.
            captured = await _run_and_capture(
                orchestrator, monkeypatch, "actually cancel my subscription entirely"
            )

        assert "STALE_MARKER_SHOULD_NOT_APPEAR" not in captured[0]
        assert FAKE_KB_FACT in captured[0]
        assert fresh_calls == ["actually cancel my subscription entirely"]

    @pytest.mark.asyncio
    async def test_no_prefetch_fired_falls_back_to_synchronous_retrieval(self, monkeypatch):
        """Existing behavior when prefetch conditions weren't met (e.g. a
        very short first utterance never fired one) — must still work,
        unmodified, via the synchronous fallback branch."""
        kb_id = uuid.uuid4()
        h = _handler_with_kb_for_prompt([kb_id])
        orchestrator = ConversationOrchestrator(h)
        assert h._rag_prefetch_task is None

        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            captured = await _run_and_capture(orchestrator, monkeypatch, "what are your hours")

        assert FAKE_KB_FACT in captured[0]
        retrieve_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rag_disabled_no_kb_ids_skips_prefetch_consumption_entirely(self, monkeypatch):
        h = _handler_with_kb_for_prompt([])  # no KB attached == RAG disabled
        orchestrator = ConversationOrchestrator(h)

        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ):
            captured = await _run_and_capture(orchestrator, monkeypatch, "what are your hours")

        retrieve_mock.assert_not_called()
        assert "--- KNOWLEDGE BASE CONTEXT ---" not in captured[0]

    @pytest.mark.asyncio
    async def test_new_utterance_prefetch_not_confused_with_prior_turns(self, monkeypatch):
        """A new utterance's own (unrelated) prefetch must be consumed on
        ITS final, never accidentally reused by / bled into a different
        turn — the consume-once-and-null contract applies per turn."""
        kb_id = uuid.uuid4()
        h = _handler_with_kb_for_prompt([kb_id])
        orchestrator = ConversationOrchestrator(h)

        retrieve_mock = AsyncMock(return_value=(FAKE_KB_BLOCK, 5.0))
        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            # Turn 1: no prefetch fired, synchronous fallback used.
            await _run_and_capture(orchestrator, monkeypatch, "what are your hours")
            assert h._rag_prefetch_task is None

            # Turn 2: a fresh prefetch fires and is consumed cleanly, with no
            # leftover state from turn 1 interfering.
            task = asyncio.create_task(
                LiveKitBrowserCallHandler._prefetch_kb_context(h, "what is your address", [str(kb_id)])
            )
            h._rag_prefetch_task = task
            h._rag_prefetch_source_text = "what is your address"
            captured2 = await _run_and_capture(orchestrator, monkeypatch, "what is your address")

        assert FAKE_KB_FACT in captured2[0]
        assert h._rag_prefetch_task is None
        assert retrieve_mock.await_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency proof (Part 5): prefetch must not block continued STT/audio
# handling while the retrieval is "in flight".
# ─────────────────────────────────────────────────────────────────────────────

class TestRagPrefetchConcurrency:
    @pytest.mark.asyncio
    async def test_prefetch_task_runs_concurrently_with_continued_stt_handling(self):
        """
        Proves _maybe_start_rag_prefetch fires a real background task (not a
        blocking call) — a concurrently scheduled "STT/audio handling" ticker
        must keep making progress while the (simulated network) KB retrieval
        is in flight. Same non-blocking-interleaving pattern used for the
        OpenAI/Groq async-streaming regression tests.

        Deliberately load-independent: rather than asserting on a total
        wall-clock budget (which flaked under a full-suite run with hundreds
        of tests contending for the CPU/event loop — a fixed absolute
        threshold has no safe margin against arbitrary scheduler jitter),
        this asserts on the *relative* interleaving of ticker iterations
        against the retrieval's own start/end timestamps, all measured in
        the same event loop. If the prefetch blocked the loop while "in
        flight" (the old-bug equivalent), zero ticks could land inside that
        window regardless of how fast or slow the machine is; if it doesn't
        block, several ticks land inside it — that's true under both a fast
        and a heavily-loaded run.
        """
        kb_id = uuid.uuid4()
        h = _real_handler_with_kb([kb_id])
        loop = asyncio.get_event_loop()

        retrieve_start: float | None = None
        retrieve_end: float | None = None

        async def _slow_retrieve(**kwargs):
            nonlocal retrieve_start, retrieve_end
            retrieve_start = loop.time()
            await asyncio.sleep(0.2)
            retrieve_end = loop.time()
            return FAKE_KB_BLOCK, 200.0

        tick_times: list[float] = []

        async def ticker():
            for _ in range(6):
                await asyncio.sleep(0.03)
                tick_times.append(loop.time())

        with patch(
            "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=_slow_retrieve
        ), patch("app.utils.redis_client.get_redis", return_value=None):
            # Mirrors what _maybe_process_interim does: fire-and-forget, must
            # return immediately without awaiting the retrieval itself.
            h._maybe_start_rag_prefetch("what are your hours today", 0.9, 5)
            assert h._rag_prefetch_task is not None

            await asyncio.gather(h._rag_prefetch_task, ticker())

        assert len(tick_times) == 6
        assert retrieve_start is not None and retrieve_end is not None

        ticks_during_retrieval = [t for t in tick_times if retrieve_start <= t <= retrieve_end]
        assert len(ticks_during_retrieval) >= 3, (
            "expected several ticker iterations to interleave with the "
            f"in-flight retrieval, got {len(ticks_during_retrieval)} of "
            f"{len(tick_times)} — the prefetch may be blocking the event loop"
        )
