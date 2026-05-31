"""
Regression tests: bidirectional interim → final (one interim LLM per turn, barge-in, regen).
Remove this file if you prefer integration-only coverage.

Run: pytest tests/services/test_bidirectional_interim_regression.py -q
"""

from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler


# --- _should_regenerate_on_final (no full __init__) ---


def _make_handler_regen(seed: str) -> SimpleNamespace:
    h = SimpleNamespace()
    h._turn_response_seed_text = seed
    h._last_offered_calendar_slots = []
    h._last_requested_calendar_date = None
    h._last_selected_calendar_slot = None
    h._is_booking_intent_turn = types.MethodType(Handler._is_booking_intent_turn, h)
    h._is_booking_context_active = types.MethodType(Handler._is_booking_context_active, h)
    h._resolve_cached_calendar_slot = types.MethodType(Handler._resolve_cached_calendar_slot, h)
    h._normalize_turn_text = Handler._normalize_turn_text
    h._normalize_calendar_slot_key = Handler._normalize_calendar_slot_key
    h._is_natural_continuation_of_seed = types.MethodType(
        Handler._is_natural_continuation_of_seed, h
    )
    return h


def test_regenerate_true_when_final_text_differs() -> None:
    h = _make_handler_regen("hello I need")
    assert Handler._should_regenerate_on_final(h, "hello I need a refund") is True


def test_regenerate_false_when_final_extends_utterance() -> None:
    """Vapi-style: user was still dictating; one LLM+TTS is enough (no second reply)."""
    h = _make_handler_regen("I need to schedule an")
    assert Handler._should_regenerate_on_final(h, "I need to schedule an appointment.") is False


def test_regenerate_false_on_stt_word_level_revision() -> None:
    """
    STT corrects a single mishear at the end (e.g. Carlton → Carter). Same intent,
    re-running the LLM just produces a second "Nice to meet you, …" and that's the
    double-audio the user complained about. Skip regen.
    """
    h = _make_handler_regen("Hello my name is Alex Carlton")
    assert Handler._should_regenerate_on_final(h, "Hello my name is Alex Carter.") is False


def test_regenerate_true_when_new_intent_added_at_end() -> None:
    """Guardrail: word-revision check must not silence genuine new intent."""
    h = _make_handler_regen("I want to book an appointment")
    assert (
        Handler._should_regenerate_on_final(h, "I want to cancel my appointment and get a refund")
        is True
    )


def test_regenerate_false_when_final_matches_seed() -> None:
    h = _make_handler_regen("how are you today")
    assert Handler._should_regenerate_on_final(h, "How are you today") is False


def test_regenerate_true_when_final_empty() -> None:
    h = _make_handler_regen("something")
    assert Handler._should_regenerate_on_final(h, "") is True


# --- _maybe_process_interim (lightweight __new__ instances) ---


def _empty_handler() -> Handler:
    h = object.__new__(Handler)
    h._turn_response_started = False
    h._turn_response_seed_text = ""
    h._last_interim_text = ""
    h._last_interim_sent_ts = 0.0
    h._enable_interim_llm = True  # tests exercise the optional early-LLM path
    h._min_interim_words = 1
    h._min_interim_confidence = 0.4
    h._min_interim_interval_sec = 0.0
    h._tts_pipeline = None
    h._llm_response_task = None
    h._rag_prefetch_task = None
    h._rag_prefetch_user_text = ""
    h.is_speaking = False
    h._barge_in_min_conf = 0.26
    h._barge_in_min_conf_1w = 0.52
    h._stt_min_final_confidence = 0.26
    h._enable_soft_final_fallback = True
    h._stt_soft_min_final_confidence = 0.16
    h._stt_soft_min_words = 2
    h._prefetch_rag_context = AsyncMock(return_value=("", {}))  # type: ignore[method-assign]
    h._llm_turn_serial_lock = asyncio.Lock()
    return h


def test_second_interim_does_not_start_second_llm() -> None:
    """A second, longer interim must not invoke generate again in the same turn."""

    async def _body() -> None:
        calls: list[str] = []

        async def fake_generate(
            self,
            user_text: str,
            confidence: float,
            is_greeting: bool = False,
        ) -> None:
            calls.append(user_text)

        h = _empty_handler()
        h._should_defer_interim_response = lambda _t: False  # type: ignore[method-assign]

        with patch.object(Handler, "generate_and_stream_response", new=fake_generate):
            await Handler._maybe_process_interim(h, "hello I", 0.5)
            await asyncio.sleep(0)
            await Handler._maybe_process_interim(h, "hello I need", 0.5)
            await asyncio.sleep(0)

        assert h._turn_response_started is True
        assert len(calls) == 1
        assert calls[0] == "hello I"

    asyncio.run(_body())


def test_interim_llm_off_skips_generate() -> None:
    """Default prod path: VOICE_ENABLE_INTERIM_LLM=False — no LLM on partials."""

    async def _body() -> None:
        calls: list[str] = []

        async def fake_generate(
            self,
            user_text: str,
            confidence: float,
            is_greeting: bool = False,
        ) -> None:
            calls.append(user_text)

        h = _empty_handler()
        h._enable_interim_llm = False
        h._should_defer_interim_response = lambda _t: False  # type: ignore[method-assign]

        with patch.object(Handler, "generate_and_stream_response", new=fake_generate):
            await Handler._maybe_process_interim(h, "hello I need help now please", 0.9)

        assert calls == []

    asyncio.run(_body())


def test_barge_in_clears_turn_no_generate() -> None:
    """
    While the agent TTS is active, a one-word user interim triggers cancel, not a new LLM.
    """

    async def _body() -> None:
        calls: list[str] = []

        async def fake_generate(self, *a, **k) -> None:  # noqa: ARG001
            calls.append("x")

        h = _empty_handler()
        h._turn_response_started = True
        h._turn_response_seed_text = "seed"
        h._last_interim_text = "x"
        tts = SimpleNamespace(
            is_speaking=True,
            cancel_current_and_clear_queue=AsyncMock(),
        )
        h._tts_pipeline = tts
        h._should_defer_interim_response = lambda _t: False  # type: ignore[method-assign]
        h._cancel_inflight_llm_response = types.MethodType(Handler._cancel_inflight_llm_response, h)

        with patch.object(Handler, "generate_and_stream_response", new=fake_generate):
            await Handler._maybe_process_interim(h, "no", 0.9)

        tts.cancel_current_and_clear_queue.assert_awaited_once()
        assert h._turn_response_started is False
        assert h._turn_response_seed_text == ""
        assert h._last_interim_text == ""
        assert calls == []

    asyncio.run(_body())


def test_complete_final_regen_calls_generate_once() -> None:
    async def _body() -> None:
        h = _empty_handler()
        h._turn_response_started = True
        h._turn_response_seed_text = "short"
        h._llm_response_task = None
        h._last_interim_text = "x"
        h._tts_cancel = asyncio.Event()
        h._tts_pipeline = SimpleNamespace(cancel_current_and_clear_queue=AsyncMock())
        h._should_regenerate_on_final = types.MethodType(lambda _self, _ft: True, h)  # type: ignore[misc,assignment]
        h.generate_and_stream_response = AsyncMock()  # type: ignore[method-assign]
        h._cancel_inflight_llm_response = types.MethodType(Handler._cancel_inflight_llm_response, h)

        await Handler._complete_llm_turn_after_stt_final(h, "a longer final utterance", 0.8)

        assert h._turn_response_started is False
        h.generate_and_stream_response.assert_awaited_once()  # type: ignore[attr-defined]
        assert h.generate_and_stream_response.call_args[0][0] == "a longer final utterance"  # type: ignore[attr-defined]
        assert h.generate_and_stream_response.call_args[1].get("is_greeting") is False  # type: ignore[attr-defined]

    asyncio.run(_body())


def test_should_accept_final_transcript_allows_soft_multword() -> None:
    h = _empty_handler()
    assert Handler._should_accept_final_transcript(h, "yes i need help", 0.18) is True


def test_should_accept_final_transcript_rejects_soft_filler() -> None:
    h = _empty_handler()
    assert Handler._should_accept_final_transcript(h, "uh huh", 0.20) is False
