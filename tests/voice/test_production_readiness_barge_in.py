"""
Production Readiness Verification Test Suite for 3-State Barge-In & Backchannel Architecture.

Verifies:
1. Known Backchannels (Suppressed while TTS is playing; no LLM, RAG, speculative TTS, or audio cancellation).
2. Explicit Interruptions (Trigger barge-in, cancel active TTS immediately).
3. Legitimate Short / Unknown Requests (Never silently dropped; barge in and generate LLM response).
4. Interim -> Final STT Deduplication (No duplicate LLM generation on final STT).
5. Race conditions (TTS finishes between interim and final STT; rapid arrivals).
6. Telephony (Twilio) and Browser (LiveKit) behavioral parity.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.voice.backchannel_classifier import (
    TurnClassification,
    classify_turn,
)
from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler


class DummyWebSocket:
    async def send_text(self, data: str) -> None:
        pass

    async def send_json(self, data: dict) -> None:
        pass

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass


def _create_test_handler() -> BidirectionalStreamHandler:
    handler = BidirectionalStreamHandler(
        websocket=DummyWebSocket(),
        call_session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        db=None,
    )
    handler.call_session = MagicMock()
    handler.call_session.id = uuid.uuid4()
    handler.call_session.tenant_id = uuid.uuid4()
    handler.agent = MagicMock()
    handler.agent.language = "en"
    handler._enable_interim_llm = False
    handler._barge_in_min_words = 2
    handler._barge_in_min_conf = 0.26
    handler._barge_in_min_conf_1w = 0.36
    handler._barge_in_dead_zone_ms = 600

    handler._cancel_inflight_llm_response = AsyncMock()
    handler._complete_llm_turn_after_stt_final = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._add_to_transcript = AsyncMock()
    handler._send_in_progress_status = AsyncMock()
    handler._check_and_end_call_if_goodbye = AsyncMock(return_value=False)
    handler._check_and_end_call_if_voicemail = AsyncMock(return_value=False)
    handler._check_and_handle_call_screener = AsyncMock(return_value=False)
    handler._check_and_handle_ivr_and_hold = AsyncMock(return_value=False)
    handler._check_and_handle_anti_bot = AsyncMock(return_value=False)
    handler._check_and_handle_compliance_monitoring = AsyncMock(return_value=False)
    handler._send_quick_acknowledgement = AsyncMock()
    return handler


# ── Focus 1: Known Backchannels ──────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "Hey",
        "Hi",
        "Hey hi",
        "Hey there",
        "Okay",
        "Okay and",
        "Yeah yeah",
        "Thank you",
        "Got it",
        "Sure",
        "Alright",
        "Good morning",
    ],
)
async def test_known_backchannels_suppressed_during_active_tts(phrase: str):
    """1. Known backchannels must be SUPPRESSED while active TTS is playing."""
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0  # Dead-zone passed

    # A. Classifier check
    classification = classify_turn(phrase, 1.00, is_tts_playing=True)
    assert classification == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL

    # B. Barge-in gate check
    assert handler._should_barge_in_on_stt(phrase, 1.00) is False

    # C. Interim STT arrival -> must NOT cancel audio, must NOT prefetch RAG
    await handler._maybe_process_interim(phrase, 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 0
    assert getattr(handler, "_rag_prefetch_task", None) is None
    assert getattr(handler, "_speculative_prefetch_task", None) is None
    assert handler._is_tts_playing is True  # Playback must continue undisturbed

    # D. Final STT arrival -> must NOT invoke LLM turn while TTS is playing
    await handler._process_transcript(phrase, 1.00)
    assert handler._complete_llm_turn_after_stt_final.call_count == 0
    assert handler._is_tts_playing is True


# ── Focus 2: Explicit Interruptions ──────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "Stop",
        "Stop please",
        "Hold on",
        "Wait",
        "Wait please",
        "Pause please",
        "Cancel that",
        "Be quiet",
    ],
)
async def test_explicit_interruptions_trigger_barge_in(phrase: str):
    """2. Explicit commands must trigger BARGE_IN and cancel active TTS immediately."""
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0  # Dead-zone passed

    # A. Classifier check
    classification = classify_turn(phrase, 0.95, is_tts_playing=True)
    assert classification == TurnClassification.BARGE_IN

    # B. Barge-in gate check
    assert handler._should_barge_in_on_stt(phrase, 0.95) is True

    # C. Interim STT arrival -> MUST cancel inflight response and reset _is_tts_playing
    await handler._maybe_process_interim(phrase, 0.95)
    assert handler._cancel_inflight_llm_response.call_count == 1
    assert handler._is_tts_playing is False


# ── Focus 3: Legitimate Short / Unknown Requests ──────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "Tell me",
        "Help me",
        "Call John",
        "Book appointment",
        "How much",
        "Blue widget",
        "Need help",
        "One more",
        "Try this",
    ],
)
async def test_legitimate_short_and_unknown_requests_never_silently_suppressed(
    phrase: str,
):
    """3. Legitimate short/unknown requests must NEVER be silently suppressed."""
    handler = _create_test_handler()

    # When spoken over active TTS: MUST barge-in and stop TTS to capture user request
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0

    classification = classify_turn(phrase, 0.90, is_tts_playing=True)
    assert classification == TurnClassification.BARGE_IN
    assert handler._should_barge_in_on_stt(phrase, 0.90) is True

    await handler._maybe_process_interim(phrase, 0.90)
    assert handler._cancel_inflight_llm_response.call_count == 1

    # When final STT arrives: MUST trigger LLM turn processing
    await handler._process_transcript(phrase, 0.90)
    assert handler._complete_llm_turn_after_stt_final.call_count == 1


# ── Focus 4: Interim -> Final STT Deduplication ──────────────────────────────
@pytest.mark.asyncio
async def test_interim_barge_in_then_final_stt_flow():
    """4. Verify interim barge-in cancels TTS, and final STT generates exactly one LLM turn."""
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0

    # 1. Interim "Hold on" arrives -> triggers barge in
    await handler._maybe_process_interim("Hold on", 0.95)
    assert handler._cancel_inflight_llm_response.call_count == 1
    assert handler._is_tts_playing is False

    # 2. Final "Hold on" arrives -> triggers exactly ONE LLM completion turn
    await handler._process_transcript("Hold on", 0.95)
    assert handler._complete_llm_turn_after_stt_final.call_count == 1


# ── Focus 5: Race Conditions ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_race_condition_tts_finishes_before_final_backchannel():
    """5. Race condition: Caller says 'Yeah yeah' while TTS plays; TTS ends naturally before final STT."""
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0

    # Interim arrives while TTS is playing -> suppressed
    await handler._maybe_process_interim("Yeah yeah", 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 0

    # TTS finishes playing naturally right before Deepgram endpoints
    handler._is_tts_playing = False

    # Final STT arrives when agent is silent -> processed normally without crashes or double-cancellations
    await handler._process_transcript("Yeah yeah", 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 0


@pytest.mark.asyncio
async def test_rapid_consecutive_interims():
    """5. Race condition: Rapid sequence of interims does not cause duplicate cancels or lock deadlocks."""
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0

    # Fire 5 rapid backchannel interims -> 0 cancels
    for _ in range(5):
        await handler._maybe_process_interim("Yeah yeah", 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 0

    # Fire 1 command interim -> cancels once
    await handler._maybe_process_interim("Stop please", 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 1


@pytest.mark.asyncio
async def test_twilio_and_livekit_handler_parity():
    """6. Ensure BidirectionalStreamHandler and LiveKitBrowserCallHandler share identical classification logic."""
    test_phrases = [
        ("Hey. Hi.", 1.00, False),
        ("Good morning", 1.00, False),
        ("Yeah yeah", 1.00, False),
        ("Stop please", 1.00, True),
        ("Hold on", 1.00, True),
        ("Cancel that", 1.00, True),
        ("Tell me", 0.90, True),
        ("Help me", 0.90, True),
        ("Book appointment", 0.90, True),
        ("Blue widget", 0.90, True),
    ]

    twilio_h = _create_test_handler()

    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_agent = MagicMock()
    mock_agent.id = uuid.uuid4()

    livekit_h = LiveKitBrowserCallHandler(
        db=None,
        call_session=mock_session,
        agent=mock_agent,
    )
    livekit_h._barge_in_min_words = 2
    livekit_h._barge_in_min_conf = 0.26
    livekit_h._barge_in_min_conf_1w = 0.36

    for phrase, conf, expected_barge in test_phrases:
        twilio_res = twilio_h._should_barge_in_on_stt(phrase, conf)
        livekit_res = livekit_h._should_barge_in_on_stt(phrase, conf)

        assert twilio_res == expected_barge, f"Twilio mismatch for '{phrase}'"
        assert livekit_res == expected_barge, f"LiveKit mismatch for '{phrase}'"
        assert twilio_res == livekit_res, f"Parity mismatch for '{phrase}'"
