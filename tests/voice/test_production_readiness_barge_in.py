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

from app.core.config import settings
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


# ── Focus 7: Twilio-specific barge-in confidence config resolution ───────────
# Regression coverage for the bug where BidirectionalStreamHandler.__init__
# read the *shared* VOICE_BARGE_IN_MIN_CONFIDENCE / _1W settings (default
# 0.26 / 0.36) via getattr(), which always resolved because Settings already
# declared those fields -- silently discarding the intended 0.70 / 0.75
# literal fallback. The fix introduces Twilio-specific settings keys
# (VOICE_BARGE_IN_MIN_CONFIDENCE_TWILIO / _1W_TWILIO) that the Twilio handler
# now reads instead, while the browser/LiveKit handler keeps reading the
# original shared keys unchanged.


def _raw_twilio_handler() -> BidirectionalStreamHandler:
    """Construct a BidirectionalStreamHandler without post-init attribute
    overrides, so __init__'s settings resolution can be observed directly."""
    return BidirectionalStreamHandler(
        websocket=DummyWebSocket(),
        call_session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        db=None,
    )


def _raw_livekit_handler() -> LiveKitBrowserCallHandler:
    """Construct a LiveKitBrowserCallHandler without post-init attribute
    overrides, so __init__'s settings resolution can be observed directly."""
    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_agent = MagicMock()
    mock_agent.id = uuid.uuid4()
    return LiveKitBrowserCallHandler(
        db=None,
        call_session=mock_session,
        agent=mock_agent,
    )


@pytest.mark.asyncio
async def test_twilio_handler_barge_in_confidence_defaults_to_070_and_075():
    """Twilio handler must resolve barge-in confidence thresholds to the
    intended 0.70 / 0.75 defaults, not the shared browser-path 0.26 / 0.36
    defaults that previously shadowed them via getattr()."""
    handler = _raw_twilio_handler()

    assert handler._barge_in_min_conf == pytest.approx(0.70)
    assert handler._barge_in_min_conf_1w == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_twilio_handler_barge_in_confidence_reads_dedicated_settings_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overriding VOICE_BARGE_IN_MIN_CONFIDENCE_TWILIO / _1W_TWILIO must
    actually change the resolved handler thresholds -- proves the settings
    knob is live, not dead code."""
    monkeypatch.setattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE_TWILIO", 0.55)
    monkeypatch.setattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE_1W_TWILIO", 0.60)

    handler = _raw_twilio_handler()

    assert handler._barge_in_min_conf == pytest.approx(0.55)
    assert handler._barge_in_min_conf_1w == pytest.approx(0.60)

    # Overriding the Twilio-specific keys must NOT perturb the shared keys
    # that the browser/LiveKit handler reads.
    assert settings.VOICE_BARGE_IN_MIN_CONFIDENCE == pytest.approx(0.26)
    assert settings.VOICE_BARGE_IN_MIN_CONFIDENCE_1W == pytest.approx(0.36)


def test_livekit_browser_handler_barge_in_confidence_unchanged():
    """The browser/LiveKit path must still resolve its barge-in confidence
    thresholds from the original SHARED settings keys (default 0.26 / 0.36)
    -- confirms the Twilio-specific fix did not accidentally change
    browser-path behavior, which intentionally stays more sensitive since
    WebRTC has native echo cancellation."""
    handler = _raw_livekit_handler()

    assert handler._barge_in_min_conf == pytest.approx(0.26)
    assert handler._barge_in_min_conf_1w == pytest.approx(0.36)


# ── Focus 8: STT final-confidence config resolution ───────────────────────────
# Regression coverage for a sibling dead-code bug: BidirectionalStreamHandler
# .__init__ read VOICE_STT_MIN_FINAL_CONFIDENCE / VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE
# via getattr(settings, key, <literal default>) or <literal default>. Because
# Settings previously declared these fields with different literal defaults
# (0.15 / 0.12), getattr() always found the declared field and the intended
# 0.26 / 0.16 fallback literals baked into the getattr(...) calls were dead
# code -- silently overridden. The fix aligns Settings' declared defaults
# (0.26 / 0.16) with the code's intended literals. Unlike the barge-in
# confidence bug, this doesn't touch LiveKitBrowserCallHandler at all --
# grep confirms the browser handler never references either settings key.


@pytest.mark.asyncio
async def test_twilio_handler_stt_final_confidence_defaults_to_026_and_016():
    """BidirectionalStreamHandler must resolve STT final-confidence
    thresholds to the intended 0.26 / 0.16 defaults, not the previously
    mis-declared Settings defaults (0.15 / 0.12) that shadowed them via
    getattr()."""
    handler = _raw_twilio_handler()

    assert handler._stt_min_final_confidence == pytest.approx(0.26)
    assert handler._stt_soft_min_final_confidence == pytest.approx(0.16)


@pytest.mark.asyncio
async def test_twilio_handler_stt_final_confidence_reads_live_settings_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overriding VOICE_STT_MIN_FINAL_CONFIDENCE / VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE
    must actually change the resolved handler thresholds -- proves the
    settings knob is live, not dead code. Values are chosen within the
    clamp ranges enforced by __init__ (0.15-0.45 / 0.10-0.35)."""
    monkeypatch.setattr(settings, "VOICE_STT_MIN_FINAL_CONFIDENCE", 0.30)
    monkeypatch.setattr(settings, "VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE", 0.20)

    handler = _raw_twilio_handler()

    assert handler._stt_min_final_confidence == pytest.approx(0.30)
    assert handler._stt_soft_min_final_confidence == pytest.approx(0.20)


# ── Focus 9: Pickup-RMS / interim-gate config resolution ──────────────────────
# Regression coverage for a sibling dead-code bug: Settings previously declared
# VOICE_MIN_AUDIO_RMS_FOR_PICKUP=20, VOICE_MIN_INTERIM_WORDS=2, and
# VOICE_MIN_INTERIM_CONFIDENCE=0.14, which shadowed the intended 70 / 4 / 0.52
# literal fallbacks baked into both handlers' getattr(settings, key, <literal>)
# calls via getattr() always finding the declared field. With the old default
# of 20, VOICE_MIN_AUDIO_RMS_FOR_PICKUP was silently clamped to the handler's
# own floor of 20 -- risking false "user picked up" triggers on line noise for
# every live call on both transports. The interim-gate defaults (2 / 0.14) only
# mattered once VOICE_ENABLE_INTERIM_LLM is turned on (default off), so that
# part of the bug was dormant but still a landmine. The fix aligns Settings'
# declared defaults (70 / 4 / 0.52) with the code's intended literals.


@pytest.mark.asyncio
async def test_twilio_handler_pickup_rms_threshold_defaults_to_70():
    """BidirectionalStreamHandler must resolve the pickup-detection RMS
    threshold to the intended 70 default, not the previously mis-declared
    Settings default (20) that silently clamped every call to the handler's
    own floor via getattr()."""
    handler = _raw_twilio_handler()

    assert handler._min_audio_level_threshold == 70


@pytest.mark.asyncio
async def test_twilio_handler_pickup_rms_threshold_reads_live_settings_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overriding VOICE_MIN_AUDIO_RMS_FOR_PICKUP must actually change the
    resolved handler threshold -- proves the settings knob is live, not dead
    code. Value chosen within the clamp range enforced by __init__ (20-250)."""
    monkeypatch.setattr(settings, "VOICE_MIN_AUDIO_RMS_FOR_PICKUP", 100)

    handler = _raw_twilio_handler()

    assert handler._min_audio_level_threshold == 100


def test_livekit_browser_handler_pickup_rms_threshold_also_defaults_to_70():
    """The browser/LiveKit path must resolve the SAME pickup-detection RMS
    threshold default (70) as the Twilio path -- unlike the barge-in
    confidence keys, this particular knob is intentionally NOT decoupled
    between transports; both want parity here."""
    handler = _raw_livekit_handler()

    assert handler._min_audio_level_threshold == 70


@pytest.mark.asyncio
async def test_twilio_handler_interim_gate_defaults_to_4_words_052_confidence():
    """BidirectionalStreamHandler must resolve the interim-LLM gate
    (VOICE_MIN_INTERIM_WORDS / VOICE_MIN_INTERIM_CONFIDENCE) to the intended
    4 / 0.52 defaults, not the previously mis-declared Settings defaults
    (2 / 0.14) that shadowed them via getattr(). These gates only activate
    when VOICE_ENABLE_INTERIM_LLM=True (default off), so the bug was dormant
    but still a landmine for whenever that feature is enabled."""
    handler = _raw_twilio_handler()

    assert handler._min_interim_words == 4
    assert handler._min_interim_confidence == pytest.approx(0.52)


@pytest.mark.asyncio
async def test_twilio_handler_interim_gate_reads_live_settings_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overriding VOICE_MIN_INTERIM_WORDS / VOICE_MIN_INTERIM_CONFIDENCE must
    actually change the resolved handler gate -- proves the settings knobs
    are live, not dead code."""
    monkeypatch.setattr(settings, "VOICE_MIN_INTERIM_WORDS", 3)
    monkeypatch.setattr(settings, "VOICE_MIN_INTERIM_CONFIDENCE", 0.35)

    handler = _raw_twilio_handler()

    assert handler._min_interim_words == 3
    assert handler._min_interim_confidence == pytest.approx(0.35)
