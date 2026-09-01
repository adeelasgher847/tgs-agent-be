"""
Unit & Regression Tests for Gate #7: Interim LLM Lockout & Final Transcript Recovery
Tests:
A. Interim starts -> valid response completes -> speech_final arrives -> no duplicate LLM response.
B. Interim starts -> empty response -> speech_final arrives -> final LLM response is generated.
C. Interim starts -> generation fails -> speech_final arrives -> final LLM response is generated.
D. Interim starts -> generation is cancelled/suppressed -> speech_final arrives -> final LLM response is generated.
"""
import asyncio
import time
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.routers.bidirectional_stream import BidirectionalStreamHandler

class DummyWebSocket:
    async def accept(self): pass
    async def send_text(self, data: str): pass
    async def send_json(self, data: dict): pass
    async def receive_text(self): return "{}"
    async def close(self): pass

@pytest.mark.asyncio
async def test_case_a_interim_valid_no_duplicate():
    """Case A: Interim starts -> valid response completes -> speech_final arrives -> no duplicate LLM response."""
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
    handler._enable_interim_llm = True

    llm_generated_texts = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        llm_generated_texts.append(user_text)
        await asyncio.sleep(0.01) # Simulate synthesis
        return "Here is your valid answer."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to schedule an", 5 words -- satisfies the
    # VOICE_MIN_INTERIM_WORDS=4 default gate).
    await handler._maybe_process_interim("I want to schedule an", 0.90)
    assert handler._turn_response_started is True
    assert handler._llm_response_task is not None

    # Wait for interim task to finish producing valid response
    await handler._llm_response_task

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: LLM was called ONCE (by interim) and NOT duplicated by speech_final
    assert len(llm_generated_texts) == 1
    assert llm_generated_texts[0] == "I want to schedule an"

@pytest.mark.asyncio
async def test_case_b_interim_empty_final_generated():
    """Case B: Interim starts -> empty response -> speech_final arrives -> final LLM response is generated."""
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
    handler._enable_interim_llm = True

    llm_generated_texts = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        llm_generated_texts.append(user_text)
        if user_text == "I want to schedule an":
            # Interim produces empty/no response
            return ""
        return "Final response for interview scheduling."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to schedule an", 5 words -- satisfies the
    # VOICE_MIN_INTERIM_WORDS=4 default gate).
    await handler._maybe_process_interim("I want to schedule an", 0.90)
    await handler._llm_response_task

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: Final LLM was generated because interim produced empty response!
    assert len(llm_generated_texts) == 2
    assert llm_generated_texts[0] == "I want to schedule an"
    assert llm_generated_texts[1] == "I want to schedule an interview."

@pytest.mark.asyncio
async def test_case_c_interim_failed_final_generated():
    """Case C: Interim starts -> generation fails -> speech_final arrives -> final LLM response is generated."""
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
    handler._enable_interim_llm = True

    llm_generated_texts = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        llm_generated_texts.append(user_text)
        if user_text == "I want to schedule an":
            raise RuntimeError("Primary LLM network timeout")
        return "Final response for interview scheduling."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to schedule an", 5 words -- satisfies the
    # VOICE_MIN_INTERIM_WORDS=4 default gate).
    await handler._maybe_process_interim("I want to schedule an", 0.90)
    try:
        await handler._llm_response_task
    except Exception:
        pass

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: Final LLM was generated because interim failed!
    assert len(llm_generated_texts) == 2
    assert llm_generated_texts[0] == "I want to schedule an"
    assert llm_generated_texts[1] == "I want to schedule an interview."

@pytest.mark.asyncio
async def test_case_d_interim_cancelled_final_generated():
    """Case D: Interim starts -> generation is cancelled/suppressed -> speech_final arrives -> final LLM response is generated."""
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
    handler._enable_interim_llm = True

    llm_generated_texts = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        llm_generated_texts.append(user_text)
        if user_text == "I want to schedule an":
            await asyncio.sleep(1.0) # Long running task to be cancelled
            return "Should not reach here"
        return "Final response for interview scheduling."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to schedule an", 5 words -- satisfies the
    # VOICE_MIN_INTERIM_WORDS=4 default gate).
    await handler._maybe_process_interim("I want to schedule an", 0.90)
    await asyncio.sleep(0.01)  # Yield to let interim task start

    # Cancel the interim task
    handler._llm_response_task.cancel()
    try:
        await handler._llm_response_task
    except asyncio.CancelledError:
        pass

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: Final LLM was generated because interim was cancelled!
    assert "I want to schedule an interview." in llm_generated_texts
    assert len(llm_generated_texts) == 2
    assert llm_generated_texts[0] == "I want to schedule an"
    assert llm_generated_texts[1] == "I want to schedule an interview."

@pytest.mark.asyncio
async def test_case_e_concurrent_overlapping_turns_isolation():
    """Case E: Turn A interim starts (in-flight) -> Turn A final arrives -> Turn B starts concurrently -> Turn A cannot be polluted by Turn B."""
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
    handler._enable_interim_llm = True
    handler._min_interim_words = 2
    handler._min_interim_confidence = 0.14
    handler._min_interim_interval_sec = 0.0
    handler._is_booking_context_active = MagicMock(return_value=False)

    llm_generated_texts = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        llm_generated_texts.append(user_text)
        if user_text == "Turn A interim":
            await asyncio.sleep(0.05)  # In-flight task
            # Turn A interim fails / produces empty
            return ""
        elif user_text == "Turn B interim":
            await asyncio.sleep(0.01)
            # Turn B interim succeeds
            return "Turn B success"
        elif user_text == "Turn A final full text":
            return "Turn A full answer"
        return "Generic response"

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Turn A interim starts (in flight)
    await handler._maybe_process_interim("Turn A interim", 0.90)
    assert handler._turn_response_started is True
    await asyncio.sleep(0.005)  # Yield to let Turn A task start running

    # 2. Turn A final arrives while Task A is still in-flight
    # In background, simulate Turn B interim starting while Turn A final is awaiting Task A
    async def simulate_turn_b():
        await asyncio.sleep(0.01)
        # Turn B starts an interim task
        handler._turn_response_started = False
        await handler._maybe_process_interim("Turn B interim", 0.90)

    b_task = asyncio.create_task(simulate_turn_b())

    # Turn A final processes
    await handler._complete_llm_turn_after_stt_final("Turn A final full text", 0.95)
    await b_task
    if handler._llm_response_task:
        await handler._llm_response_task

    # Verify:
    # 1. Turn A final MUST be generated because Turn A interim produced empty output.
    # 2. Turn A did NOT observe Turn B's success.
    assert "Turn A interim" in llm_generated_texts
    assert "Turn A final full text" in llm_generated_texts

@pytest.mark.asyncio
async def test_case_f_interim_disabled_waits_for_speech_final():
    """Case F: Interim LLM disabled (default/demo mode) -> partials do NOT generate LLM -> speech_final triggers authoritative response."""
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
    handler._enable_interim_llm = False

    llm_generated_texts = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        llm_generated_texts.append(user_text)
        await asyncio.sleep(0.01)
        return "Hawk Auto Care provides automotive repair and maintenance services."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Incomplete partial arrives ("Can you please")
    await handler._maybe_process_interim("Can you please", 0.90)

    # Verify: No LLM task was started
    assert handler._turn_response_started is False
    assert handler._llm_response_task is None
    assert len(llm_generated_texts) == 0

    # 2. Complete speech_final arrives ("Can you please tell me about your business?")
    await handler._complete_llm_turn_after_stt_final("Can you please tell me about your business?", 0.95)

    # Verify: Exactly one authoritative LLM response was generated
    assert len(llm_generated_texts) == 1
    assert llm_generated_texts[0] == "Can you please tell me about your business?"

@pytest.mark.asyncio
async def test_case_g_barge_in_thresholds_with_interim_disabled():
    """Case G: Verify barge-in behavior with interim disabled:
    - 1-word input does not barge in (min_words=2)
    - 2+ word input with confidence >= 0.26 triggers barge-in
    """
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
    handler._enable_interim_llm = False
    handler._barge_in_min_words = 2
    handler._barge_in_min_conf = 0.26
    handler._barge_in_min_conf_1w = 0.36
    handler._barge_in_dead_zone_ms = 600

    handler._cancel_inflight_llm_response = AsyncMock()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0  # Dead-zone passed

    # 1. 1-word utterance ("Hey" or "Stop") -> should NOT barge in when min_words=2
    await handler._maybe_process_interim("Hey", 0.95)
    assert handler._cancel_inflight_llm_response.call_count == 0

    # 2. 2-word low-confidence command ("Stop please", conf=0.20 < 0.26) -> should NOT barge in
    await handler._maybe_process_interim("Stop please", 0.20)
    assert handler._cancel_inflight_llm_response.call_count == 0

    # 3. 2-word confident command ("Stop please", conf=0.85 >= 0.26) -> SHOULD barge in
    await handler._maybe_process_interim("Stop please", 0.85)
    assert handler._cancel_inflight_llm_response.call_count == 1

@pytest.mark.asyncio
async def test_case_h_no_google_tts_precaching_on_call_startup():
    """Case H: Verify that initializing BidirectionalStreamHandler does NOT trigger bulk Google TTS precaching."""
    with patch("app.services.google_tts_service.google_tts_service.text_to_speech") as mock_google_tts:
        handler = BidirectionalStreamHandler(
            websocket=DummyWebSocket(),
            call_session_id=str(uuid.uuid4()),
            agent_id=str(uuid.uuid4()),
            db=None,
        )
        
        # Let event loop spin and sleep to allow any pending background tasks to execute
        await asyncio.sleep(0.1)
        
        # Verify: _precache_common_phrases attribute does NOT exist on handler
        assert not hasattr(handler, "_precache_common_phrases")
        
        # Verify: Google TTS was NOT called for batch precaching
        assert mock_google_tts.call_count == 0

@pytest.mark.asyncio
async def test_case_i_conversational_backchannel_vs_explicit_command_barge_in():
    """Case I: Verify that conversational 2-word backchannels/greetings do NOT cut TTS,
    while 2-word explicit commands and 3+ word utterances DO cut TTS."""
    handler = BidirectionalStreamHandler(
        websocket=DummyWebSocket(),
        call_session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        db=None,
    )
    handler._enable_interim_llm = False
    handler._barge_in_min_words = 2
    handler._barge_in_min_conf = 0.26
    handler._barge_in_min_conf_1w = 0.36
    handler._barge_in_dead_zone_ms = 600

    handler._cancel_inflight_llm_response = AsyncMock()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0  # Dead-zone passed

    # A. Conversational / backchannel phrases (MUST NOT cut TTS)
    assert handler._should_barge_in_on_stt("Hey", 1.00) is False
    assert handler._should_barge_in_on_stt("Hi", 1.00) is False
    assert handler._should_barge_in_on_stt("Hey. Hi.", 1.00) is False
    assert handler._should_barge_in_on_stt("Hi there", 1.00) is False
    assert handler._should_barge_in_on_stt("Good morning", 1.00) is False
    assert handler._should_barge_in_on_stt("Yeah yeah", 1.00) is False
    assert handler._should_barge_in_on_stt("Okay okay", 1.00) is False
    assert handler._should_barge_in_on_stt("Thank you", 1.00) is False
    assert handler._should_barge_in_on_stt("Got it", 1.00) is False
    assert handler._should_barge_in_on_stt("I see", 1.00) is False

    # Simulate interim "Hey. Hi." arriving while TTS is playing -> verify no cancellation
    await handler._maybe_process_interim("Hey. Hi.", 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 0

    # B. Explicit 2-word interruption commands (MUST cut TTS)
    assert handler._should_barge_in_on_stt("Stop please", 1.00) is True
    assert handler._should_barge_in_on_stt("Wait please", 1.00) is True
    assert handler._should_barge_in_on_stt("Hold on", 1.00) is True
    assert handler._should_barge_in_on_stt("Pause please", 1.00) is True
    assert handler._should_barge_in_on_stt("Cancel that", 1.00) is True
    assert handler._should_barge_in_on_stt("Stop talking", 1.00) is True

    # Simulate interim "Stop please" arriving -> verify cancellation is called
    await handler._maybe_process_interim("Stop please", 1.00)
    assert handler._cancel_inflight_llm_response.call_count == 1

    # C. 3+ word utterances (MUST cut TTS)
    assert handler._should_barge_in_on_stt("Wait a second", 1.00) is True
    assert handler._should_barge_in_on_stt("I have a question", 1.00) is True
    assert handler._should_barge_in_on_stt("Hey. Hi. Stop please", 1.00) is True

@pytest.mark.asyncio
async def test_case_j_legitimate_short_and_unknown_requests_not_dropped():
    """Case J: Verify that legitimate short 2-word requests and unknown phrases are NOT dropped
    when spoken over active TTS (they barge in) and are processed normally when silent."""
    handler = BidirectionalStreamHandler(
        websocket=DummyWebSocket(),
        call_session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        db=None,
    )
    handler._enable_interim_llm = False
    handler._barge_in_min_words = 2
    handler._barge_in_min_conf = 0.26
    handler._barge_in_min_conf_1w = 0.36
    handler._barge_in_dead_zone_ms = 600

    handler._cancel_inflight_llm_response = AsyncMock()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0

    # Legitimate 2-word user requests MUST barge in when spoken over active TTS
    legitimate_short_requests = [
        "Tell me",
        "Help me",
        "Call John",
        "Book appointment",
        "Transfer me",
        "What now",
        "What's that",
        "Who is that",
        "Where exactly",
        "How much",
        "Tell me more",
        "Stop billing",
        "Cancel order",
        "Blue widget",  # Unknown / unseen 2-word phrase
    ]

    for req in legitimate_short_requests:
        assert handler._should_barge_in_on_stt(req, 0.90) is True, f"Failed to barge in for legitimate request: {req}"

    # Verify that sending a legitimate request over active TTS triggers cancellation
    await handler._maybe_process_interim("Tell me", 0.90)
    assert handler._cancel_inflight_llm_response.call_count == 1


@pytest.mark.asyncio
async def test_case_f_slow_but_valid_interim_not_duplicated_under_latency():
    """
    Case F (duplicate-audio-under-latency regression):

    Interim seed text == final transcript (no regeneration needed per
    _should_regenerate_on_final) -> handler waits on the in-flight interim
    task via asyncio.wait_for(asyncio.shield(...), timeout=2.5). If the
    interim is simply SLOW (e.g. under latency) but not actually stuck --
    it has already queued/committed its TTS response for this turn -- the
    2.5s wait timing out must NOT cause a duplicate cancel+regenerate+re-
    speak of the same content. asyncio.shield() means the interim keeps
    running (and may already be mid-playback) even after the wait_for
    gives up, so falling through to a fresh generation on timeout alone
    re-synthesizes and re-plays the same answer -- the "same audio plays
    again, slightly pitch-shifted" symptom reported from live calls.
    """
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
    handler._enable_interim_llm = True

    calls = []

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        calls.append(user_text)
        # Simulate slow-but-successful LLM+TTS under latency: the real
        # generate_and_stream_response queues TTS + commits the transcript
        # (setting _interim_task_response_produced=True) well before it
        # actually *returns* -- reproduce that ordering here, with the
        # overall generation taking longer than the 2.5s fallback timeout.
        await asyncio.sleep(1.0)
        handler._interim_task_response_produced = True
        await asyncio.sleep(2.0)
        return "Sure, I can help with that."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    utterance = "I want to schedule an interview"

    # Interim arrives with the FULL utterance already (common for short
    # utterances that finish right as Deepgram emits its first interim).
    await handler._maybe_process_interim(utterance, 0.90)
    assert handler._turn_response_started is True

    # Final arrives with the SAME transcript almost immediately --
    # _should_regenerate_on_final sees final_norm == seed_norm (not a
    # booking context) -> returns False -> handler must just wait for/trust
    # the in-flight interim, not start a second generation.
    await handler._complete_llm_turn_after_stt_final(utterance, 0.95)

    assert len(calls) == 1, (
        f"expected exactly ONE generation for identical interim/final text "
        f"even when the interim is slower than the 2.5s fallback timeout, "
        f"got {len(calls)} calls: {calls} (duplicate-audio-under-latency regression)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configurable turn timeout (VOICE_TURN_TIMEOUT_SEC)
#
# `_complete_llm_turn_after_stt_final` used to hardcode
# `asyncio.wait_for(self.generate_and_stream_response(...), timeout=12.0)`.
# It now resolves the timeout from `settings.VOICE_TURN_TIMEOUT_SEC` (falling
# back to 20.0, matching both the Settings field's own default in
# app/core/config.py and the browser handler's fallback, so the two
# transports share one real-world default). See the equivalent `_timeout`
# resolution in
# app/voice/livekit_browser_call_handler.py::_complete_llm_turn_after_stt_final.
# ─────────────────────────────────────────────────────────────────────────────

def _make_plain_final_handler() -> BidirectionalStreamHandler:
    """A handler with no interim in-flight, ready to go straight through
    `_complete_llm_turn_after_stt_final`'s Phase-2 generate-and-wait path."""
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
    handler._enable_interim_llm = False
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)
    return handler


@pytest.mark.asyncio
async def test_turn_timeout_uses_configured_setting_value(monkeypatch, caplog):
    """When VOICE_TURN_TIMEOUT_SEC is overridden (e.g. to 0.05s), that value
    -- not the old hardcoded 12.0 -- gates asyncio.wait_for."""
    monkeypatch.setattr(settings, "VOICE_TURN_TIMEOUT_SEC", 0.05, raising=False)

    handler = _make_plain_final_handler()

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        # Longer than the configured 0.05s timeout, but far shorter than the
        # old hardcoded 12.0s -- only fails/times out if the new setting is
        # actually being honored.
        await asyncio.sleep(0.3)
        return "late answer"

    handler.generate_and_stream_response = mock_generate

    start = time.monotonic()
    with caplog.at_level("ERROR"):
        await handler._complete_llm_turn_after_stt_final("hello there", 0.95)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"expected wait_for to bail around 0.05s, took {elapsed:.2f}s"
    assert any(
        "generate_and_stream_response timed out" in r.message and "0.1s" in r.message
        for r in caplog.records
    ), f"expected timeout log reflecting the configured 0.05s value, got: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_turn_timeout_falls_back_to_twenty_seconds_when_falsy(monkeypatch):
    """When VOICE_TURN_TIMEOUT_SEC is explicitly falsy (e.g. 0.0), the `or`
    fallback kicks in and resolves to 20.0 -- verified here by patching
    asyncio.wait_for directly (a real multi-second sleep is impractical in a
    unit test) and asserting the timeout kwarg it was called with."""
    monkeypatch.setattr(settings, "VOICE_TURN_TIMEOUT_SEC", 0.0, raising=False)

    handler = _make_plain_final_handler()

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        return "quick answer"

    handler.generate_and_stream_response = mock_generate

    captured_timeout = {}
    real_wait_for = asyncio.wait_for

    async def spy_wait_for(aw, timeout=None):
        captured_timeout["value"] = timeout
        return await real_wait_for(aw, timeout=timeout)

    with patch("app.routers.bidirectional_stream.asyncio.wait_for", side_effect=spy_wait_for):
        await handler._complete_llm_turn_after_stt_final("hello there", 0.95)

    assert captured_timeout.get("value") == 20.0


@pytest.mark.asyncio
async def test_turn_timeout_uses_real_default_setting_when_not_overridden():
    """Regression guard: with settings.VOICE_TURN_TIMEOUT_SEC left completely
    unpatched (its real Settings-field default), the resolved timeout must
    equal that real default -- not a stale hardcoded literal that would
    silently diverge from app/core/config.py's actual default over time."""
    handler = _make_plain_final_handler()

    async def mock_generate(user_text: str, confidence: float = 1.0, is_greeting: bool = False):
        return "quick answer"

    handler.generate_and_stream_response = mock_generate

    captured_timeout = {}
    real_wait_for = asyncio.wait_for

    async def spy_wait_for(aw, timeout=None):
        captured_timeout["value"] = timeout
        return await real_wait_for(aw, timeout=timeout)

    with patch("app.routers.bidirectional_stream.asyncio.wait_for", side_effect=spy_wait_for):
        await handler._complete_llm_turn_after_stt_final("hello there", 0.95)

    assert captured_timeout.get("value") == settings.VOICE_TURN_TIMEOUT_SEC





