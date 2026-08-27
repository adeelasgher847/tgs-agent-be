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

    # 1. Interim arrives ("I want to")
    await handler._maybe_process_interim("I want to", 0.90)
    assert handler._turn_response_started is True
    assert handler._llm_response_task is not None

    # Wait for interim task to finish producing valid response
    await handler._llm_response_task

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: LLM was called ONCE (by interim) and NOT duplicated by speech_final
    assert len(llm_generated_texts) == 1
    assert llm_generated_texts[0] == "I want to"

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
        if user_text == "I want to":
            # Interim produces empty/no response
            return ""
        return "Final response for interview scheduling."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to")
    await handler._maybe_process_interim("I want to", 0.90)
    await handler._llm_response_task

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: Final LLM was generated because interim produced empty response!
    assert len(llm_generated_texts) == 2
    assert llm_generated_texts[0] == "I want to"
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
        if user_text == "I want to":
            raise RuntimeError("Primary LLM network timeout")
        return "Final response for interview scheduling."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to")
    await handler._maybe_process_interim("I want to", 0.90)
    try:
        await handler._llm_response_task
    except Exception:
        pass

    # 2. Final arrives ("I want to schedule an interview.")
    await handler._complete_llm_turn_after_stt_final("I want to schedule an interview.", 0.95)

    # Verify: Final LLM was generated because interim failed!
    assert len(llm_generated_texts) == 2
    assert llm_generated_texts[0] == "I want to"
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
        if user_text == "I want to":
            await asyncio.sleep(1.0) # Long running task to be cancelled
            return "Should not reach here"
        return "Final response for interview scheduling."

    handler.generate_and_stream_response = mock_generate
    handler._add_to_transcript = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._should_accept_final_transcript = MagicMock(return_value=True)

    # 1. Interim arrives ("I want to")
    await handler._maybe_process_interim("I want to", 0.90)
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
    assert llm_generated_texts[0] == "I want to"
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




