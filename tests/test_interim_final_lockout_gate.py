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
