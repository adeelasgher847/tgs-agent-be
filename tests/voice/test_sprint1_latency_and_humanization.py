import asyncio
import time
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services.deepgram_stt_service import DeepgramSTTService
from app.services.google_stt_service import GoogleSttService
from app.voice.humanization_engine import analyze_response
from app.voice.metrics import VoiceTurnMetrics


def test_quick_ack_disabled_by_default():
    """Verify quick acknowledgements are disabled by default in settings."""
    assert getattr(settings, "VOICE_QUICK_ACK_PROBABILITY", 0.0) == 0.0


def test_metrics_latency_calculations():
    """Verify VoiceTurnMetrics accurately computes latency deltas from monotonic timestamps."""
    metrics = VoiceTurnMetrics(
        call_sid="CA12345",
        turn_id="turn-1",
        transport="telephony",
        agent_id="agent-abc",
        provider="anthropic",
    )
    t0 = time.perf_counter()
    metrics.turn_stt_final_mono = t0
    metrics.generation_anchor_mono = t0 + 0.05
    metrics.prompt_start_mono = t0 + 0.05
    metrics.rag_start_mono = t0 + 0.05
    metrics.rag_end_mono = t0 + 0.15
    metrics.prompt_ready_mono = t0 + 0.16
    metrics.llm_request_mono = t0 + 0.17
    metrics.turn_llm_first_token_mono = t0 + 0.45
    metrics.turn_first_tts_queued_mono = t0 + 0.46
    metrics.tts_first_audio_mono = t0 + 0.60
    metrics.first_playback_mono = t0 + 0.62
    metrics.turn_complete_mono = t0 + 1.20

    latencies = metrics.calculate_latencies()
    assert latencies["stt_endpoint_latency_ms"] == pytest.approx(50.0, abs=5.0)
    assert latencies["rag_latency_ms"] == pytest.approx(100.0, abs=5.0)
    assert latencies["prompt_assembly_latency_ms"] == pytest.approx(110.0, abs=5.0)
    assert latencies["llm_ttft_ms"] == pytest.approx(280.0, abs=5.0)
    assert latencies["tts_ttfa_ms"] == pytest.approx(140.0, abs=5.0)
    assert latencies["stt_final_to_first_audio_ms"] == pytest.approx(600.0, abs=5.0)
    assert latencies["total_turn_latency_ms"] == pytest.approx(620.0, abs=5.0)

    payload = metrics.build_telemetry_payload(user_preview="Hello there")
    assert payload["call_sid"] == "CA12345"
    assert payload["turn_id"] == "turn-1"
    assert payload["transport"] == "telephony"
    assert payload["agent_id"] == "agent-abc"
    assert payload["provider"] == "anthropic"
    assert payload["user_preview"] == "Hello there"
    assert payload["latencies"]["llm_ttft_ms"] == pytest.approx(280.0, abs=5.0)


@pytest.mark.asyncio
async def test_deepgram_stt_async_queue():
    """Verify Deepgram StreamingSTTSession delivers items asynchronously without blocking executor."""
    service = DeepgramSTTService()
    mock_client = MagicMock()
    session = service.StreamingSTTSession(
        client=mock_client,
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        interim_results=True,
        single_utterance=False,
    )
    session._loop = asyncio.get_running_loop()

    # Simulate message arriving from Deepgram SDK thread
    test_item = {"transcript": "hello world", "confidence": 0.95, "is_final": True}
    session._push_result(test_item)

    # Read from async get_result()
    result = await asyncio.wait_for(session.get_result(), timeout=1.0)
    assert result["transcript"] == "hello world"
    assert result["confidence"] == 0.95
    assert result["is_final"] is True


@pytest.mark.asyncio
async def test_google_stt_async_queue():
    """Verify Google STT StreamingSTTSession delivers items asynchronously without blocking executor."""
    service = GoogleSttService()
    session = service.StreamingSTTSession(
        language_code="en-US",
        sample_rate_hz=8000,
        encoding="MULAW",
        interim_results=True,
    )
    session._loop = asyncio.get_running_loop()

    # Simulate message arriving from Google SDK thread
    test_item = {"transcript": "booking confirmation", "confidence": 0.98, "is_final": True}
    session._push_result(test_item)

    # Read from async get_result()
    result = await asyncio.wait_for(session.get_result(), timeout=1.0)
    assert result["transcript"] == "booking confirmation"
    assert result["confidence"] == 0.98
    assert result["is_final"] is True


@pytest.mark.asyncio
async def test_hubspot_crm_caching_in_memory_without_db_flush():
    """Verify HubSpot CRM context block is cached in memory on call_metadata without db.flush()."""
    from app.services.hubspot_service import get_crm_context_block_for_call

    mock_db = MagicMock()
    mock_call_session = MagicMock()
    mock_call_session.call_metadata = {}
    mock_call_session.customer_phone_number = None

    # First call — caches empty context without db.flush
    result1 = await get_crm_context_block_for_call(mock_db, mock_call_session)
    assert result1 == ""
    assert "hubspot_crm_context" in mock_call_session.call_metadata
    assert mock_db.flush.call_count == 0
    assert mock_db.begin_nested.call_count == 0

    # Second call — immediate 0ms cache return
    result2 = await get_crm_context_block_for_call(mock_db, mock_call_session)
    assert result2 == ""
    assert mock_db.flush.call_count == 0


@pytest.mark.asyncio
async def test_salesforce_crm_caching_in_memory_without_db_flush():
    """Verify Salesforce CRM context block is cached in memory on call_metadata without db.flush()."""
    from app.services.salesforce_service import get_crm_context_block_for_call

    mock_db = MagicMock()
    mock_call_session = MagicMock()
    mock_call_session.call_metadata = {}
    mock_call_session.customer_phone_number = None

    result = await get_crm_context_block_for_call(mock_db, mock_call_session)
    assert result == ""
    assert "salesforce_crm_context" in mock_call_session.call_metadata
    assert mock_db.flush.call_count == 0
    assert mock_db.begin_nested.call_count == 0


@pytest.mark.asyncio
async def test_ghl_crm_caching_in_memory_without_db_flush():
    """Verify GHL CRM context block is cached in memory on call_metadata without db.flush()."""
    from app.services.ghl_service import get_crm_context_block_for_call

    mock_db = MagicMock()
    mock_call_session = MagicMock()
    mock_call_session.call_metadata = {}
    mock_call_session.customer_phone_number = None

    result = await get_crm_context_block_for_call(mock_db, mock_call_session)
    assert result == ""
    assert "ghl_crm_context" in mock_call_session.call_metadata
    assert mock_db.flush.call_count == 0
    assert mock_db.begin_nested.call_count == 0


def test_humanization_engine_tone_adapter_clean_preservation():
    """Verify humanization engine preserves clean responses without injecting forced artificial fillers."""
    response = analyze_response(
        text="I can certainly help you book an appointment for tomorrow at 2 PM.",
        user_text="Can I book a consultation tomorrow?",
        stt_confidence=0.95,
        use_ssml=False,
        is_final=True,
    )
    assert response is not None
    assert "umm" not in response.text.lower()
    assert "hmm" not in response.text.lower()
    assert "uhm" not in response.text.lower()
    assert len(response.text) > 0
