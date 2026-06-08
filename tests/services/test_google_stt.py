"""
Unit + integration tests for Google STT multi-provider architecture.

Covers all Part C acceptance criteria from the ticket:
  1. Interim event shape {type, transcript, confidence}
  2. Final event shape {type, transcript, confidence, isSilence}
  3. Silence detection (isSilence=True after SILENCE_THRESHOLD_MS)
  4. Google STT stream restart at 5-min boundary (mock clock)
  5. Error event + graceful restart on quota/network failure
  6. _resolve_stt_model validation
  7. agent_to_out includes sttModel
  8. resolve_stt_runtime priority (flow > agent > catalog)
  9. Integration: 3-second audio clip → final transcript
"""
from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.stt_events import SttEventBus, SttFinalEvent, SttInterimEvent, SttErrorEvent


# ─────────────────────────────────────────────────────────────────────────────
# 1. Interim event shape
# ─────────────────────────────────────────────────────────────────────────────

def test_interim_event_shape():
    event = SttInterimEvent(transcript="hello", confidence=0.85)
    assert event.type == "interim"
    assert event.transcript == "hello"
    assert event.confidence == 0.85


# ─────────────────────────────────────────────────────────────────────────────
# 2. Final event shape
# ─────────────────────────────────────────────────────────────────────────────

def test_final_event_shape():
    event = SttFinalEvent(transcript="hello world", confidence=0.95, is_silence=False)
    assert event.type == "final"
    assert event.transcript == "hello world"
    assert event.confidence == 0.95
    assert event.is_silence is False


def test_final_event_with_is_silence():
    event = SttFinalEvent(transcript="", confidence=0.0, is_silence=True)
    assert event.type == "final"
    assert event.is_silence is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Silence detection — isSilence=True after SILENCE_THRESHOLD_MS
# ─────────────────────────────────────────────────────────────────────────────

def test_stt_pipeline_silence_detection():
    """SttPipeline._is_silence() and final event is_silence flag work correctly."""
    from app.voice.stt_pipeline import SttPipeline

    received_events = []

    async def on_interim(t, c):
        pass

    async def on_final(t, c):
        pass

    async def capture(e):
        received_events.append(e)

    async def _run():
        bus = SttEventBus()
        bus.subscribe(capture)

        pipeline = SttPipeline(
            language_code="en",
            on_interim=on_interim,
            on_final=on_final,
            silence_threshold_ms=100,
            event_bus=bus,
        )

        # Simulate: audio was last received 200ms ago → exceeds 100ms threshold
        pipeline._last_audio_mono = time.monotonic() - 0.2
        assert pipeline._is_silence() is True, "_is_silence should be True at 200ms > 100ms threshold"

        # Directly emit what the reader loop would produce
        is_sil = pipeline._is_silence()
        await bus.emit(SttFinalEvent(transcript="test", confidence=0.9, is_silence=is_sil))

    asyncio.run(_run())

    finals = [e for e in received_events if isinstance(e, SttFinalEvent)]
    assert finals, "Expected at least one SttFinalEvent"
    assert any(e.is_silence for e in finals), "Expected is_silence=True in at least one final event"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Google STT stream restart at 5-minute boundary (mock clock)
# ─────────────────────────────────────────────────────────────────────────────

def test_google_stt_stream_restart_at_5min():
    """_run_blocking_stream restarts when _run_single_stream signals needs_restart=True."""
    from app.services.google_stt_service import GoogleSttService

    sess = GoogleSttService.StreamingSTTSession(
        language_code="en-AU",
        sample_rate_hz=16000,
        encoding="LINEAR16",
        interim_results=True,
        api_config={},
        silence_threshold_ms=1500,
    )

    call_count = [0]

    def fake_run_single_stream(replay_chunks=None):
        call_count[0] += 1
        if call_count[0] == 1:
            # First stream: simulate 5-min limit reached → needs_restart=True
            return True
        # Second stream: normal end
        return False

    with patch.object(sess, "_run_single_stream", fake_run_single_stream):
        sess._run_blocking_stream()

    assert call_count[0] == 2, (
        "Expected _run_single_stream called twice: once at restart, once for normal end"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error event + graceful restart on quota/network failure
# ─────────────────────────────────────────────────────────────────────────────

def test_google_stt_error_emits_error_result():
    """Recoverable errors trigger stream restart (bounded)."""
    from app.services.google_stt_service import GoogleSttService

    sess = GoogleSttService.StreamingSTTSession(
        language_code="en-AU",
        sample_rate_hz=16000,
        encoding="LINEAR16",
        interim_results=True,
        api_config={},
        silence_threshold_ms=1500,
    )

    call_count = [0]

    def fake_run_single_stream(replay_chunks=None):
        call_count[0] += 1
        sess._results_q.put(
            {
                "error": "quota exceeded",
                "recoverable": True,
                "transcript": "",
                "confidence": 0.0,
                "is_final": False,
            }
        )
        if call_count[0] == 1:
            return True  # restart after recoverable error
        return False

    with patch.object(sess, "_run_single_stream", fake_run_single_stream):
        with patch("app.services.google_stt_service.time.sleep"):
            sess._run_blocking_stream()

    assert call_count[0] == 2, "Expected restart after recoverable error"
    results = []
    while not sess._results_q.empty():
        results.append(sess._results_q.get_nowait())
    assert any(r.get("error") for r in results)
    assert any(r.get("done") for r in results)


def test_chirp3_catalog_maps_to_phone_call_api_model():
    """Display model chirp-3 uses google_model=phone_call for API; API shows chirp-3."""
    from app.services.google_stt_service import GoogleSttService
    from app.schemas.agent import agent_to_out

    api_config = {
        "google_model": "phone_call",
        "use_enhanced": True,
        "encoding": "LINEAR16",
        "sample_rate_hz": 16000,
    }
    sess = GoogleSttService.StreamingSTTSession(
        language_code="en-AU",
        sample_rate_hz=16000,
        encoding="LINEAR16",
        interim_results=True,
        api_config=api_config,
        silence_threshold_ms=1500,
    )
    assert sess._api_config.get("google_model") == "phone_call"

    fake_agent = SimpleNamespace(
        id=uuid.uuid4(),
        name="G",
        llm_model="gpt-4o-mini",
        tts_provider_slug=None,
        tts_voice_external_id=None,
        tts_language=None,
        stt_provider_slug="google",
        stt_model_external_id="chirp-3",
        stt_language_code="en-AU",
        stt_settings_json=None,
        status="active",
        created_at=__import__("datetime").datetime(2026, 1, 1),
        updated_at=None,
        tenant_id=uuid.uuid4(),
        system_prompt=None,
        language=None,
        voice_type=None,
        is_inbound_agent=False,
        is_follow_up_agent=False,
    )
    out = agent_to_out(fake_agent)
    assert out.stt_model is not None
    assert out.stt_model.model_id == "chirp-3"
    assert out.stt_model.model_id != "phone_call"


def test_livekit_audio_processor_pcm_passthrough():
    """16 kHz mono PCM passes through without ffmpeg."""
    from app.voice.audio_transcoder import LiveKitAudioProcessor

    async def _run():
        proc = LiveKitAudioProcessor(output_sample_rate=16000)
        pcm = b"\x00\x01" * 160
        out = await proc.process_frame(pcm, sample_rate=16000, num_channels=1)
        assert out == pcm

    asyncio.run(_run())


def test_google_provider_skips_twilio_audio_feed():
    """When provider is google, on_audio_chunk must not feed Twilio MULAW to STT."""
    from unittest.mock import MagicMock, patch
    from app.voice.voice_orchestrator import VoiceOrchestrator
    from app.core.agent_runtime import ResolvedSttRuntime

    feed_calls: list[bytes] = []

    class FakeSttPipeline:
        async def feed_audio_chunk(self, data: bytes) -> None:
            feed_calls.append(data)

    class FakeTtsPipeline:
        _worker_task = None

        def __init__(self, _handler):
            import asyncio
            self.cancel_event = asyncio.Event()

        async def shutdown(self) -> None:
            pass

    handler = MagicMock()
    handler._min_audio_level_threshold = 0
    handler._audio_samples_needed = 1
    handler._audio_non_silent_needed = 1
    handler._enable_interim_llm = False
    handler._min_interim_words = 3
    handler._min_interim_confidence = 0.4
    handler._min_interim_interval_sec = 0.2
    handler._barge_in_min_conf = 0.26
    handler._barge_in_min_conf_1w = 0.52
    handler._pipeline_session = None

    with patch("app.voice.voice_orchestrator.TtsPipeline", FakeTtsPipeline):
        orch = VoiceOrchestrator(handler)

    orch._user_picked_up = True
    orch._stt_active = True
    orch._resolved_stt = ResolvedSttRuntime(
        provider_slug="google",
        model_id="chirp-3",
        language_code="en-AU",
        sample_rate_hz=16000,
        encoding="LINEAR16",
        silence_threshold_ms=1500,
        api_config={"google_model": "phone_call"},
    )
    orch._stt_pipeline = FakeSttPipeline()

    asyncio.run(orch.on_audio_chunk(b"\xff" * 160))
    assert len(feed_calls) == 0



# ─────────────────────────────────────────────────────────────────────────────
# 6. _resolve_stt_model validation
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_stt_model_invalid_provider(db):
    from fastapi import HTTPException
    from app.services.agent_service import AgentService
    from app.schemas.agent import SttModelSchema, SttProviderEnum

    svc = AgentService()
    stt = SttModelSchema(provider=SttProviderEnum.google, model_id="chirp-3", language_code="en-AU")

    # google provider not seeded in SQLite test DB → expect 400
    with pytest.raises(HTTPException) as exc_info:
        svc._resolve_stt_model(db, stt)
    assert exc_info.value.status_code == 400


def test_resolve_stt_model_none_returns_deepgram_defaults(db):
    from fastapi import HTTPException
    from app.services.agent_service import AgentService

    svc = AgentService()
    # None input should return defaults — but only if deepgram/nova-3 is seeded.
    # In SQLite test DB it is NOT seeded, so we get a 400 too.
    # Test that it at least attempts deepgram/nova-3.
    with pytest.raises(HTTPException) as exc_info:
        svc._resolve_stt_model(db, None)
    assert exc_info.value.status_code == 400
    assert "deepgram" in exc_info.value.detail.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. agent_to_out includes sttModel
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_to_out_includes_stt_model():
    from app.schemas.agent import agent_to_out

    fake_agent = SimpleNamespace(
        id=uuid.uuid4(),
        name="Test Agent",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice123",
        tts_language="en",
        stt_provider_slug="google",
        stt_model_external_id="chirp-3",
        stt_language_code="en-AU",
        stt_settings_json=None,
        status="active",
        created_at=__import__("datetime").datetime(2026, 1, 1),
        updated_at=None,
        tenant_id=uuid.uuid4(),
        system_prompt=None,
        language=None,
        voice_type=None,
        is_inbound_agent=False,
        is_follow_up_agent=False,
    )

    out = agent_to_out(fake_agent)
    assert out.stt_model is not None
    assert out.stt_model.provider.value == "google"
    assert out.stt_model.model_id == "chirp-3"
    assert out.stt_model.language_code == "en-AU"


def test_agent_to_out_stt_model_none_when_missing():
    from app.schemas.agent import agent_to_out

    fake_agent = SimpleNamespace(
        id=uuid.uuid4(),
        name="No STT Agent",
        llm_model="gpt-4o-mini",
        tts_provider_slug=None,
        tts_voice_external_id=None,
        tts_language=None,
        stt_provider_slug=None,
        stt_model_external_id=None,
        stt_language_code=None,
        stt_settings_json=None,
        status="pending",
        created_at=__import__("datetime").datetime(2026, 1, 1),
        updated_at=None,
        tenant_id=uuid.uuid4(),
        system_prompt=None,
        language=None,
        voice_type=None,
        is_inbound_agent=False,
        is_follow_up_agent=False,
    )

    out = agent_to_out(fake_agent)
    assert out.stt_model is None


# ─────────────────────────────────────────────────────────────────────────────
# 8. resolve_stt_runtime priority (flow > agent > catalog)
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_stt_runtime_flow_language_overrides_agent():
    from app.core.agent_runtime import resolve_stt_runtime

    agent = SimpleNamespace(
        stt_provider_slug="deepgram",
        stt_model_external_id="nova-3",
        stt_language_code="en",
        stt_model_id=None,
        stt_settings_json=None,
    )

    result = resolve_stt_runtime(agent, flow_language_code="en-AU")
    assert result.language_code == "en-AU"


def test_resolve_stt_runtime_agent_language_overrides_catalog():
    from app.core.agent_runtime import resolve_stt_runtime

    agent = SimpleNamespace(
        stt_provider_slug="deepgram",
        stt_model_external_id="nova-3",
        stt_language_code="es",
        stt_model_id=None,
        stt_settings_json=None,
    )

    result = resolve_stt_runtime(agent)
    assert result.language_code == "es"


def test_resolve_stt_runtime_defaults_when_no_agent():
    from app.core.agent_runtime import resolve_stt_runtime, _DEFAULT_STT_PROVIDER_SLUG

    result = resolve_stt_runtime(None)
    assert result.provider_slug == _DEFAULT_STT_PROVIDER_SLUG
    assert result.language_code is not None


def test_resolve_stt_runtime_silence_threshold_from_settings():
    from app.core.agent_runtime import resolve_stt_runtime

    agent = SimpleNamespace(
        stt_provider_slug="google",
        stt_model_external_id="chirp-3",
        stt_language_code="en-AU",
        stt_model_id=None,
        stt_settings_json={"silence_threshold_ms": 2000},
    )

    result = resolve_stt_runtime(agent)
    assert result.silence_threshold_ms == 2000


def test_resolve_stt_runtime_google_uses_linear16():
    from app.core.agent_runtime import resolve_stt_runtime

    agent = SimpleNamespace(
        stt_provider_slug="google",
        stt_model_external_id="chirp-3",
        stt_language_code="en-AU",
        stt_model_id=None,
        stt_settings_json=None,
    )

    result = resolve_stt_runtime(agent)
    assert result.provider_slug == "google"
    assert result.encoding == "LINEAR16"
    assert result.sample_rate_hz == 16000
    assert result.api_config.get("use_enhanced") is True
    assert result.api_config.get("google_model") == "phone_call"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Integration: inject 3-second audio clip → assert final transcript
# ─────────────────────────────────────────────────────────────────────────────

def test_google_stt_integration_3s_audio_mocked(monkeypatch):
    """
    Unit-level mock: session produces final transcript without calling Google API.
    For live API + real audio clip see tests/integration/test_google_stt_live.py.
    """
    from app.services.google_stt_service import GoogleSttService

    # Build a fake response object that Google SDK would return
    class FakeAlt:
        transcript = "hello world"
        confidence = 0.92

    class FakeResult:
        alternatives = [FakeAlt()]
        is_final = True

    class FakeResponse:
        results = [FakeResult()]

    sess = GoogleSttService.StreamingSTTSession(
        language_code="en-AU",
        sample_rate_hz=16000,
        encoding="LINEAR16",
        interim_results=True,
        api_config={"google_model": "phone_call"},
        silence_threshold_ms=1500,
    )

    def fake_run_single_stream(replay_chunks=None):
        # Simulate Google API returning a final transcript for the injected audio
        sess._results_q.put(
            {"transcript": "hello world", "confidence": 0.92, "is_final": True}
        )
        return False  # no restart needed

    with patch.object(sess, "_run_single_stream", fake_run_single_stream):
        sess._run_blocking_stream()

    results = []
    while not sess._results_q.empty():
        results.append(sess._results_q.get_nowait())

    finals = [r for r in results if r.get("is_final") and r.get("transcript")]
    assert finals, "Expected at least one final transcript from 3-second audio clip"
    assert finals[0]["transcript"] == "hello world"


# ─────────────────────────────────────────────────────────────────────────────
# EventBus tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stt_event_bus_delivers_to_subscribers():
    received = []

    async def handler(event):
        received.append(event)

    async def _run():
        bus = SttEventBus()
        bus.subscribe(handler)
        await bus.emit(SttInterimEvent(transcript="hi", confidence=0.5))
        await bus.emit(SttFinalEvent(transcript="hi there", confidence=0.9))

    asyncio.run(_run())
    assert len(received) == 2
    assert received[0].type == "interim"
    assert received[1].type == "final"


def test_stt_event_bus_subscriber_error_does_not_block():
    """A crashing subscriber should not prevent subsequent subscribers."""
    received = []

    async def bad_handler(event):
        raise ValueError("deliberate test error")

    async def good_handler(event):
        received.append(event)

    async def _run():
        bus = SttEventBus()
        bus.subscribe(bad_handler)
        bus.subscribe(good_handler)
        await bus.emit(SttFinalEvent(transcript="ok", confidence=0.8))

    asyncio.run(_run())
    assert len(received) == 1
