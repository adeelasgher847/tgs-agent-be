"""Speechmatics STT: session creation, event-translation, and EndOfUtterance-gated
turn-final mapping into SttPipeline's result-dict contract."""
import asyncio

import pytest

from app.services.speechmatics_stt_service import SpeechmaticsSTTService
from app.voice.stt_pipeline import SttPipeline
from speechmatics.rt import ServerMessageType


class _FakeClient:
    """Minimal stand-in for speechmatics.rt.AsyncClient's event-registration API."""

    def __init__(self):
        self._handlers: dict = {}

    def on(self, event):
        def decorator(fn):
            self._handlers[event] = fn
            return fn
        return decorator

    def emit(self, event, msg):
        handler = self._handlers.get(event)
        assert handler is not None, f"no handler registered for {event}"
        handler(msg)


def _partial_msg(transcript: str) -> dict:
    return {"message": "AddPartialTranscript", "metadata": {"transcript": transcript}, "results": []}


def _transcript_msg(transcript: str, confidence: float) -> dict:
    return {
        "message": "AddTranscript",
        "metadata": {"transcript": transcript},
        "results": [
            {"type": "word", "alternatives": [{"content": transcript, "confidence": confidence}]}
        ],
    }


# ── create_streaming_session ──────────────────────────────────────────────


def test_create_streaming_session_requires_api_key():
    svc = SpeechmaticsSTTService()
    svc._api_key = None
    with pytest.raises(RuntimeError):
        svc.create_streaming_session()


def test_create_streaming_session_uses_defaults_and_overrides():
    svc = SpeechmaticsSTTService()
    svc._api_key = "test-key"

    default_session = svc.create_streaming_session()
    assert default_session._eot_silence_trigger == pytest.approx(0.5)
    assert default_session._max_delay == pytest.approx(2.0)
    assert default_session._encoding == "MULAW"

    overridden = svc.create_streaming_session(
        api_config={"end_of_utterance_silence_trigger": 0.8, "max_delay": 3.0, "model": "standard"},
    )
    assert overridden._eot_silence_trigger == pytest.approx(0.8)
    assert overridden._max_delay == pytest.approx(3.0)
    assert overridden._model == "standard"


# ── Event translation: partial / AddTranscript-accumulate / EndOfUtterance ──


def _make_session() -> "SpeechmaticsSTTService.StreamingSTTSession":
    return SpeechmaticsSTTService.StreamingSTTSession(
        api_key="test-key",
        url=None,
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        model="enhanced",
        end_of_utterance_silence_trigger=0.5,
        max_delay=2.0,
    )


def test_partial_transcript_emits_interim_result():
    session = _make_session()
    client = _FakeClient()
    session._register_handlers(client)

    client.emit(ServerMessageType.ADD_PARTIAL_TRANSCRIPT, _partial_msg("hel"))

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hel", "confidence": 0.0, "is_final": False}


def test_add_transcript_is_withheld_until_end_of_utterance():
    session = _make_session()
    client = _FakeClient()
    session._register_handlers(client)

    client.emit(ServerMessageType.ADD_TRANSCRIPT, _transcript_msg("hello", 0.9))

    # AddTranscript alone must not surface a pipeline result -- native EOU gates it.
    assert session._results_q.empty()
    assert session._pending_transcript == "hello"


def test_end_of_utterance_emits_accumulated_final_result():
    session = _make_session()
    client = _FakeClient()
    session._register_handlers(client)

    client.emit(ServerMessageType.ADD_TRANSCRIPT, _transcript_msg("hello", 0.9))
    client.emit(ServerMessageType.ADD_TRANSCRIPT, _transcript_msg("there", 0.7))
    client.emit(ServerMessageType.END_OF_UTTERANCE, {"message": "EndOfUtterance"})

    result = session._results_q.get_nowait()
    assert result["transcript"] == "hello there"
    assert result["is_final"] is True
    # Confidence reflects the latest accumulated segment (mirrors Deepgram's adapter).
    assert result["confidence"] == pytest.approx(0.7)
    # Pending buffer must reset so the next utterance starts clean.
    assert session._pending_transcript == ""


def test_end_of_utterance_without_prior_transcript_emits_nothing():
    session = _make_session()
    client = _FakeClient()
    session._register_handlers(client)

    client.emit(ServerMessageType.END_OF_UTTERANCE, {"message": "EndOfUtterance"})

    assert session._results_q.empty()


def test_sender_loop_flushes_pending_transcript_on_close_without_end_of_utterance():
    """A caller's final utterance may finalize (AddTranscript) right as the call
    tears down, before Speechmatics' own silence timer ever fires EndOfUtterance.
    finish()/aclose() must not silently drop it."""
    session = _make_session()
    client = _FakeClient()
    session._register_handlers(client)
    session._client = client

    async def _fake_stop_session():
        return None

    client.stop_session = _fake_stop_session

    # AddTranscript arrives but EndOfUtterance never does (call ends first).
    client.emit(ServerMessageType.ADD_TRANSCRIPT, _transcript_msg("book it", 0.85))
    assert session._results_q.empty()

    async def _run():
        session.finish()  # pushes the None sentinel _sender_loop is waiting on
        await session._sender_loop()

    asyncio.run(_run())

    flushed = session._results_q.get_nowait()
    assert flushed == {"transcript": "book it", "confidence": 0.85, "is_final": True}

    done = session._results_q.get_nowait()
    assert done == {"done": True}


def test_server_error_emits_error_then_done_and_stops_session():
    session = _make_session()
    client = _FakeClient()
    session._register_handlers(client)

    client.emit(ServerMessageType.ERROR, {"message": "Error", "reason": "boom"})

    error_result = session._results_q.get_nowait()
    assert error_result["error"] == "boom"
    assert error_result["is_final"] is True
    assert error_result["recoverable"] is False

    done_result = session._results_q.get_nowait()
    assert done_result == {"done": True}
    assert session._closed is True


# ── SttPipeline forwards model_id + api_config to the Speechmatics service ──


def test_stt_pipeline_forwards_model_and_api_config_to_speechmatics(monkeypatch):
    captured = {}

    def fake_create_streaming_session(**kwargs):
        captured.update(kwargs)

        class FakeSession:
            async def start(self):
                return None

            async def get_result(self):
                await asyncio.sleep(1)
                return {}

            def push_audio(self, _):
                return None

            def finish(self):
                return None

        return FakeSession()

    from app.services import speechmatics_stt_service as sm_module

    monkeypatch.setattr(
        sm_module.speechmatics_stt_service,
        "create_streaming_session",
        fake_create_streaming_session,
    )

    async def on_interim(_, __):
        return None

    async def on_final(_, __):
        return None

    async def _run():
        pipeline = SttPipeline(
            language_code="en",
            on_interim=on_interim,
            on_final=on_final,
            provider_slug="speechmatics",
            model_id="enhanced",
            api_config={"end_of_utterance_silence_trigger": 0.8},
        )
        await pipeline.feed_audio_chunk(b"\x00")
        await pipeline.aclose()

    asyncio.run(_run())

    assert captured.get("model") == "enhanced"
    assert captured.get("api_config") == {"end_of_utterance_silence_trigger": 0.8}
