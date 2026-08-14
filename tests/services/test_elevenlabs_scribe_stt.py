"""ElevenLabs Scribe v2 Realtime STT: session creation, event-translation, and
VAD-commit-gated turn-final mapping into SttPipeline's result-dict contract.

Wire field for transcript text is "text" (verified against the raw AsyncAPI
spec) -- the SDK's own docstring examples showing "transcript" are stale.
"""
import asyncio

import pytest

from app.services.elevenlabs_scribe_stt_service import ElevenLabsScribeSTTService
from app.voice.stt_pipeline import SttPipeline
from elevenlabs.realtime import RealtimeEvents


class _FakeConnection:
    """Minimal stand-in for elevenlabs.realtime.RealtimeConnection's .on() API."""

    def __init__(self):
        self._handlers: dict = {}
        self.committed = False
        self.closed = False

    def on(self, event, callback):
        self._handlers.setdefault(event, []).append(callback)

    def emit(self, event, data):
        for handler in self._handlers.get(event, []):
            handler(data)

    async def send(self, data):
        return None

    async def commit(self):
        self.committed = True

    async def close(self):
        self.closed = True


def _partial_msg(text: str) -> dict:
    return {"message_type": "partial_transcript", "text": text}


def _committed_msg(text: str) -> dict:
    return {"message_type": "committed_transcript", "text": text}


# ── create_streaming_session ──────────────────────────────────────────────


def test_create_streaming_session_requires_api_key():
    svc = ElevenLabsScribeSTTService()
    svc._api_key = None
    with pytest.raises(RuntimeError):
        svc.create_streaming_session()


def test_create_streaming_session_uses_defaults_and_overrides():
    svc = ElevenLabsScribeSTTService()
    svc._api_key = "test-key"

    default_session = svc.create_streaming_session()
    assert default_session._model == "scribe_v2_realtime"
    assert default_session._vad_silence_threshold_secs == pytest.approx(1.5)
    assert default_session._vad_threshold == pytest.approx(0.4)
    assert default_session._encoding == "MULAW"

    overridden = svc.create_streaming_session(
        api_config={"vad_silence_threshold_secs": 0.8, "vad_threshold": 0.6, "min_speech_duration_ms": 200},
    )
    assert overridden._vad_silence_threshold_secs == pytest.approx(0.8)
    assert overridden._vad_threshold == pytest.approx(0.6)
    assert overridden._min_speech_duration_ms == 200


# ── Event translation: partial / committed (single-event turn-final) ───────


def _make_session() -> "ElevenLabsScribeSTTService.StreamingSTTSession":
    return ElevenLabsScribeSTTService.StreamingSTTSession(
        api_key="test-key",
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        model="scribe_v2_realtime",
        vad_silence_threshold_secs=1.5,
        vad_threshold=0.4,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
    )


def test_partial_transcript_emits_interim_result_reading_text_field():
    session = _make_session()
    conn = _FakeConnection()
    session._register_handlers(conn)

    conn.emit(RealtimeEvents.PARTIAL_TRANSCRIPT, _partial_msg("hel"))

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hel", "confidence": 0.0, "is_final": False}
    assert session._audio_since_commit is True


def test_committed_transcript_emits_final_result_directly_no_accumulation():
    """Unlike Speechmatics, ElevenLabs delivers the finalized segment directly
    on the commit event -- no separate accumulate-then-flush step needed."""
    session = _make_session()
    conn = _FakeConnection()
    session._register_handlers(conn)

    conn.emit(RealtimeEvents.PARTIAL_TRANSCRIPT, _partial_msg("hello there"))
    conn.emit(RealtimeEvents.COMMITTED_TRANSCRIPT, _committed_msg("hello there"))

    # Drain the interim first.
    interim = session._results_q.get_nowait()
    assert interim["is_final"] is False

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hello there", "confidence": 0.0, "is_final": True}
    assert session._audio_since_commit is False


def test_committed_transcript_with_empty_text_emits_nothing():
    """A VAD commit on pure silence/noise must not trigger a spurious LLM turn."""
    session = _make_session()
    conn = _FakeConnection()
    session._register_handlers(conn)

    conn.emit(RealtimeEvents.COMMITTED_TRANSCRIPT, _committed_msg(""))

    assert session._results_q.empty()


def test_server_error_reads_error_field_not_reason_and_stops_session():
    """Verified against the raw AsyncAPI spec: error payloads use "error", not
    Speechmatics' "reason" -- this would silently produce "unknown" if wrong."""
    session = _make_session()
    conn = _FakeConnection()
    session._register_handlers(conn)

    conn.emit(RealtimeEvents.ERROR, {"message_type": "auth_error", "error": "invalid api key"})

    error_result = session._results_q.get_nowait()
    assert error_result["error"] == "invalid api key"
    assert error_result["is_final"] is True
    assert error_result["recoverable"] is False

    done_result = session._results_q.get_nowait()
    assert done_result == {"done": True}
    assert session._closed is True


# ── Forced commit on close (no flush-on-close primitive in this SDK) ───────


def test_sender_loop_force_commits_pending_audio_before_close():
    """The SDK has no equivalent of Speechmatics' stop_session() flush. A
    caller's final utterance may not have crossed the VAD silence threshold
    yet when the call tears down -- finish() must force a commit so it isn't
    silently dropped."""
    session = _make_session()
    conn = _FakeConnection()
    session._connection = conn
    session._register_handlers(conn)

    async def _run():
        session._audio_since_commit = True  # audio was pushed, no commit yet
        session.finish()
        await session._sender_loop()

    asyncio.run(_run())

    assert conn.committed is True
    assert conn.closed is True

    done = session._results_q.get_nowait()
    assert done == {"done": True}


def test_sender_loop_skips_forced_commit_when_nothing_pending():
    session = _make_session()
    conn = _FakeConnection()
    session._connection = conn
    session._register_handlers(conn)

    async def _run():
        session._audio_since_commit = False  # last segment already committed
        session.finish()
        await session._sender_loop()

    asyncio.run(_run())

    assert conn.committed is False
    assert conn.closed is True


# ── SttPipeline forwards model_id + api_config to the ElevenLabs service ───


def test_stt_pipeline_forwards_model_and_api_config_to_elevenlabs(monkeypatch):
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

    from app.services import elevenlabs_scribe_stt_service as el_module

    monkeypatch.setattr(
        el_module.elevenlabs_scribe_stt_service,
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
            provider_slug="elevenlabs",
            model_id="scribe_v2_realtime",
            api_config={"vad_silence_threshold_secs": 1.0},
        )
        await pipeline.feed_audio_chunk(b"\x00")
        await pipeline.aclose()

    asyncio.run(_run())

    assert captured.get("model") == "scribe_v2_realtime"
    assert captured.get("api_config") == {"vad_silence_threshold_secs": 1.0}
