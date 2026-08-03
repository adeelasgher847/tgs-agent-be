import asyncio
from types import SimpleNamespace

from fastapi import WebSocketDisconnect

from app.voice.stt_pipeline import SttPipeline
from app.services.deepgram_stt_service import DeepgramSTTService
from app.routers import bidirectional_stream as bidi_module


def test_stt_pipeline_exits_reader_loop_on_done(monkeypatch):
    class FakeSession:
        def __init__(self):
            self._returned = False

        async def start(self):
            return None

        async def get_result(self):
            if not self._returned:
                self._returned = True
                return {"done": True}
            await asyncio.sleep(1)
            return {}

        def push_audio(self, _):
            return None

        def finish(self):
            return None

    fake_session = FakeSession()

    from app.services import deepgram_stt_service as dg_module

    monkeypatch.setattr(
        dg_module.deepgram_stt_service,
        "create_streaming_session",
        lambda **_: fake_session,
    )

    seen = {"interim": 0, "final": 0}

    async def on_interim(_, __):
        seen["interim"] += 1

    async def on_final(_, __):
        seen["final"] += 1

    async def _run():
        pipeline = SttPipeline(language_code="en-US", on_interim=on_interim, on_final=on_final)
        await pipeline.feed_audio_chunk(b"\x00")
        await asyncio.wait_for(pipeline._reader_task, timeout=0.5)  # type: ignore[arg-type]
        assert pipeline._reader_task.done()

    asyncio.run(_run())

    assert seen["interim"] == 0
    assert seen["final"] == 0


def test_feed_audio_chunk_after_aclose_does_not_reopen_session(monkeypatch):
    """
    Regression: a trailing/late-arriving audio chunk (e.g. an ffmpeg buffer
    flush racing the call's own shutdown sequence) fed to an already-closed
    SttPipeline used to lazily reopen a brand-new provider session via
    _ensure_session() — a session that could never receive further audio
    (the call had already ended and the LiveKit room had disconnected) and
    would just sit there until it timed out and errored minutes later
    (observed in production: "Deepgram did not receive audio data ... within
    the timeout window", ~12s after a call had already ended cleanly).
    aclose() must mark the pipeline closed so a subsequent feed_audio_chunk()
    is a no-op instead of opening a new session.
    """
    sessions_created = {"count": 0}

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

    def _create_session(**_):
        sessions_created["count"] += 1
        return FakeSession()

    from app.services import deepgram_stt_service as dg_module

    monkeypatch.setattr(
        dg_module.deepgram_stt_service, "create_streaming_session", _create_session
    )

    async def on_interim(_, __):
        return None

    async def on_final(_, __):
        return None

    async def _run():
        pipeline = SttPipeline(language_code="en-US", on_interim=on_interim, on_final=on_final)
        await pipeline.feed_audio_chunk(b"\x00\x01")
        assert sessions_created["count"] == 1

        await pipeline.aclose()

        # A trailing chunk arriving after shutdown must not reopen a session.
        await pipeline.feed_audio_chunk(b"\x00\x01")
        assert sessions_created["count"] == 1
        assert pipeline._stt_session is None

    asyncio.run(_run())


def test_bidirectional_disconnect_triggers_finish_session(monkeypatch):
    close_state = {"closed": False, "finished": 0}

    class FakeDB:
        def close(self):
            close_state["closed"] = True

    class FakeSttPipeline:
        def finish_session(self):
            close_state["finished"] += 1

    class FakeHandler:
        def __init__(self, **kwargs):
            self._stt_pipeline = FakeSttPipeline()
            self._stop_event = asyncio.Event()

        async def _full_shutdown(self):
            if self._stop_event.is_set():
                return
            self._stop_event.set()
            self._stt_pipeline.finish_session()

    class FakeWebSocket:
        async def accept(self):
            return None

        async def receive_text(self):
            raise WebSocketDisconnect()

    monkeypatch.setattr(bidi_module, "BidirectionalStreamHandler", FakeHandler)

    from app.db import session as db_session_module

    monkeypatch.setattr(db_session_module, "SessionLocal", lambda: FakeDB())

    async def _run():
        await bidi_module.bidirectional_stream_websocket(
            websocket=FakeWebSocket(),
            callSessionId="call-1",
            agentId="agent-1",
        )

    asyncio.run(_run())

    assert close_state["finished"] == 1
    assert close_state["closed"] is True


def test_prerecorded_mulaw_path_passes_sample_rate(monkeypatch):
    captured = {}

    class FakeMediaClient:
        def transcribe_file(self, **kwargs):
            captured.update(kwargs)
            alt = SimpleNamespace(transcript="hello", confidence=0.88)
            channel = SimpleNamespace(alternatives=[alt])
            results = SimpleNamespace(channels=[channel])
            return SimpleNamespace(results=results)

    class FakeClient:
        def __init__(self):
            self.listen = SimpleNamespace(v1=SimpleNamespace(media=FakeMediaClient()))

    from app.services import deepgram_stt_service as dg_module

    monkeypatch.setattr(dg_module.settings, "STT_SAMPLE_RATE", 8000, raising=False)

    svc = DeepgramSTTService()
    svc._client = FakeClient()

    out = svc._transcribe_sync(
        audio_content=b"\x00\x01\x02\x03",
        language_code="en-US",
        encoding=None,
        sample_rate=None,
    )

    assert out["transcript"] == "hello"
    assert captured.get("encoding") == "mulaw"
    assert captured.get("sample_rate") == 8000
