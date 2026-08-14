"""xAI Grok STT: session creation, three-state transcript-flag mapping, Smart
Turn configuration, error handling (non-fatal per xAI's docs), and the
finalize -> audio.done graceful shutdown flow.

No vendor SDK exists for wss://api.x.ai/v1/stt -- this adapter uses the raw
websockets library directly, so tests exercise the session's own message
handlers/sender loop rather than a vendor connection object's event API.
"""
import asyncio
import json

import pytest

from app.services.xai_grok_stt_service import XaiGrokSTTService, _mask_key
from app.voice.stt_pipeline import SttPipeline


class _FakeWebSocket:
    """Minimal stand-in for the raw `websockets` connection object."""

    def __init__(self):
        self.sent: list = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    def sent_json_types(self) -> list:
        types = []
        for item in self.sent:
            if isinstance(item, str):
                try:
                    types.append(json.loads(item).get("type"))
                except (TypeError, ValueError):
                    pass
        return types


def _make_session() -> "XaiGrokSTTService.StreamingSTTSession":
    return XaiGrokSTTService.StreamingSTTSession(
        api_key="test-key",
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        endpointing_ms=10,
        smart_turn_threshold=0.5,
        smart_turn_timeout_ms=3000,
    )


# ── Authentication: key never logged raw ────────────────────────────────


def test_mask_key_never_reveals_the_raw_key():
    assert _mask_key(None) == "<empty>"
    assert _mask_key("") == "<empty>"
    masked = _mask_key("xai-abcdefghijklmnopqrstuvwxyz")
    assert "xai-abcdefghijklmnopqrstuvwxyz" not in masked
    assert masked.startswith("xai-")


# ── create_streaming_session ────────────────────────────────────────────


def test_create_streaming_session_requires_api_key():
    svc = XaiGrokSTTService()
    svc._api_key = None
    with pytest.raises(RuntimeError):
        svc.create_streaming_session()


def test_create_streaming_session_uses_defaults_and_overrides():
    svc = XaiGrokSTTService()
    svc._api_key = "test-key"

    default_session = svc.create_streaming_session()
    assert default_session._smart_turn_threshold == pytest.approx(0.5)
    assert default_session._smart_turn_timeout_ms == 3000
    assert default_session._encoding == "MULAW"

    overridden = svc.create_streaming_session(
        api_config={"smart_turn_threshold": 0.7, "smart_turn_timeout_ms": 2000, "endpointing_ms": 200},
    )
    assert overridden._smart_turn_threshold == pytest.approx(0.7)
    assert overridden._smart_turn_timeout_ms == 2000
    assert overridden._endpointing_ms == 200


def test_create_streaming_session_clamps_out_of_range_params():
    svc = XaiGrokSTTService()
    svc._api_key = "test-key"

    session = svc.create_streaming_session(
        api_config={
            "smart_turn_threshold": 5.0,  # docs range: 0.0-1.0
            "smart_turn_timeout_ms": 99999,  # docs range: 1-5000
            "endpointing_ms": -5,  # docs range: 0-5000
        },
    )
    assert session._smart_turn_threshold == pytest.approx(1.0)
    assert session._smart_turn_timeout_ms == 5000
    assert session._endpointing_ms == 0


def test_create_streaming_session_never_sends_model_param(monkeypatch):
    """xAI's STT API has no model parameter at all -- model must be accepted
    (for interface parity with SttPipeline's dispatch) but never forwarded."""
    svc = XaiGrokSTTService()
    svc._api_key = "test-key"

    session = svc.create_streaming_session(model="some-model-id-should-be-ignored")
    assert not hasattr(session, "_model")


# ── Three-state is_final/speech_final mapping ───────────────────────────


def test_interim_state_emits_interim_result():
    session = _make_session()
    session._handle_partial({"text": "hel", "is_final": False, "speech_final": False})

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hel", "confidence": 0.0, "is_final": False}
    assert session._audio_since_final is True


def test_chunk_final_state_is_withheld_from_pipeline():
    """(is_final=true, speech_final=false) -- text locked for this window but
    NOT a turn boundary. Must not spuriously trigger an LLM turn."""
    session = _make_session()
    session._handle_partial({"text": "I will buy two", "is_final": True, "speech_final": False})

    assert session._results_q.empty()
    assert session._audio_since_final is True


def test_utterance_final_state_emits_pipeline_final():
    """(is_final=true, speech_final=true) -- the real turn boundary."""
    session = _make_session()
    session._handle_partial(
        {"text": "I will buy two of those, please.", "is_final": True, "speech_final": True, "end_of_turn_confidence": 0.98}
    )

    result = session._results_q.get_nowait()
    assert result == {"transcript": "I will buy two of those, please.", "confidence": 0.0, "is_final": True}
    assert session._audio_since_final is False


def test_empty_text_emits_nothing_regardless_of_flags():
    session = _make_session()
    session._handle_partial({"text": "", "is_final": True, "speech_final": True})
    assert session._results_q.empty()


# ── transcript.created / transcript.done / error routing ───────────────


def test_transcript_created_sets_ready_event():
    session = _make_session()
    assert not session._ready_evt.is_set()
    session._handle_message({"type": "transcript.created"})
    assert session._ready_evt.is_set()


def test_transcript_done_sets_done_event():
    session = _make_session()
    assert not session._transcript_done_evt.is_set()
    session._handle_message({"type": "transcript.done", "text": "full transcript", "duration": 4.2})
    assert session._transcript_done_evt.is_set()


def test_error_event_is_recoverable_and_does_not_close_session():
    """Per xAI's docs, the connection stays open after an error -- unlike
    every other provider here, this must NOT be treated as fatal."""
    session = _make_session()
    session._handle_message({"type": "error", "message": "transient backend hiccup"})

    result = session._results_q.get_nowait()
    assert result["error"] == "transient backend hiccup"
    assert result["recoverable"] is True
    # Session must remain open -- no done sentinel, no _closed flag flip.
    assert session._results_q.empty()
    assert session._closed is False


# ── Graceful shutdown: finalize -> audio.done ───────────────────────────


def test_sender_loop_sends_finalize_then_audio_done_when_pending():
    session = _make_session()
    ws = _FakeWebSocket()
    session._ws = ws

    async def _run():
        session._ready_evt.set()
        session._audio_since_final = True  # speech pushed, no speech_final yet
        session.finish()
        # Simulate the server acking audio.done promptly so the wait doesn't time out.
        async def _ack():
            await asyncio.sleep(0.05)
            session._transcript_done_evt.set()
        asyncio.create_task(_ack())
        await session._sender_loop()

    asyncio.run(_run())

    types = ws.sent_json_types()
    assert types == ["finalize", "audio.done"]
    assert ws.closed is True

    done = session._results_q.get_nowait()
    assert done == {"done": True}


def test_sender_loop_skips_finalize_when_nothing_pending():
    session = _make_session()
    ws = _FakeWebSocket()
    session._ws = ws

    async def _run():
        session._ready_evt.set()
        session._audio_since_final = False  # last segment already speech_final
        session.finish()
        async def _ack():
            await asyncio.sleep(0.05)
            session._transcript_done_evt.set()
        asyncio.create_task(_ack())
        await session._sender_loop()

    asyncio.run(_run())

    types = ws.sent_json_types()
    assert types == ["audio.done"]
    assert ws.closed is True


def test_sender_loop_forwards_audio_chunks_as_raw_binary():
    session = _make_session()
    ws = _FakeWebSocket()
    session._ws = ws

    async def _run():
        session._ready_evt.set()
        session.push_audio(b"\x01\x02\x03")
        session.finish()
        async def _ack():
            await asyncio.sleep(0.05)
            session._transcript_done_evt.set()
        asyncio.create_task(_ack())
        await session._sender_loop()

    asyncio.run(_run())

    # First sent item must be the raw audio bytes, not base64/JSON-wrapped.
    assert ws.sent[0] == b"\x01\x02\x03"


def test_push_done_is_idempotent():
    session = _make_session()
    session._push_done()
    session._push_done()
    session._push_done()

    assert session._results_q.qsize() == 1
    assert session._results_q.get_nowait() == {"done": True}


# ── SttPipeline forwards model_id + api_config to the xAI service ──────


def test_stt_pipeline_forwards_api_config_to_xai(monkeypatch):
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

    from app.services import xai_grok_stt_service as xai_module

    monkeypatch.setattr(
        xai_module.xai_grok_stt_service,
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
            provider_slug="xai",
            model_id="default",
            api_config={"smart_turn_threshold": 0.7},
        )
        await pipeline.feed_audio_chunk(b"\x00")
        await pipeline.aclose()

    asyncio.run(_run())

    assert captured.get("api_config") == {"smart_turn_threshold": 0.7}
