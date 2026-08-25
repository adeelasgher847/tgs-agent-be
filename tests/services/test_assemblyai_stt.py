"""AssemblyAI Universal-Streaming STT: session creation, two-state
end_of_turn transcript mapping (contrast with xAI's three-state
is_final/speech_final), no-in-band-error fatal-on-close handling, and the
Terminate -> Termination graceful shutdown flow.

No vendor SDK is used for wss://streaming.assemblyai.com/v3/ws -- this
adapter uses the raw websockets library directly, so tests exercise the
session's own message handlers/sender loop rather than a vendor connection
object's event API.
"""
import asyncio
import json

import pytest

from app.services.assemblyai_stt_service import AssemblyAiSTTService, _mask_key
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


def _make_session() -> "AssemblyAiSTTService.StreamingSTTSession":
    return AssemblyAiSTTService.StreamingSTTSession(
        api_key="test-key",
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        speech_model="universal-3-5-pro",
        min_turn_silence_ms=400,
        max_turn_silence_ms=1536,
        end_of_turn_confidence_threshold=0.4,
        vad_threshold=0.2,
        format_turns=True,
    )


# ── Authentication: key never logged raw ────────────────────────────────


def test_mask_key_never_reveals_the_raw_key():
    assert _mask_key(None) == "<empty>"
    assert _mask_key("") == "<empty>"
    masked = _mask_key("aai-abcdefghijklmnopqrstuvwxyz")
    assert "aai-abcdefghijklmnopqrstuvwxyz" not in masked
    assert masked.startswith("aai-")


# ── create_streaming_session ────────────────────────────────────────────


def test_create_streaming_session_requires_api_key():
    svc = AssemblyAiSTTService()
    svc._api_key = None
    with pytest.raises(RuntimeError):
        svc.create_streaming_session()


def test_create_streaming_session_uses_defaults_and_overrides():
    svc = AssemblyAiSTTService()
    svc._api_key = "test-key"

    default_session = svc.create_streaming_session()
    assert default_session._speech_model == "universal-streaming-multilingual"
    assert default_session._min_turn_silence_ms == 700
    assert default_session._max_turn_silence_ms == 1536
    assert default_session._end_of_turn_confidence_threshold == pytest.approx(0.4)
    assert default_session._vad_threshold == pytest.approx(0.4)
    assert default_session._format_turns is True
    assert default_session._encoding == "MULAW"

    overridden = svc.create_streaming_session(
        model="universal-streaming-multilingual",
        api_config={
            "min_turn_silence_ms": 800,
            "max_turn_silence_ms": 2000,
            "end_of_turn_confidence_threshold": 0.7,
            "vad_threshold": 0.5,
            "format_turns": False,
        },
    )
    assert overridden._speech_model == "universal-streaming-multilingual"
    assert overridden._min_turn_silence_ms == 800
    assert overridden._max_turn_silence_ms == 2000
    assert overridden._end_of_turn_confidence_threshold == pytest.approx(0.7)
    assert overridden._vad_threshold == pytest.approx(0.5)
    assert overridden._format_turns is False


def test_create_streaming_session_clamps_out_of_range_params():
    svc = AssemblyAiSTTService()
    svc._api_key = "test-key"

    session = svc.create_streaming_session(
        api_config={
            "min_turn_silence_ms": 1,  # docs range: 50-10000
            "max_turn_silence_ms": 999999,  # docs range: 50-10000 (mirrored)
            "end_of_turn_confidence_threshold": 5.0,  # docs range: 0.0-1.0
            "vad_threshold": -1.0,  # docs range: 0.0-1.0
        },
    )
    assert session._min_turn_silence_ms == 50
    assert session._max_turn_silence_ms == 10000
    assert session._end_of_turn_confidence_threshold == pytest.approx(1.0)
    assert session._vad_threshold == pytest.approx(0.0)


def test_create_streaming_session_forwards_model_as_speech_model():
    """Unlike xAI's `model`, which is accepted but never sent, AssemblyAI's
    `speech_model` IS a real, sent API parameter."""
    svc = AssemblyAiSTTService()
    svc._api_key = "test-key"

    session = svc.create_streaming_session(model="universal-streaming-english")
    assert session._speech_model == "universal-streaming-english"


# ── Two-state end_of_turn mapping (NOT three-state like xAI) ───────────


def test_interim_state_emits_interim_result():
    session = _make_session()
    session._handle_turn({"transcript": "hel", "end_of_turn": False})

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hel", "confidence": 0.0, "is_final": False}


def test_handle_turn_forwards_real_confidence_value():
    """confidence must be forwarded from AssemblyAI's payload, not hardcoded
    to 0.0 -- barge-in gates (VOICE_BARGE_IN_MIN_CONFIDENCE) depend on it."""
    session = _make_session()
    session._handle_turn({"transcript": "hello", "end_of_turn": False, "confidence": 0.87})

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hello", "confidence": pytest.approx(0.87), "is_final": False}


def test_end_of_turn_true_emits_pipeline_final_directly():
    """end_of_turn=true's transcript IS the finalized text for the turn --
    there is no separate withheld "chunk-final" state like xAI's
    (is_final=true, speech_final=false). This is the key behavioral
    contrast with the three-state xAI adapter."""
    session = _make_session()
    session._handle_turn(
        {"transcript": "I will buy two of those, please.", "end_of_turn": True, "end_of_turn_confidence": 0.98}
    )

    result = session._results_q.get_nowait()
    assert result == {"transcript": "I will buy two of those, please.", "confidence": 0.0, "is_final": True}


def test_no_third_withheld_state_exists():
    """Explicitly assert the simpler two-state behavior: every Turn message
    with a non-empty transcript is surfaced immediately, gated only by
    end_of_turn -- there is no intermediate state that gets dropped."""
    session = _make_session()
    session._handle_turn({"transcript": "partial one", "end_of_turn": False})
    session._handle_turn({"transcript": "partial one two", "end_of_turn": False})
    session._handle_turn({"transcript": "partial one two three", "end_of_turn": True})

    results = [session._results_q.get_nowait() for _ in range(3)]
    assert results[0]["is_final"] is False
    assert results[1]["is_final"] is False
    assert results[2]["is_final"] is True
    assert session._results_q.empty()


def test_empty_transcript_emits_nothing_regardless_of_end_of_turn():
    session = _make_session()
    session._handle_turn({"transcript": "", "end_of_turn": True})
    session._handle_turn({"transcript": "   ", "end_of_turn": False})
    assert session._results_q.empty()


def test_multiple_sequential_turns_each_finalize_independently():
    session = _make_session()
    session._handle_turn({"transcript": "first turn", "end_of_turn": True})
    session._handle_turn({"transcript": "second turn", "end_of_turn": True})

    first = session._results_q.get_nowait()
    second = session._results_q.get_nowait()
    assert first["transcript"] == "first turn"
    assert first["is_final"] is True
    assert second["transcript"] == "second turn"
    assert second["is_final"] is True


# ── Begin / Termination routing ─────────────────────────────────────────


def test_begin_sets_ready_event():
    session = _make_session()
    assert not session._ready_evt.is_set()
    session._handle_message({"type": "Begin", "id": "sess-1", "expires_at": 1234567890})
    assert session._ready_evt.is_set()


def test_termination_sets_termination_event_and_flag():
    session = _make_session()
    assert not session._termination_evt.is_set()
    session._handle_message(
        {"type": "Termination", "audio_duration_seconds": 4.2, "session_duration_seconds": 5.0}
    )
    assert session._termination_evt.is_set()
    assert session._termination_received is True


def test_turn_message_dispatches_through_handle_message():
    session = _make_session()
    session._handle_message({"type": "Turn", "transcript": "hello", "end_of_turn": True})

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hello", "confidence": 0.0, "is_final": True}


# ── No in-band error type: unexpected close before Termination is fatal ─


def test_unexpected_close_before_termination_is_treated_as_fatal():
    """Key behavioral contrast with xAI: there is no documented in-band
    recoverable error event here, so a socket that ends before Termination
    was observed must push a non-recoverable error + done, not a
    recoverable one."""
    session = _make_session()

    class _DeadWebSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    session._ws = _DeadWebSocket()

    asyncio.run(session._receiver_loop())

    error_result = session._results_q.get_nowait()
    assert error_result["recoverable"] is False
    assert "Termination" in error_result["error"]

    done_result = session._results_q.get_nowait()
    assert done_result == {"done": True}
    assert session._closed is True


def test_unrecognized_message_type_is_captured_not_silently_dropped():
    """Regression test: a live incident showed AssemblyAI closing with code
    3007 and reason 'See Error message for details' -- a generic reason that
    points at a separate JSON message sent just before the close. The old
    if/elif in _handle_message had no else branch, so that message (an
    undocumented type outside Begin/Turn/Termination) was silently dropped,
    leaving zero diagnostic detail in the logs. It must now be captured."""
    session = _make_session()
    session._handle_message({"type": "Error", "error": "invalid encoding parameter"})

    assert "Error" in session._last_server_message
    assert "invalid encoding parameter" in session._last_server_message
    # Must not be mistaken for Begin/Turn/Termination routing.
    assert not session._ready_evt.is_set()
    assert not session._termination_received
    assert session._results_q.empty()


def test_fatal_close_error_message_includes_captured_server_detail():
    """The pushed fatal-error result must surface whatever server message was
    captured before the close, not just a generic 'closed before
    Termination' string with no actionable detail."""
    session = _make_session()

    class _DeadWebSocketWithPriorErrorMessage:
        def __init__(self, session):
            self._session = session
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent:
                self._sent = True
                return json.dumps({"type": "Error", "error": "invalid encoding parameter"})
            raise StopAsyncIteration

    session._ws = _DeadWebSocketWithPriorErrorMessage(session)

    asyncio.run(session._receiver_loop())

    error_result = session._results_q.get_nowait()
    assert error_result["recoverable"] is False
    assert "invalid encoding parameter" in error_result["error"]


def test_close_after_termination_received_is_not_treated_as_fatal():
    session = _make_session()

    class _DeadWebSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    session._ws = _DeadWebSocket()
    session._termination_received = True

    asyncio.run(session._receiver_loop())

    assert session._results_q.empty()
    assert session._closed is False


# ── Graceful shutdown: Terminate -> wait for Termination -> close ──────


def test_sender_loop_sends_terminate_and_waits_for_termination():
    session = _make_session()
    ws = _FakeWebSocket()
    session._ws = ws

    async def _run():
        session._ready_evt.set()
        session.finish()

        async def _ack():
            await asyncio.sleep(0.05)
            session._termination_received = True
            session._termination_evt.set()

        asyncio.create_task(_ack())
        await session._sender_loop()

    asyncio.run(_run())

    types = ws.sent_json_types()
    assert types == ["Terminate"]
    assert ws.closed is True

    done = session._results_q.get_nowait()
    assert done == {"done": True}


def test_sender_loop_normal_exit_marks_session_closed():
    """Regression test: `_closed` must be set even on a graceful
    Terminate/Termination exit, not only on the exception branch --
    otherwise a straggler push_audio() after shutdown queues into a
    buffer nothing will ever drain."""
    session = _make_session()
    ws = _FakeWebSocket()
    session._ws = ws

    async def _run():
        session._ready_evt.set()
        session.finish()

        async def _ack():
            await asyncio.sleep(0.05)
            session._termination_received = True
            session._termination_evt.set()

        asyncio.create_task(_ack())
        await session._sender_loop()

    asyncio.run(_run())

    assert session._closed is True


def test_finish_while_utterance_pending_is_not_dropped():
    """Regression test for the Speechmatics-review class of bug: a pending
    final that arrives (via the receive loop) between Terminate being sent
    and Termination being observed must reach the pipeline, not be
    silently dropped by shutdown."""
    session = _make_session()
    ws = _FakeWebSocket()
    session._ws = ws

    async def _run():
        session._ready_evt.set()
        session.push_audio(b"\x01\x02")
        session.finish()

        async def _server_finalizes_then_terminates():
            await asyncio.sleep(0.02)
            # Server finalizes the still-open turn before Termination, per
            # AssemblyAI's documented Terminate semantics.
            session._handle_turn({"transcript": "trailing utterance", "end_of_turn": True})
            await asyncio.sleep(0.02)
            session._termination_received = True
            session._termination_evt.set()

        asyncio.create_task(_server_finalizes_then_terminates())
        await session._sender_loop()

    asyncio.run(_run())

    # The pending final must be in the queue ahead of the done sentinel.
    pending_final = session._results_q.get_nowait()
    assert pending_final == {"transcript": "trailing utterance", "confidence": 0.0, "is_final": True}

    done = session._results_q.get_nowait()
    assert done == {"done": True}


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
            session._termination_received = True
            session._termination_evt.set()

        asyncio.create_task(_ack())
        await session._sender_loop()

    asyncio.run(_run())

    # First sent item must be the raw audio bytes, not base64/JSON-wrapped.
    assert ws.sent[0] == b"\x01\x02\x03"


def test_send_failure_result_is_marked_non_recoverable():
    session = _make_session()

    class _FailingWebSocket:
        async def send(self, data):
            raise ConnectionError("boom")

        async def close(self):
            return None

    session._ws = _FailingWebSocket()

    async def _run():
        session._ready_evt.set()
        session.push_audio(b"\x01\x02\x03")
        session.finish()
        await session._sender_loop()

    asyncio.run(_run())

    error_result = session._results_q.get_nowait()
    assert error_result["error"] == "boom"
    assert error_result["recoverable"] is False

    done_result = session._results_q.get_nowait()
    assert done_result == {"done": True}


def test_sender_loop_unexpected_exception_is_marked_non_recoverable_and_closes():
    """Regression test: the broad catch-all around the sender loop's try body
    (ready-wait / audio-drain / Terminate-and-close sequence) must uphold the
    same 'always fatal' contract as the rest of this adapter -- a code-review
    finding caught this path defaulting to recoverable via
    SttPipeline's `recoverable = bool(result.get("recoverable", True))`."""
    session = _make_session()

    class _ExplodingQueue:
        async def get(self):
            raise RuntimeError("unexpected internal failure")

    session._audio_q = _ExplodingQueue()

    async def _run():
        session._ready_evt.set()
        await session._sender_loop()

    asyncio.run(_run())

    error_result = session._results_q.get_nowait()
    assert error_result["error"] == "unexpected internal failure"
    assert error_result["recoverable"] is False
    assert session._closed is True

    done_result = session._results_q.get_nowait()
    assert done_result == {"done": True}


def test_start_without_api_key_closes_session_so_push_audio_is_dropped():
    session = AssemblyAiSTTService.StreamingSTTSession(
        api_key=None,
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        speech_model="universal-3-5-pro",
        min_turn_silence_ms=400,
        max_turn_silence_ms=1536,
        end_of_turn_confidence_threshold=0.4,
        vad_threshold=0.2,
        format_turns=True,
    )

    asyncio.run(session.start())

    error_result = session._results_q.get_nowait()
    assert error_result["recoverable"] is False
    done_result = session._results_q.get_nowait()
    assert done_result == {"done": True}

    # push_audio() after the failed start() must be a no-op, not silently queued forever.
    session.push_audio(b"\x01\x02\x03")
    assert session._audio_q.qsize() == 0


def test_push_done_is_idempotent():
    session = _make_session()
    session._push_done()
    session._push_done()
    session._push_done()

    assert session._results_q.qsize() == 1
    assert session._results_q.get_nowait() == {"done": True}


# ── SttPipeline forwards model_id + api_config to the AssemblyAI service ─


def test_stt_pipeline_forwards_api_config_to_assemblyai(monkeypatch):
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

    from app.services import assemblyai_stt_service as assemblyai_module

    monkeypatch.setattr(
        assemblyai_module.assemblyai_stt_service,
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
            provider_slug="assemblyai",
            model_id="universal-3-5-pro",
            api_config={"vad_threshold": 0.5},
        )
        await pipeline.feed_audio_chunk(b"\x00")
        await pipeline.aclose()

    asyncio.run(_run())

    assert captured.get("api_config") == {"vad_threshold": 0.5}
    assert captured.get("model") == "universal-3-5-pro"
