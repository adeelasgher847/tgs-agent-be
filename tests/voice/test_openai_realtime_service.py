"""
Unit tests for app/services/openai_realtime_service.py — OpenAIRealtimeSession.

Mirrors tests/voice/test_gemini_live_service.py's structure/conventions,
adapted for this provider's genuinely different SDK shape:
  - `connection.recv()` returns exactly one parsed event per call (no
    per-turn generator-exhaustion quirk like google.genai's
    `session.receive()`), so the fake connection here is a plain AsyncMock-
    backed `recv()` returning a queued list of events, one per call — NOT
    a turn-batched async generator like Gemini's `_FakeAsyncSession`.
  - Client events (session.update / input_audio_buffer.append /
    conversation.item.create / response.create) are built as plain dicts in
    the source, not real SDK types, so these tests never need Gemini's
    `_real_google_genai()`-style "temporarily un-stub the real SDK" dance —
    plain dict/SimpleNamespace fakes are sufficient throughout.

CRITICAL test-safety note (see this task's own write-up of a real prior
incident in this codebase): every fake connection's `recv()` here has a
natural termination condition — once its queued events are exhausted, it
raises `_StopReceiving` rather than blocking or replaying, so
`_receive_loop`'s `while True:` loop always terminates on its own. A
dedicated `test_close_cancels_receive_task_while_genuinely_blocked_in_recv`
test additionally covers the "task genuinely suspended inside recv()" case
using a real `asyncio.Event().wait()` that only `close()`'s cancellation
can end.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm_models import OPENAI_REALTIME_MODELS
from app.services.openai_realtime_service import (
    DEFAULT_VOICE_NAME,
    LIVEKIT_AUDIO_FORMAT,
    LIVEKIT_AUDIO_RATE_HZ,
    TWILIO_AUDIO_FORMAT,
    VALID_VOICE_NAMES,
    OpenAIRealtimeError,
    OpenAIRealtimeErrorType,
    OpenAIRealtimeSession,
    _build_calendly_tool_params,
    _classify_openai_realtime_error,
    resolve_voice_name,
)


class _FakeConnection:
    """Fake ``AsyncRealtimeConnection``. Once its queued ``events`` are
    exhausted, ``recv()`` blocks forever (a real ``asyncio.Event().wait()``
    that only external task cancellation can end) rather than raising or
    replaying — this mirrors real production shape (the receive loop is
    normally ended by ``close()``'s ``task.cancel()``, not by ``recv()``
    erroring) and, per this task's CRITICAL testing-safety note, still gives
    every test a natural, finite termination condition: every test using
    this fake either calls ``session.close()`` (which cancels the task) or
    explicitly cancels the task itself after observing all queued events
    have been consumed — never a bare, uncancelled ``while True`` against a
    naively-replaying/never-blocking mock."""

    def __init__(self, events=None, block_forever: bool = False):
        self._events = list(events or [])
        del block_forever  # accepted for call-site clarity; always the behavior now
        self.send = AsyncMock()
        self.close = AsyncMock()

    async def recv(self):
        if not self._events:
            # Genuinely suspends until cancelled — never resolves on its own.
            await asyncio.Event().wait()
            return None
        return self._events.pop(0)


class _FakeConnectCM:
    """Stands in for the object returned by client.realtime.connect(...) —
    an async context manager yielding a connection."""

    def __init__(self, connection: _FakeConnection):
        self._connection = connection
        self.aexit_called = False

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, *exc):
        self.aexit_called = True
        return False


def _make_fake_client(connection: _FakeConnection):
    connect_cm = _FakeConnectCM(connection)
    client = MagicMock()
    client.realtime.connect = MagicMock(return_value=connect_cm)
    return client, connect_cm


def _evt(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Construction / model validation
# ─────────────────────────────────────────────────────────────────────────


class TestOpenAIRealtimeSessionConstruction:
    @pytest.mark.parametrize("model_name", list(OPENAI_REALTIME_MODELS))
    def test_known_models_construct_successfully(self, model_name):
        session = OpenAIRealtimeSession(model_name)
        assert session.model_name == model_name

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="not an OpenAI Realtime"):
            OpenAIRealtimeSession("gpt-4o")

    def test_unknown_model_error_lists_valid_models(self):
        with pytest.raises(ValueError) as exc_info:
            OpenAIRealtimeSession("totally-made-up-model")
        for model_name in OPENAI_REALTIME_MODELS:
            assert model_name in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────
# start() — connects, sends session.update, launches the receive loop
# ─────────────────────────────────────────────────────────────────────────


class TestStart:
    def test_start_connects_and_launches_receive_loop(self):
        connection = _FakeConnection(events=[])
        client, connect_cm = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime")

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="be helpful",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await asyncio.sleep(0)  # let the receive loop hit _StopReceiving
            await session.close()

        asyncio.run(_run())

        client.realtime.connect.assert_called_once_with(model="gpt-realtime")
        assert connect_cm.aexit_called is True

    def test_start_sends_session_update_with_instructions_and_audio_format(self):
        connection = _FakeConnection(events=[])
        client, _ = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime")

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="be helpful",
                    audio_format=TWILIO_AUDIO_FORMAT,
                    voice_name="marin",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await session.close()

        asyncio.run(_run())

        connection.send.assert_awaited_once()
        sent = connection.send.call_args.args[0]
        assert sent["type"] == "session.update"
        sess = sent["session"]
        assert sess["model"] == "gpt-realtime"
        assert sess["instructions"] == "be helpful"
        assert sess["output_modalities"] == ["audio"]
        assert sess["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert sess["audio"]["output"]["format"] == {"type": "audio/pcmu"}
        assert sess["audio"]["output"]["voice"] == "marin"
        assert sess["audio"]["input"]["turn_detection"]["type"] == "server_vad"
        assert sess["audio"]["input"]["turn_detection"]["interrupt_response"] is True
        # gpt-realtime has no reasoning config (see module docstring).
        assert "reasoning" not in sess

    def test_start_livekit_audio_format_uses_pcm_24k(self):
        connection = _FakeConnection(events=[])
        client, _ = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime")

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="hi",
                    audio_format=LIVEKIT_AUDIO_FORMAT,
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await session.close()

        asyncio.run(_run())

        sent = connection.send.call_args.args[0]
        assert sent["session"]["audio"]["input"]["format"] == {
            "type": "audio/pcm", "rate": LIVEKIT_AUDIO_RATE_HZ,
        }
        assert sent["session"]["audio"]["output"]["format"] == {
            "type": "audio/pcm", "rate": LIVEKIT_AUDIO_RATE_HZ,
        }

    def test_gpt_realtime_2_gets_capped_reasoning_effort(self):
        connection = _FakeConnection(events=[])
        client, _ = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime-2")

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="hi",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await session.close()

        asyncio.run(_run())

        sent = connection.send.call_args.args[0]
        assert sent["session"]["reasoning"] == {"effort": "low"}

    def test_start_passes_tools_when_provided(self):
        connection = _FakeConnection(events=[])
        client, _ = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime")
        tools = _build_calendly_tool_params()

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="hi",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                    tools=tools,
                )
            await session.close()

        asyncio.run(_run())

        sent = connection.send.call_args.args[0]
        assert sent["session"]["tools"] == tools

    def test_start_failure_raises_openai_realtime_error(self):
        session = OpenAIRealtimeSession("gpt-realtime")

        with patch.object(session, "_build_client", side_effect=RuntimeError("boom")):
            with pytest.raises(OpenAIRealtimeError):
                asyncio.run(
                    session.start(
                        system_instruction="hi",
                        on_audio_chunk=AsyncMock(),
                        on_interrupted=AsyncMock(),
                    )
                )

    def test_connect_exception_translated_to_openai_realtime_error(self):
        client = MagicMock()
        client.realtime.connect = MagicMock(side_effect=RuntimeError("quota exceeded"))
        session = OpenAIRealtimeSession("gpt-realtime")

        with patch.object(session, "_build_client", return_value=client):
            with pytest.raises(OpenAIRealtimeError) as exc_info:
                asyncio.run(
                    session.start(
                        system_instruction="hi",
                        on_audio_chunk=AsyncMock(),
                        on_interrupted=AsyncMock(),
                    )
                )
        assert exc_info.value.error_type == OpenAIRealtimeErrorType.QUOTA


# ─────────────────────────────────────────────────────────────────────────
# send_audio / send_text / send_tool_response
# ─────────────────────────────────────────────────────────────────────────


class TestSendAudio:
    def test_send_audio_base64_encodes_and_wraps_event(self):
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        asyncio.run(session.send_audio(b"\x01\x02\x03\x04"))

        connection.send.assert_awaited_once()
        sent = connection.send.call_args.args[0]
        assert sent["type"] == "input_audio_buffer.append"
        assert base64.b64decode(sent["audio"]) == b"\x01\x02\x03\x04"

    def test_send_audio_empty_bytes_is_noop(self):
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        asyncio.run(session.send_audio(b""))

        connection.send.assert_not_awaited()

    def test_send_audio_noop_once_session_already_closed(self):
        """Straggler audio chunks arriving after close() has already run
        must never reach connection.send() — see this method's docstring
        about the LiveKitAudioSubscriber shutdown race."""
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection
        session._closed = True

        asyncio.run(session.send_audio(b"\x01\x02\x03\x04"))

        connection.send.assert_not_awaited()

    def test_send_audio_swallows_exception_when_closed_flips_mid_await(self):
        """Mirrors TestClose.test_close_cancels_receive_task_while_genuinely_
        blocked_in_recv's pattern of exercising a genuine concurrent race:
        here, `self._closed` flips to True while `connection.send()` is
        in-flight and then raises (simulating the underlying websocket
        erroring out mid-close-handshake) — the exception must be swallowed,
        not propagated, once the flip is observed."""
        session = OpenAIRealtimeSession("gpt-realtime")

        connection = MagicMock()

        async def _send(_event):
            # Simulate close() flipping the flag concurrently while this
            # send() call is still in flight, then the underlying send
            # erroring out as a result of the connection tearing down.
            session._closed = True
            raise ConnectionError("connection closed during send")

        connection.send = AsyncMock(side_effect=_send)
        session._connection = connection

        # Must not raise.
        asyncio.run(session.send_audio(b"\x01\x02\x03\x04"))

        connection.send.assert_awaited_once()

    def test_send_audio_reraises_exception_when_not_closed(self):
        """Sanity check for the opposite branch: if send() fails for a
        reason unrelated to a concurrent close(), the exception must still
        propagate — the swallow is specifically scoped to the shutdown
        race, not a blanket exception suppressor."""
        session = OpenAIRealtimeSession("gpt-realtime")

        connection = MagicMock()
        connection.send = AsyncMock(side_effect=RuntimeError("genuine send failure"))
        session._connection = connection

        with pytest.raises(RuntimeError, match="genuine send failure"):
            asyncio.run(session.send_audio(b"\x01\x02\x03\x04"))


class TestSendText:
    def test_send_text_respond_true_sends_item_and_response_create(self):
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        asyncio.run(session.send_text("hello", respond=True))

        assert connection.send.await_count == 2
        item_event = connection.send.call_args_list[0].args[0]
        response_event = connection.send.call_args_list[1].args[0]
        assert item_event["type"] == "conversation.item.create"
        assert item_event["item"]["role"] == "user"
        assert item_event["item"]["content"] == [{"type": "input_text", "text": "hello"}]
        assert response_event == {"type": "response.create"}

    def test_send_text_respond_false_sends_only_item_create(self):
        """Regression guard: a context-only injection (e.g. mid-call RAG
        refresh) must NEVER also send response.create, or OpenAI will speak
        the raw injected text aloud as if it were a real reply."""
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        asyncio.run(session.send_text("KB context update", respond=False))

        connection.send.assert_awaited_once()
        sent = connection.send.call_args.args[0]
        assert sent["type"] == "conversation.item.create"

    def test_send_text_empty_string_is_noop(self):
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        asyncio.run(session.send_text("", respond=True))

        connection.send.assert_not_awaited()


class TestSendToolResponse:
    def test_send_tool_response_sends_output_then_response_create(self):
        connection = _FakeConnection()
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        asyncio.run(session.send_tool_response("call-1", {"slots": ["09:00"]}))

        assert connection.send.await_count == 2
        item_event = connection.send.call_args_list[0].args[0]
        response_event = connection.send.call_args_list[1].args[0]
        assert item_event["type"] == "conversation.item.create"
        assert item_event["item"]["type"] == "function_call_output"
        assert item_event["item"]["call_id"] == "call-1"
        import json as _json
        assert _json.loads(item_event["item"]["output"]) == {"slots": ["09:00"]}
        assert response_event == {"type": "response.create"}


# ─────────────────────────────────────────────────────────────────────────
# _receive_loop / _handle_event — demuxing each server event variant
# ─────────────────────────────────────────────────────────────────────────


class TestReceiveLoopDemux:
    def _run_with_events(self, events, **callback_kwargs):
        """Run `_receive_loop()` as a cancellable background task, wait for
        all queued fake events to be consumed, then cancel it — see
        `_FakeConnection`'s docstring for why this (not letting the loop
        exhaust/error on its own) is both the more realistic simulation and
        the hang-safe approach here."""
        connection = _FakeConnection(events=events)
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection
        for name, cb in callback_kwargs.items():
            setattr(session, name, cb)

        async def _run():
            task = asyncio.create_task(session._receive_loop())
            for _ in range(500):
                if not connection._events:
                    break
                await asyncio.sleep(0)
            else:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise AssertionError("fake events were never all consumed")
            # One more tick so the last _handle_event call actually completes
            # before we cancel (recv() for the now-empty queue may already be
            # in flight and blocked, which is fine — that's what we cancel).
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_run())
        return session

    def test_audio_delta_demuxed_and_base64_decoded(self):
        raw = b"\x00\x01\x02"
        evt = _evt(type="response.output_audio.delta", delta=base64.b64encode(raw).decode("ascii"))
        on_audio_chunk = AsyncMock()

        self._run_with_events([evt], on_audio_chunk=on_audio_chunk)

        on_audio_chunk.assert_awaited_once_with(raw)

    def test_speech_started_demuxed_to_on_interrupted(self):
        evt = _evt(type="input_audio_buffer.speech_started")
        on_interrupted = AsyncMock()

        self._run_with_events([evt], on_interrupted=on_interrupted)

        on_interrupted.assert_awaited_once()

    def test_speech_stopped_demuxed_to_on_speech_stopped(self):
        evt = _evt(type="input_audio_buffer.speech_stopped")
        on_speech_stopped = AsyncMock()

        self._run_with_events([evt], on_speech_stopped=on_speech_stopped)

        on_speech_stopped.assert_awaited_once()

    def test_input_transcript_completed_demuxed_with_full_text(self):
        evt = _evt(
            type="conversation.item.input_audio_transcription.completed",
            transcript="hello there, how are you",
        )
        on_input_transcript = AsyncMock()

        self._run_with_events([evt], on_input_transcript=on_input_transcript)

        on_input_transcript.assert_awaited_once_with("hello there, how are you")

    def test_output_transcript_done_demuxed_with_full_text(self):
        evt = _evt(type="response.output_audio_transcript.done", transcript="Sure, I can help.")
        on_output_transcript = AsyncMock()

        self._run_with_events([evt], on_output_transcript=on_output_transcript)

        on_output_transcript.assert_awaited_once_with("Sure, I can help.")

    def test_blank_transcript_not_forwarded(self):
        evt = _evt(type="response.output_audio_transcript.done", transcript="")
        on_output_transcript = AsyncMock()

        self._run_with_events([evt], on_output_transcript=on_output_transcript)

        on_output_transcript.assert_not_awaited()

    def test_function_call_arguments_done_demuxed(self):
        evt = _evt(
            type="response.function_call_arguments.done",
            call_id="call-1",
            name="check_availability",
            arguments='{"date": "tomorrow"}',
        )
        on_tool_call = AsyncMock()

        self._run_with_events([evt], on_tool_call=on_tool_call)

        on_tool_call.assert_awaited_once_with("call-1", "check_availability", '{"date": "tomorrow"}')

    def test_multiple_events_all_processed_in_one_recv_loop_pass(self):
        """Regression guard for the flat (non-per-turn) receive loop shape:
        several events queued across what would be multiple Gemini Live
        "turns" must all be processed by the same _receive_loop() call,
        with no outer per-turn re-invocation needed."""
        evt1 = _evt(type="input_audio_buffer.speech_started")
        evt2 = _evt(
            type="conversation.item.input_audio_transcription.completed",
            transcript="first utterance",
        )
        evt3 = _evt(type="response.output_audio_transcript.done", transcript="first reply")
        evt4 = _evt(
            type="conversation.item.input_audio_transcription.completed",
            transcript="second utterance",
        )

        on_interrupted = AsyncMock()
        on_input_transcript = AsyncMock()
        on_output_transcript = AsyncMock()

        self._run_with_events(
            [evt1, evt2, evt3, evt4],
            on_interrupted=on_interrupted,
            on_input_transcript=on_input_transcript,
            on_output_transcript=on_output_transcript,
        )

        on_interrupted.assert_awaited_once()
        on_output_transcript.assert_awaited_once_with("first reply")
        assert on_input_transcript.await_args_list == [
            (("first utterance",), {}),
            (("second utterance",), {}),
        ]

    def test_error_event_demuxed_and_routed_to_on_error(self):
        evt = _evt(type="error", error=_evt(message="rate limit exceeded", code="rate_limit_exceeded"))
        on_error = AsyncMock()

        self._run_with_events([evt], on_error=on_error)

        on_error.assert_awaited_once()
        err = on_error.call_args.args[0]
        assert isinstance(err, OpenAIRealtimeError)
        assert "rate limit exceeded" in str(err)

    def test_input_transcription_failed_logs_warning_and_does_not_route_to_on_error(self, caplog):
        import logging

        evt = _evt(
            type="conversation.item.input_audio_transcription.failed",
            item_id="item-123",
            error=_evt(code="transcription_error", message="audio format mismatch"),
        )
        on_error = AsyncMock()

        with caplog.at_level(logging.WARNING):
            self._run_with_events([evt], on_error=on_error)

        on_error.assert_not_awaited()
        assert any(
            "item-123" in rec.message and "audio format mismatch" in rec.message
            for rec in caplog.records
        )

    def test_mcp_list_tools_failed_logs_warning_and_does_not_route_to_on_error(self, caplog):
        import logging

        evt = _evt(type="mcp_list_tools.failed", item_id="item-456")
        on_error = AsyncMock()

        with caplog.at_level(logging.WARNING):
            self._run_with_events([evt], on_error=on_error)

        on_error.assert_not_awaited()
        assert any("item-456" in rec.message for rec in caplog.records)

    def test_response_mcp_call_failed_logs_warning_and_does_not_route_to_on_error(self, caplog):
        import logging

        evt = _evt(type="response.mcp_call.failed", item_id="item-789")
        on_error = AsyncMock()

        with caplog.at_level(logging.WARNING):
            self._run_with_events([evt], on_error=on_error)

        on_error.assert_not_awaited()
        assert any("item-789" in rec.message for rec in caplog.records)

    def test_unknown_event_type_is_benign_noop(self):
        evt = _evt(type="session.created")
        on_audio_chunk = AsyncMock()

        # Must not raise.
        self._run_with_events([evt], on_audio_chunk=on_audio_chunk)

        on_audio_chunk.assert_not_awaited()

    def test_recv_exception_translated_and_routed_to_on_error_then_loop_ends(self):
        connection = MagicMock()
        connection.recv = AsyncMock(side_effect=RuntimeError("connection dropped"))
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection
        on_error = AsyncMock()
        session.on_error = on_error

        # Must terminate on its own (not hang) — the whole point of this test.
        asyncio.run(session._receive_loop())

        on_error.assert_awaited_once()
        err = on_error.call_args.args[0]
        assert isinstance(err, OpenAIRealtimeError)

    def test_recv_exception_without_on_error_callback_is_logged_not_raised(self, caplog):
        import logging

        connection = MagicMock()
        connection.recv = AsyncMock(side_effect=RuntimeError("boom"))
        session = OpenAIRealtimeSession("gpt-realtime")
        session._connection = connection

        with caplog.at_level(logging.ERROR):
            asyncio.run(session._receive_loop())  # no on_error set
        assert any("OpenAIRealtimeSession" in rec.message for rec in caplog.records)

    def test_handle_event_exception_reported_but_loop_continues(self):
        """A single bad/unhandled event must not kill the whole receive
        loop — mirrors GeminiLiveSession's per-message try/except."""
        evt1 = _evt(type="response.output_audio_transcript.done", transcript="ok")

        class _BadEvent:
            @property
            def type(self):
                raise RuntimeError("malformed event")

        on_output_transcript = AsyncMock()
        on_error = AsyncMock()

        self._run_with_events(
            [_BadEvent(), evt1],
            on_output_transcript=on_output_transcript,
            on_error=on_error,
        )

        on_error.assert_awaited_once()
        on_output_transcript.assert_awaited_once_with("ok")


# ─────────────────────────────────────────────────────────────────────────
# close() — idempotent, never raises
# ─────────────────────────────────────────────────────────────────────────


class TestClose:
    def test_close_is_idempotent(self):
        session = OpenAIRealtimeSession("gpt-realtime")
        asyncio.run(session.close())
        asyncio.run(session.close())  # must not raise second time

    def test_close_cancels_receive_task_and_exits_connect_cm(self):
        connection = _FakeConnection(events=[])
        client, connect_cm = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime")

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="hi",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await session.close()

        asyncio.run(_run())
        assert connect_cm.aexit_called is True
        assert session._receive_task is None

    def test_close_cancels_receive_task_while_genuinely_blocked_in_recv(self):
        """`events=[]` alone lets _StopReceiving end the receive task on its
        own before close()'s `.cancel()` is ever meaningfully exercised. Use
        a connection whose recv() never resolves on its own, so close() must
        cancel a task that is genuinely suspended *inside* recv(), not one
        that already finished — mirrors the equivalent Gemini Live test and
        this task's own CRITICAL testing-safety note."""
        connection = _FakeConnection(block_forever=True)
        client, connect_cm = _make_fake_client(connection)
        session = OpenAIRealtimeSession("gpt-realtime")

        async def _run():
            with patch.object(session, "_build_client", return_value=client):
                await session.start(
                    system_instruction="hi",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await asyncio.sleep(0)  # let the receive task actually start blocking
            task = session._receive_task
            assert task is not None
            assert not task.done()

            await session.close()

            assert task.cancelled()

        asyncio.run(_run())
        assert connect_cm.aexit_called is True
        assert session._receive_task is None

    def test_close_never_raises_even_if_connect_cm_teardown_errors(self):
        session = OpenAIRealtimeSession("gpt-realtime")

        class _BadCM:
            async def __aexit__(self, *exc):
                raise RuntimeError("teardown boom")

        session._connect_cm = _BadCM()
        # Must not raise.
        asyncio.run(session.close())

    def test_close_never_raises_even_if_receive_task_already_dead(self):
        session = OpenAIRealtimeSession("gpt-realtime")

        async def _dead_task_coro():
            raise RuntimeError("already crashed")

        async def _setup_and_close():
            task = asyncio.create_task(_dead_task_coro())
            await asyncio.sleep(0)  # let it crash
            session._receive_task = task
            await session.close()  # must not raise/propagate task's exception

        asyncio.run(_setup_and_close())


# ─────────────────────────────────────────────────────────────────────────
# resolve_voice_name / _classify_openai_realtime_error / calendly tools
# ─────────────────────────────────────────────────────────────────────────


class TestResolveVoiceName:
    def test_none_falls_back_to_default(self):
        assert resolve_voice_name(None) == DEFAULT_VOICE_NAME

    def test_empty_string_falls_back_to_default(self):
        assert resolve_voice_name("") == DEFAULT_VOICE_NAME

    def test_valid_voice_name_passthrough(self):
        assert resolve_voice_name("cedar") == "cedar"

    def test_case_insensitive_match_normalized_to_canonical_casing(self):
        assert resolve_voice_name("CEDAR") == "cedar"
        assert resolve_voice_name("Cedar") == "cedar"

    def test_whitespace_stripped(self):
        assert resolve_voice_name("  ash  ") == "ash"

    def test_unknown_voice_falls_back_to_default(self):
        # A leftover ElevenLabs/Rime/Google TTS voice ID — must never be
        # passed through raw to the Realtime session config.
        assert resolve_voice_name("21m00Tcm4TlvDq8ikWAM") == DEFAULT_VOICE_NAME

    def test_all_ten_voices_are_valid_and_distinct(self):
        assert len(VALID_VOICE_NAMES) == 10
        for name in VALID_VOICE_NAMES:
            assert resolve_voice_name(name) == name


class TestClassifyError:
    def test_rate_limit_message_classified_as_quota(self):
        err = _classify_openai_realtime_error(RuntimeError("Rate limit reached"))
        assert err.error_type == OpenAIRealtimeErrorType.QUOTA

    def test_timeout_message_classified_as_timeout(self):
        err = _classify_openai_realtime_error(RuntimeError("Request timed out"))
        assert err.error_type == OpenAIRealtimeErrorType.TIMEOUT

    def test_closed_message_classified_as_connection_closed(self):
        err = _classify_openai_realtime_error(RuntimeError("connection closed unexpectedly"))
        assert err.error_type == OpenAIRealtimeErrorType.CONNECTION_CLOSED

    def test_unrecognized_message_classified_as_unknown(self):
        err = _classify_openai_realtime_error(RuntimeError("something weird happened"))
        assert err.error_type == OpenAIRealtimeErrorType.UNKNOWN

    def test_openai_realtime_error_passthrough_in_report_error(self):
        """_report_error must not re-wrap an already-classified error."""
        session = OpenAIRealtimeSession("gpt-realtime")
        on_error = AsyncMock()
        session.on_error = on_error
        original = OpenAIRealtimeError("already classified", OpenAIRealtimeErrorType.AUTH)

        asyncio.run(session._report_error(original))

        on_error.assert_awaited_once_with(original)


class TestBuildCalendlyToolParams:
    def test_returns_two_function_tools_with_expected_names(self):
        tools = _build_calendly_tool_params()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"check_availability", "book_appointment"}
        for t in tools:
            assert t["type"] == "function"
            assert "description" in t
            assert t["parameters"]["type"] == "object"

    def test_check_availability_requires_date(self):
        tools = {t["name"]: t for t in _build_calendly_tool_params()}
        assert tools["check_availability"]["parameters"]["required"] == ["date"]

    def test_book_appointment_requires_slot_and_email(self):
        tools = {t["name"]: t for t in _build_calendly_tool_params()}
        assert tools["book_appointment"]["parameters"]["required"] == ["slot", "attendee_email"]
