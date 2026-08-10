"""
Phase 6-3: ElevenLabsWebSocketSession protocol-level unit tests
(app.services.elevenlabs_ws_session).

These exercise the session class in isolation against a fake `websockets`
module (no real network) to prove:
  - init message uses only WS-supported voice_settings keys.
  - single-writer serialization (no concurrent `ws.send()` calls even when
    multiple `send_text()`/`finalize()` calls race).
  - a closed session rejects further writes.
  - `iter_audio()` decodes audio chunks and stops at `isFinal`.
  - `aclose()` is idempotent and safely callable more than once.

Pipeline-level integration (routing, lazy creation, turn lifecycle,
cancellation/turn_id discard, fallback) is covered in
tests/voice/test_elevenlabs_ws_streaming_session.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.elevenlabs_ws_session import (
    ElevenLabsWebSocketSession,
    ElevenLabsWebSocketSessionError,
)


class _FakeWsConnection:
    def __init__(self, fail_on_send_number: int | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._in_flight = 0
        self.max_concurrent_sends = 0
        self.closed = False
        self.close_code: int | None = None
        self._incoming: "asyncio.Queue[str | None]" = asyncio.Queue()
        # 1-indexed: if set, the Nth call to send() raises instead of
        # succeeding, simulating a transient network blip / ConnectionClosed
        # mid-session.
        self._fail_on_send_number = fail_on_send_number
        self._send_count = 0

    async def send(self, payload: str) -> None:
        self._send_count += 1
        if (
            self._fail_on_send_number is not None
            and self._send_count == self._fail_on_send_number
        ):
            raise ConnectionError("simulated transient network failure")
        self._in_flight += 1
        self.max_concurrent_sends = max(self.max_concurrent_sends, self._in_flight)
        # Yield control so a genuinely-concurrent second send() (a bug)
        # would overlap and get caught by max_concurrent_sends > 1.
        await asyncio.sleep(0.01)
        self.sent.append(json.loads(payload))
        self._in_flight -= 1

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        self.close_code = code
        self._incoming.put_nowait(None)

    def push_server_message(self, msg: dict[str, Any]) -> None:
        self._incoming.put_nowait(json.dumps(msg))

    def end_stream(self) -> None:
        self._incoming.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item


def _install_fake_websockets(monkeypatch, conn: _FakeWsConnection):
    async def _connect(url, additional_headers=None):
        conn.connect_url = url
        conn.connect_headers = additional_headers
        return conn

    fake_module = SimpleNamespace(connect=_connect)
    monkeypatch.setitem(sys.modules, "websockets", fake_module)


@pytest.fixture(autouse=True)
def _elevenlabs_api_key(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ELEVENLABS_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Init message / voice_settings filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_message_only_forwards_ws_supported_voice_settings(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(
        voice_external_id="voice-1",
        voice_settings={
            "stability": 0.6,
            "similarity_boost": 0.8,
            "previous_text": "should never be sent over WS",
            "next_request_ids": ["a", "b"],
        },
    )
    await session.start()

    assert len(conn.sent) == 1
    init_msg = conn.sent[0]
    assert init_msg["voice_settings"] == {"stability": 0.6, "similarity_boost": 0.8}
    assert "previous_text" not in init_msg["voice_settings"]
    assert "next_request_ids" not in init_msg["voice_settings"]

    conn.end_stream()
    await session.aclose()


@pytest.mark.asyncio
async def test_connect_url_uses_auto_mode_and_inactivity_timeout(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(
        voice_external_id="voice-xyz", model_id="eleven_flash_v2_5"
    )
    await session.start()

    assert "auto_mode=true" in conn.connect_url
    assert "inactivity_timeout=20" in conn.connect_url
    assert "voice-xyz" in conn.connect_url
    assert conn.connect_headers == {"xi-api-key": "test-key"}

    conn.end_stream()
    await session.aclose()


# ---------------------------------------------------------------------------
# Single-writer serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_send_text_calls_are_serialized(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()

    # Two chunks racing to write "concurrently" — send_text() only enqueues
    # and returns instantly, so gather() here proves the CALLERS don't
    # block each other, while the writer task's max_concurrent_sends proves
    # the actual socket writes never overlap.
    await asyncio.gather(
        session.send_text("chunk A text"),
        session.send_text("chunk B text"),
    )

    # Wait for the writer task to actually drain both queued messages
    # (init + 2 text sends = 3 total).
    for _ in range(50):
        if len(conn.sent) >= 3:
            break
        await asyncio.sleep(0.01)

    assert conn.max_concurrent_sends == 1  # never more than one in-flight ws.send()
    assert len(conn.sent) == 3

    conn.end_stream()
    await session.aclose()


# ---------------------------------------------------------------------------
# Closed session rejects further writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_session_rejects_send_text(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()
    conn.end_stream()
    await session.aclose()

    with pytest.raises(ElevenLabsWebSocketSessionError):
        await session.send_text("too late")


@pytest.mark.asyncio
async def test_closed_session_rejects_start_reuse(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()
    conn.end_stream()
    await session.aclose()

    with pytest.raises(ElevenLabsWebSocketSessionError):
        await session.start(initial_text="second turn attempt")


# ---------------------------------------------------------------------------
# iter_audio(): decodes audio, stops at isFinal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iter_audio_yields_decoded_chunks_and_stops_at_final(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()

    raw_audio_1 = b"\x01\x02\x03"
    raw_audio_2 = b"\x04\x05"
    conn.push_server_message({"audio": base64.b64encode(raw_audio_1).decode()})
    conn.push_server_message({"audio": base64.b64encode(raw_audio_2).decode()})
    conn.push_server_message({"audio": None, "isFinal": True})

    received = []
    async for chunk in session.iter_audio():
        received.append(chunk)

    assert received == [raw_audio_1, raw_audio_2]

    await session.aclose()


# ---------------------------------------------------------------------------
# aclose() idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_is_idempotent(monkeypatch):
    conn = _FakeWsConnection()
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()
    conn.end_stream()

    await session.aclose()
    await session.aclose()  # must not raise / hang

    assert session.is_closed is True
    assert conn.closed is True


# ---------------------------------------------------------------------------
# Session open failure -> ElevenLabsWebSocketSessionError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_raises_typed_error_on_connect_failure(monkeypatch):
    async def _failing_connect(url, additional_headers=None):
        raise OSError("connection refused")

    fake_module = SimpleNamespace(connect=_failing_connect)
    monkeypatch.setitem(sys.modules, "websockets", fake_module)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    with pytest.raises(ElevenLabsWebSocketSessionError):
        await session.start()
    assert session.is_closed is True


# ---------------------------------------------------------------------------
# Mid-session write failure (code-review follow-up, blocking bug fix)
#
# Before the fix: a failed `ws.send()` inside `_writer_loop` only logged a
# WARNING and returned — the session stayed "started"/not-closed forever,
# subsequent send_text()/finalize() calls kept silently enqueueing into a
# queue nobody was draining, and `iter_audio()` was left awaiting
# `_audio_queue.get()` with no code-level unblock — up to ~20s of dead air
# on a live call, bounded only by ElevenLabs' own server-side
# inactivity_timeout (and only then if the receive side also happened to
# fail, which isn't guaranteed for a write-only failure).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_failure_unblocks_iter_audio_promptly(monkeypatch):
    """The init message is send #1 — fail send #2 (the first real
    send_text()) and prove iter_audio() ends immediately rather than
    hanging."""
    # fail_on_send_number=2: init message succeeds (send #1), the next
    # write (send_text below) fails.
    conn = _FakeWsConnection(fail_on_send_number=2)
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()

    await session.send_text("this write will fail")

    # Must unblock promptly (bounded by the test's own timeout, NOT by
    # ElevenLabs' 20s server-side inactivity_timeout backstop) rather than
    # hang indefinitely.
    received = []
    await asyncio.wait_for(
        _drain(session.iter_audio(), received),
        timeout=2.0,
    )
    assert received == []  # no audio arrived before the write failed


async def _drain(aiter, into: list) -> None:
    async for chunk in aiter:
        into.append(chunk)


@pytest.mark.asyncio
async def test_write_failure_marks_session_failed_and_closed(monkeypatch):
    conn = _FakeWsConnection(fail_on_send_number=2)
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()
    await session.send_text("triggers the failure")

    # Drain iter_audio() so the failure has definitely propagated.
    async for _ in session.iter_audio():
        pass

    # Give the fire-and-forget aclose() task (scheduled from inside
    # _writer_loop's exception handler) a beat to actually run.
    for _ in range(50):
        if session.is_closed:
            break
        await asyncio.sleep(0.01)

    assert session.write_failed is True
    assert session.is_closed is True
    assert conn.closed is True  # aclose() was actually invoked, not just flagged


@pytest.mark.asyncio
async def test_write_failure_causes_subsequent_send_text_to_raise(monkeypatch):
    conn = _FakeWsConnection(fail_on_send_number=2)
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()
    await session.send_text("triggers the failure")

    async for _ in session.iter_audio():
        pass

    # A relay chunk queued after the failure must be told immediately that
    # its text was NOT delivered — never silently "succeed" into a queue
    # nobody drains.
    with pytest.raises(ElevenLabsWebSocketSessionError):
        await session.send_text("queued after the failure")


@pytest.mark.asyncio
async def test_write_failure_causes_subsequent_finalize_to_raise(monkeypatch):
    conn = _FakeWsConnection(fail_on_send_number=2)
    _install_fake_websockets(monkeypatch, conn)

    session = ElevenLabsWebSocketSession(voice_external_id="voice-1")
    await session.start()
    await session.send_text("triggers the failure")

    async for _ in session.iter_audio():
        pass

    # The turn-final flush must NOT silently no-op after a write failure —
    # that would strand the owner's iter_audio() consumer waiting for an
    # `isFinal` that can never arrive from a dead write path.
    with pytest.raises(ElevenLabsWebSocketSessionError):
        await session.finalize()
