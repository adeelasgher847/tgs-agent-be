"""
Regression tests for the LiveKit recording-mirror agent-track keep-alive loop.

Root cause #1 (confirmed via real S3 recording analysis): the agent's
LiveKit recording-mirror `AudioSource` was only fed real frames during
bursty TTS-playback windows and went silent for multi-second gaps between
turns. LiveKit's room-composite egress does not reliably keep such a gappy
track mixed into the recording, unlike the caller's continuously-fed mirror
track — resulting in recordings with the caller's voice but true silence
where the agent should be speaking.

Fix #1: `TtsStreamMixin._agent_mirror_keepalive_loop` wakes on a steady
~20ms cadence for the life of the call and fills any gap since the last
REAL mirrored frame with a silent mu-law frame, so the underlying track
never goes quiet for more than one tick.

Root cause #2 (confirmed via frame-level audio analysis of a real Twilio
recording): the mirror callback used to call `publisher.publish_mulaw()`
directly, INLINE, awaited by the caller-facing Twilio send path (both
`stream_mulaw_bytes_over_twilio` and `_stream_tts_chunk`'s `send_frame`).
`publish_mulaw` wraps LiveKit's rate-limited `AudioSource.capture_frame`,
which can legitimately take longer than one 20ms tick — coupling the live
call's audio pacing to the recording mirror's independent pacing clock.
Measured ~5.5% of frames showing a brief volume dip during continuous
agent speech, consistent with this coupling occasionally delaying a live
frame's send (audible as slight choppiness/distortion).

Fix #2: `_livekit_recording_mirror()`'s callback is now a fast, in-memory,
non-blocking ENQUEUE onto `self._agent_mirror_queue` — it never calls the
publisher or awaits anything network/pacing-related. `_agent_mirror_
keepalive_loop` becomes the SOLE writer to the publisher: each tick it
drains one queued real frame (FIFO, oldest first) if available, otherwise
applies the same silence keep-alive logic as before. This means the
Twilio-facing call sites' `await mirror_mulaw(frame)` now awaits a
synchronous queue operation only — it can never block, delay, or affect
caller-facing audio — and, as a bonus, there is now exactly one coroutine
that ever calls `publish_mulaw()`, removing any possibility of the
keep-alive loop's own silence frame racing a concurrently-pushed real one.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.utils.audio_utils import MULAW_FRAME_BYTES
from app.voice.tts_stream_mixin import (
    _AGENT_MIRROR_KEEPALIVE_GAP_THRESHOLD_S,
    _AGENT_MIRROR_QUEUE_MAXSIZE,
    TtsStreamMixin,
)

SILENCE_FRAME = bytes([0xFF]) * MULAW_FRAME_BYTES
REAL_FRAME = bytes([0x10]) * MULAW_FRAME_BYTES


def _handler_with_publisher():
    h = object.__new__(TtsStreamMixin)
    publisher = MagicMock()
    publisher.connected = True
    publisher.publish_mulaw = AsyncMock()
    h._lk_agent_publisher = publisher
    h._lk_agent_keepalive_task = None
    h._agent_mirror_last_real_frame_ts = 0.0
    h._agent_mirror_queue = asyncio.Queue(maxsize=_AGENT_MIRROR_QUEUE_MAXSIZE)
    return h, publisher


def test_livekit_recording_mirror_returns_none_when_publisher_missing():
    h = object.__new__(TtsStreamMixin)
    h._lk_agent_publisher = None
    assert h._livekit_recording_mirror() is None


def test_livekit_recording_mirror_returns_none_when_not_connected():
    h = object.__new__(TtsStreamMixin)
    publisher = MagicMock()
    publisher.connected = False
    h._lk_agent_publisher = publisher
    assert h._livekit_recording_mirror() is None


@pytest.mark.asyncio
async def test_mirror_callback_only_enqueues_never_touches_publisher():
    """The Twilio-facing contract: awaiting the mirror callback must be a
    fast, non-blocking, in-memory operation that never calls the publisher
    directly (that's the keep-alive loop's job) and therefore can never
    block/delay the caller-facing send that awaits it."""
    h, publisher = _handler_with_publisher()
    mirror = h._livekit_recording_mirror()
    assert mirror is not None

    await mirror(REAL_FRAME)

    publisher.publish_mulaw.assert_not_awaited()
    assert h._agent_mirror_queue.qsize() == 1
    assert h._agent_mirror_queue.get_nowait() == REAL_FRAME


@pytest.mark.asyncio
async def test_mirror_enqueue_drops_oldest_when_queue_full():
    h, _ = _handler_with_publisher()
    h._agent_mirror_queue = asyncio.Queue(maxsize=2)
    mirror = h._livekit_recording_mirror()

    await mirror(b"\x01" * MULAW_FRAME_BYTES)
    await mirror(b"\x02" * MULAW_FRAME_BYTES)
    await mirror(b"\x03" * MULAW_FRAME_BYTES)  # queue full -> drops oldest (0x01)

    assert h._agent_mirror_queue.qsize() == 2
    remaining = [h._agent_mirror_queue.get_nowait(), h._agent_mirror_queue.get_nowait()]
    assert remaining == [b"\x02" * MULAW_FRAME_BYTES, b"\x03" * MULAW_FRAME_BYTES]


@pytest.mark.asyncio
async def test_mirror_enqueue_is_noop_when_queue_not_initialized():
    """Defensive: if the keep-alive task (and its queue) hasn't been started
    yet, the mirror callback must silently no-op, not raise."""
    h = object.__new__(TtsStreamMixin)
    publisher = MagicMock()
    publisher.connected = True
    publisher.publish_mulaw = AsyncMock()
    h._lk_agent_publisher = publisher
    # No h._agent_mirror_queue attribute at all.
    mirror = h._livekit_recording_mirror()
    await mirror(REAL_FRAME)  # must not raise
    publisher.publish_mulaw.assert_not_awaited()


@pytest.mark.asyncio
async def test_keepalive_sends_silence_frame_after_gap():
    """After the configured gap threshold elapses with no real frame, the
    keep-alive loop must push exactly one silence frame per tick."""
    h, publisher = _handler_with_publisher()
    # Simulate a real frame long enough ago to be outside the gap threshold
    # (production code stamps this with time.monotonic()).
    h._agent_mirror_last_real_frame_ts = time.monotonic() - (
        _AGENT_MIRROR_KEEPALIVE_GAP_THRESHOLD_S * 10
    )

    task = asyncio.create_task(h._agent_mirror_keepalive_loop())
    try:
        # Give the loop a couple of ticks to run (interval is 20ms).
        await asyncio.sleep(0.09)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert publisher.publish_mulaw.await_count >= 1
    for call in publisher.publish_mulaw.await_args_list:
        assert call.args[0] == SILENCE_FRAME


@pytest.mark.asyncio
async def test_keepalive_drains_queued_real_frames_in_order_never_silence():
    """If real frames are queued (via the mirror callback, exactly as the
    real streaming loop would use it), the keep-alive loop's single writer
    must drain and publish them in FIFO order, and must never interleave a
    silence frame while real frames are still pending."""
    h, publisher = _handler_with_publisher()

    async def _keep_enqueueing_real_frames():
        # Mimic real TTS frames arriving every ~15ms (faster than the
        # keep-alive's 20ms drain cadence) via the wrapped mirror callback,
        # exactly as the real streaming loop would.
        mirror = h._livekit_recording_mirror()
        for i in range(6):
            await mirror(bytes([i]) * MULAW_FRAME_BYTES)
            await asyncio.sleep(0.015)

    keepalive_task = asyncio.create_task(h._agent_mirror_keepalive_loop())
    try:
        await _keep_enqueueing_real_frames()
        # Production (6 frames @ ~15ms) outpaces consumption (1 frame per
        # 20ms tick), so the queue may not be fully drained the instant
        # production finishes -- wait for it to empty before asserting.
        for _ in range(50):
            if h._agent_mirror_queue.empty():
                break
            await asyncio.sleep(0.02)
        assert h._agent_mirror_queue.empty()
    finally:
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass

    # Every publish_mulaw call must be a REAL frame, in the exact order
    # enqueued — never the silence frame, since real frames were always
    # available to drain first.
    published = [call.args[0] for call in publisher.publish_mulaw.await_args_list]
    assert published == [bytes([i]) * MULAW_FRAME_BYTES for i in range(6)]


@pytest.mark.asyncio
async def test_keepalive_is_noop_when_publisher_not_connected():
    h, publisher = _handler_with_publisher()
    publisher.connected = False
    h._agent_mirror_last_real_frame_ts = time.monotonic() - 10.0

    task = asyncio.create_task(h._agent_mirror_keepalive_loop())
    try:
        await asyncio.sleep(0.07)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    publisher.publish_mulaw.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_stop_agent_mirror_keepalive_lifecycle():
    h, _publisher = _handler_with_publisher()

    h._start_agent_mirror_keepalive()
    task = h._lk_agent_keepalive_task
    assert task is not None
    assert not task.done()
    assert h._agent_mirror_queue is not None

    # Calling start again while already running must not create a second task.
    h._start_agent_mirror_keepalive()
    assert h._lk_agent_keepalive_task is task

    await h._stop_agent_mirror_keepalive()
    assert h._lk_agent_keepalive_task is None
    assert h._agent_mirror_queue is None
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_stop_agent_mirror_keepalive_is_safe_when_never_started():
    h = object.__new__(TtsStreamMixin)
    h._lk_agent_keepalive_task = None
    # Must not raise.
    await h._stop_agent_mirror_keepalive()
    assert h._lk_agent_keepalive_task is None
    assert h._agent_mirror_queue is None


@pytest.mark.asyncio
async def test_mirror_enqueue_never_blocks_even_under_heavy_load(monkeypatch):
    """Direct guard for the fix's core promise: enqueueing must complete
    near-instantly even when the publisher itself is artificially slow --
    proving the Twilio-facing send path awaiting this callback can never be
    held up by LiveKit publish latency."""
    h, publisher = _handler_with_publisher()

    async def _slow_publish(_frame):
        await asyncio.sleep(5.0)  # would time out any reasonable test if awaited inline

    publisher.publish_mulaw = AsyncMock(side_effect=_slow_publish)
    mirror = h._livekit_recording_mirror()

    start = time.monotonic()
    await asyncio.wait_for(mirror(REAL_FRAME), timeout=0.5)
    elapsed = time.monotonic() - start

    assert elapsed < 0.1
    publisher.publish_mulaw.assert_not_awaited()
