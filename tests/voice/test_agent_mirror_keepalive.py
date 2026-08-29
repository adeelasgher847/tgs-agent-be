"""
Regression tests for the LiveKit recording-mirror agent-track keep-alive loop.

Root cause (confirmed via real S3 recording analysis): the agent's LiveKit
recording-mirror `AudioSource` was only fed real frames during bursty
TTS-playback windows and went silent for multi-second gaps between turns.
LiveKit's room-composite egress does not reliably keep such a gappy track
mixed into the recording, unlike the caller's continuously-fed mirror track
— resulting in recordings with the caller's voice but true silence where the
agent should be speaking.

Fix: `TtsStreamMixin._agent_mirror_keepalive_loop` wakes on a steady ~20ms
cadence for the life of the call and fills any gap since the last REAL
mirrored frame with a silent mu-law frame, so the underlying track never
goes quiet for more than one tick — while staying a no-op during active
playback (when `_livekit_recording_mirror()`'s wrapped callback is stamping
real frames every ~20ms already).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.utils.audio_utils import MULAW_FRAME_BYTES
from app.voice.tts_stream_mixin import (
    _AGENT_MIRROR_KEEPALIVE_GAP_THRESHOLD_S,
    TtsStreamMixin,
)

SILENCE_FRAME = bytes([0xFF]) * MULAW_FRAME_BYTES


def _handler_with_publisher():
    h = object.__new__(TtsStreamMixin)
    publisher = MagicMock()
    publisher.connected = True
    publisher.publish_mulaw = AsyncMock()
    h._lk_agent_publisher = publisher
    h._lk_agent_keepalive_task = None
    h._agent_mirror_last_real_frame_ts = 0.0
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
async def test_mirror_callback_stamps_last_real_frame_ts_and_forwards():
    h, publisher = _handler_with_publisher()
    mirror = h._livekit_recording_mirror()
    assert mirror is not None
    assert h._agent_mirror_last_real_frame_ts == 0.0

    await mirror(SILENCE_FRAME)

    publisher.publish_mulaw.assert_awaited_once_with(SILENCE_FRAME)
    assert h._agent_mirror_last_real_frame_ts > 0.0


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
async def test_keepalive_does_not_double_up_right_after_a_real_frame():
    """If a real frame was JUST pushed (within the gap threshold), the
    keep-alive loop must not also push a silence frame in the same window."""
    h, publisher = _handler_with_publisher()

    async def _keep_stamping_real_frames():
        # Continuously mimic real TTS frames arriving every ~15ms (faster
        # than the keep-alive's gap threshold) via the wrapped mirror
        # callback, exactly as the real streaming loop would.
        mirror = h._livekit_recording_mirror()
        for _ in range(6):
            await mirror(bytes([0x10]) * MULAW_FRAME_BYTES)
            await asyncio.sleep(0.015)

    keepalive_task = asyncio.create_task(h._agent_mirror_keepalive_loop())
    real_frames_task = asyncio.create_task(_keep_stamping_real_frames())
    try:
        await real_frames_task
    finally:
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass

    # Every publish_mulaw call must be a REAL frame (0x10) — the keep-alive
    # loop should have found "elapsed < gap threshold" every tick and no-op'd.
    for call in publisher.publish_mulaw.await_args_list:
        assert call.args[0] == bytes([0x10]) * MULAW_FRAME_BYTES
    assert publisher.publish_mulaw.await_count == 6


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
    h, publisher = _handler_with_publisher()

    h._start_agent_mirror_keepalive()
    task = h._lk_agent_keepalive_task
    assert task is not None
    assert not task.done()

    # Calling start again while already running must not create a second task.
    h._start_agent_mirror_keepalive()
    assert h._lk_agent_keepalive_task is task

    await h._stop_agent_mirror_keepalive()
    assert h._lk_agent_keepalive_task is None
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_stop_agent_mirror_keepalive_is_safe_when_never_started():
    h = object.__new__(TtsStreamMixin)
    h._lk_agent_keepalive_task = None
    # Must not raise.
    await h._stop_agent_mirror_keepalive()
    assert h._lk_agent_keepalive_task is None
