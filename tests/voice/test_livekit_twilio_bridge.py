"""Tests for LiveKit ↔ Twilio bridge URL helpers and room parsing."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_build_livekit_stream_ws_url():
    from unittest.mock import MagicMock, patch

    from app.voice.livekit_twilio_bridge import build_livekit_stream_ws_url

    room = f"room_{uuid.uuid4()}"
    mock_settings = MagicMock()
    mock_settings.WEBHOOK_BASE_URL = "https://api.example.com"

    with patch("app.voice.livekit_twilio_bridge.settings", mock_settings):
        url = build_livekit_stream_ws_url(room)

    assert url == f"wss://api.example.com/api/v1/livekit/{room}"


def test_call_session_id_from_valid_room_name():
    from app.routers.livekit_bridge import _call_session_id_from_room

    sid = uuid.uuid4()
    room = f"room_{sid}"
    assert _call_session_id_from_room(room) == sid


def test_call_session_id_from_invalid_room_name():
    from app.routers.livekit_bridge import _call_session_id_from_room

    assert _call_session_id_from_room("not-a-room") is None
    assert _call_session_id_from_room("room_not-a-uuid") is None


@pytest.mark.asyncio
async def test_publish_mulaw_writes_pcm_into_int16_frame_without_raising():
    """Regression test for the same memoryview-format bug fixed in
    _LiveKitAgentAudioPublisher.publish_mulaw (livekit_browser_call_handler.py):
    rtc.AudioFrame.data is a memoryview already cast to int16 ("h") format,
    so assigning a plain `bytes` object (format "B") into it raises
    "ValueError: memoryview assignment: lvalue and rvalue have different
    structures" unless the destination view is first cast to "B". Uses the
    real installed livekit.rtc.AudioFrame (not a mock)."""
    from app.utils.audio_utils import MULAW_FRAME_BYTES
    from app.voice.livekit_twilio_bridge import LiveKitTwilioPublisher

    publisher = LiveKitTwilioPublisher.__new__(LiveKitTwilioPublisher)
    publisher._connected = True
    publisher._source = MagicMock()
    publisher._source.capture_frame = AsyncMock()

    mulaw_bytes = bytes([0xFF]) * MULAW_FRAME_BYTES

    await publisher.publish_mulaw(mulaw_bytes)

    publisher._source.capture_frame.assert_awaited_once()
    frame_arg = publisher._source.capture_frame.call_args[0][0]
    assert bytes(frame_arg.data.cast("B")) != bytes(len(frame_arg.data.cast("B")))
