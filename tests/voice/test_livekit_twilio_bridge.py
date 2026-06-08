"""Tests for LiveKit ↔ Twilio bridge URL helpers and room parsing."""

from __future__ import annotations

import uuid

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
