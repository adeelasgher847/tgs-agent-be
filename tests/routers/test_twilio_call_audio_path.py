"""
Integration test for Twilio Media Streams audio delivery path.

Verifies:
1. Default Twilio calls do NOT start background audio loop on user pickup.
2. Background audio mixing (mix_tts_frame) is NEVER called by default.
3. 20ms frame pacing (160 bytes per frame) is strictly maintained.
4. Opt-in agents (background_enabled: True) still properly receive mixed background audio.
"""
import base64
import uuid
from unittest.mock import MagicMock
import pytest

from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.utils.audio_utils import MULAW_FRAME_BYTES


class _MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def send_json(self, data):
        if self.closed:
            raise RuntimeError("WebSocket closed")
        self.sent_messages.append(data)

    async def send_text(self, text):
        pass


def _make_mock_agent(provider_slug="elevenlabs", background_enabled=None):
    agent = MagicMock()
    agent.id = "agent-123"
    agent.name = "Test Agent"
    agent.tts_provider_slug = provider_slug
    agent.tts_voice_external_id = "test-voice-id"
    agent.tts_voice_settings_json = None
    agent.tts_settings_json = (
        {"background_enabled": background_enabled}
        if background_enabled is not None
        else {}
    )
    agent.tts_provider_id = None
    agent.tts_voice_id = None
    agent.tts_provider = MagicMock(slug=provider_slug)
    agent.tts_voice = MagicMock(external_voice_id="test-voice-id")
    agent.language = "en"
    agent.voice_type = "female"
    return agent


@pytest.mark.asyncio
async def test_twilio_call_audio_path_clean_by_default():
    """Verify that a standard Twilio call never invokes background mixing and delivers 160-byte frames."""
    ws = _MockWebSocket()
    agent = _make_mock_agent(provider_slug="elevenlabs", background_enabled=None)
    db = MagicMock()
    session_id = str(uuid.uuid4())

    handler = BidirectionalStreamHandler(
        websocket=ws,
        call_session_id=session_id,
        agent_id="agent-123",
        db=db,
    )
    handler.agent = agent
    handler.stream_sid = "MZ1234567890"
    handler._stream_sid_ready.set()
    handler._background_audio = MagicMock()

    # 1. Simulate user pickup
    await handler._handle_user_pickup()
    assert handler._user_picked_up is True
    assert handler._is_background_audio_enabled() is False

    # 2. Mock an async iterator of 5 raw mu-law frames (800 bytes)
    async def _mock_audio_iter():
        for _ in range(5):
            yield bytes([0xAA] * 160)

    # 3. Stream through Twilio handler
    await handler._stream_tts_chunk(
        text="Hello, this is a clean call test.",
        is_final=True,
        prefetched_bytes=_mock_audio_iter(),
    )

    # 4. Verify background audio was NEVER called
    handler._background_audio.mix_tts_frame.assert_not_called()

    # 5. Inspect media frames sent to Twilio
    media_events = [m for m in ws.sent_messages if m.get("event") == "media"]
    assert len(media_events) >= 5, "Expected at least 5 media frames sent to Twilio"

    # Decode media payloads and verify 160-byte alignment
    for evt in media_events:
        payload_b64 = evt["media"]["payload"]
        raw_bytes = base64.b64decode(payload_b64)
        assert len(raw_bytes) == MULAW_FRAME_BYTES, f"Expected 160 bytes, got {len(raw_bytes)}"


@pytest.mark.asyncio
async def test_twilio_call_audio_path_opt_in():
    """Verify that an explicit opt-in agent invokes background mixing."""
    ws = _MockWebSocket()
    agent = _make_mock_agent(provider_slug="elevenlabs", background_enabled=True)
    db = MagicMock()
    session_id = str(uuid.uuid4())

    handler = BidirectionalStreamHandler(
        websocket=ws,
        call_session_id=session_id,
        agent_id="agent-123",
        db=db,
    )
    handler.agent = agent
    handler.stream_sid = "MZ1234567890"
    handler._stream_sid_ready.set()
    handler._background_audio = MagicMock()
    handler._background_audio.mix_tts_frame = MagicMock(side_effect=lambda f: f)

    assert handler._is_background_audio_enabled() is True

    # Stream audio
    async def _mock_audio_iter():
        for _ in range(3):
            yield bytes([0xAA] * 160)

    await handler._stream_tts_chunk(
        text="Hello with background enabled.",
        is_final=True,
        prefetched_bytes=_mock_audio_iter(),
    )

    # Verify background audio mixing was called
    assert handler._background_audio.mix_tts_frame.call_count >= 3
