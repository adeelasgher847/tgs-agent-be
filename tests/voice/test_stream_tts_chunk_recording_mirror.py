"""
Regression coverage: TtsStreamMixin._stream_tts_chunk's incremental streaming
branch (used for any prefetched async-iterator audio, e.g. a provider's
async_stream_synthesize output -- the dominant path for any response of
VOICE_TTS_STREAM_MIN_WORDS+ words) must mirror every frame it sends to Twilio
into the LiveKit recording publisher, exactly like the bulk-buffer path
(generate_mulaw_tts + stream_mulaw_bytes_over_twilio) already does.

Before this fix, this branch's `send_frame` sent audio straight to Twilio via
websocket.send_json with no mirror_mulaw call at all -- confirmed via a real
S3 recording where several agent turns were completely silent
(ffprobe/volumedetect near noise floor, -40 to -70dB) at exactly the
timestamps of longer/streamed responses, while short responses (which
happened to take the bulk path) had normal speech levels.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedTtsRuntime
from app.utils.audio_utils import MULAW_FRAME_BYTES


def _fake_tts_runtime(adapter_slug: str = "rime", voice_external_id: str | None = "voice-1"):
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )


async def _audio_iter(num_frames: int = 3):
    for _ in range(num_frames):
        yield bytes([0x10]) * MULAW_FRAME_BYTES


def _twilio_handler_with_recording():
    from app.voice.tts_stream_mixin import TtsStreamMixin

    h = object.__new__(TtsStreamMixin)
    h._tts_cancel = asyncio.Event()
    h._elevenlabs_prev_tts_text = ""
    h.agent = MagicMock()
    h.agent.language = "en"
    h.agent.voice_type = "female"
    h.db = None
    h.stream_sid = "MZtest"
    h._tts_lock = asyncio.Lock()
    h.is_speaking = False
    h._is_tts_playing = False
    h._twilio_buffer_primed = True
    h._is_background_audio_enabled = lambda: False
    h._resolve_voice_volume = lambda: 1.0
    h.websocket = MagicMock()
    h.websocket.send_json = AsyncMock()

    # LiveKit recording mirror publisher -- connected and ready.
    h._lk_agent_publisher = MagicMock()
    h._lk_agent_publisher.connected = True
    h._lk_agent_publisher.publish_mulaw = AsyncMock()

    return h


@pytest.mark.asyncio
async def test_streaming_branch_mirrors_every_frame_into_recording():
    h = _twilio_handler_with_recording()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=False,
            prefetched_bytes=_audio_iter(3),
        )

    # 3 real audio frames sent to Twilio...
    assert h.websocket.send_json.await_count == 3
    # ...and each one must also have been mirrored into the recording publisher.
    assert h._lk_agent_publisher.publish_mulaw.await_count == 3
    mirrored_frames = [c.args[0] for c in h._lk_agent_publisher.publish_mulaw.await_args_list]
    assert all(f == bytes([0x10]) * MULAW_FRAME_BYTES for f in mirrored_frames)


@pytest.mark.asyncio
async def test_streaming_branch_is_noop_mirror_when_publisher_absent():
    """No recording publisher configured (e.g. LiveKit disabled, or not a
    Twilio call under recording) -- streaming playback must proceed exactly
    as before, with zero mirror calls and no errors."""
    from app.voice.tts_stream_mixin import TtsStreamMixin

    h = object.__new__(TtsStreamMixin)
    h._tts_cancel = asyncio.Event()
    h._elevenlabs_prev_tts_text = ""
    h.agent = MagicMock()
    h.agent.language = "en"
    h.agent.voice_type = "female"
    h.db = None
    h.stream_sid = "MZtest"
    h._tts_lock = asyncio.Lock()
    h.is_speaking = False
    h._is_tts_playing = False
    h._twilio_buffer_primed = True
    h._is_background_audio_enabled = lambda: False
    h._resolve_voice_volume = lambda: 1.0
    h.websocket = MagicMock()
    h.websocket.send_json = AsyncMock()
    # No _lk_agent_publisher attribute at all.

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
        )

    assert h.websocket.send_json.await_count == 2


@pytest.mark.asyncio
async def test_streaming_branch_mirror_failure_does_not_break_playback():
    """A raising mirror callback (e.g. LiveKit publish hiccup) must never
    disrupt live Twilio playback -- best-effort only."""
    h = _twilio_handler_with_recording()
    h._lk_agent_publisher.publish_mulaw = AsyncMock(side_effect=RuntimeError("boom"))

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
        )

    assert h.websocket.send_json.await_count == 2
