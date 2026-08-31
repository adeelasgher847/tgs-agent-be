"""Regression coverage for the Deepgram SpeechStarted "soft duck" feature.

Deepgram's `vad_events=true`/SpeechStarted event on the Nova-3 STT path
(app/services/deepgram_stt_service.py) is a pure VAD-onset signal -- no
transcript/confidence -- wired up as an early "soft duck" that briefly lowers
TTS output volume instead of cancelling/interrupting the turn. The existing
hard-cancel barge-in logic (classify_turn() on interim transcripts) is
untouched; this suite only covers the new soft-duck path:

  1. VoiceOrchestrator._on_speech_started() duck-types a call to
     handler._on_stt_speech_started() when defined, no-ops otherwise, and
     never propagates a callback error.
  2. BidirectionalStreamHandler._on_stt_speech_started() (Twilio) sets a
     _soft_duck_until_mono deadline only while TTS is playing.
  3. TtsStreamMixin._apply_soft_duck() / _resolve_voice_volume() scale gain
     down to VOICE_SOFT_DUCK_GAIN while the duck window is active and revert
     once VOICE_SOFT_DUCK_MS has elapsed.
  4. LiveKitBrowserCallHandler._on_stt_speech_started() is observability-only
     -- no gain pipeline exists on that transport.
  5. Config defaults for VOICE_SOFT_DUCK_ENABLED / _MS / _GAIN.
  6. Regression: the duck must apply WITHIN an in-flight TTS chunk (real
     _stream_tts_chunk() streaming loop, and the shared
     stream_mulaw_bytes_over_twilio() frame-send primitive), not just on the
     next chunk after the one that was already playing when SpeechStarted
     fired.
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedTtsRuntime
from app.core.config import settings
from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.utils.audio_utils import MULAW_FRAME_BYTES, stream_mulaw_bytes_over_twilio
from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler
from app.voice.tts_stream_mixin import TtsStreamMixin
from app.voice.voice_orchestrator import VoiceOrchestrator


class DummyWebSocket:
    async def send_text(self, data: str) -> None:
        pass

    async def send_json(self, data: dict) -> None:
        pass

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass


def _raw_twilio_handler() -> BidirectionalStreamHandler:
    return BidirectionalStreamHandler(
        websocket=DummyWebSocket(),
        call_session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        db=None,
    )


def _raw_livekit_handler() -> LiveKitBrowserCallHandler:
    mock_session = MagicMock()
    mock_session.id = uuid.uuid4()
    mock_agent = MagicMock()
    mock_agent.id = uuid.uuid4()
    return LiveKitBrowserCallHandler(
        db=None,
        call_session=mock_session,
        agent=mock_agent,
    )


# ── 1. VoiceOrchestrator._on_speech_started duck-typing ────────────────────


class _HandlerWithHook:
    def __init__(self) -> None:
        self._on_stt_speech_started = AsyncMock()


class _HandlerWithoutHook:
    """Deliberately has no _on_stt_speech_started attribute at all."""


class _HandlerWithFailingHook:
    async def _on_stt_speech_started(self) -> None:
        raise RuntimeError("boom")


async def test_orchestrator_on_speech_started_calls_handler_hook_when_defined():
    orchestrator = VoiceOrchestrator.__new__(VoiceOrchestrator)
    handler = _HandlerWithHook()
    orchestrator._h = handler

    await orchestrator._on_speech_started()

    handler._on_stt_speech_started.assert_awaited_once_with()


async def test_orchestrator_on_speech_started_noops_when_hook_undefined():
    orchestrator = VoiceOrchestrator.__new__(VoiceOrchestrator)
    orchestrator._h = _HandlerWithoutHook()

    # Must not raise even though the handler has no _on_stt_speech_started.
    await orchestrator._on_speech_started()


async def test_orchestrator_on_speech_started_swallows_hook_exception():
    orchestrator = VoiceOrchestrator.__new__(VoiceOrchestrator)
    orchestrator._h = _HandlerWithFailingHook()

    # Must not raise -- errors in the optional hook are caught and logged.
    await orchestrator._on_speech_started()


# ── 2. BidirectionalStreamHandler._on_stt_speech_started (Twilio) ──────────


async def test_twilio_handler_sets_soft_duck_deadline_only_when_tts_playing():
    handler = _raw_twilio_handler()
    handler._is_tts_playing = True
    assert handler._soft_duck_until_mono == 0.0

    before = time.monotonic()
    await handler._on_stt_speech_started()
    after = time.monotonic()

    # Deadline must be roughly now + _soft_duck_ms.
    expected_min = before + (handler._soft_duck_ms / 1000.0)
    expected_max = after + (handler._soft_duck_ms / 1000.0)
    assert expected_min <= handler._soft_duck_until_mono <= expected_max


async def test_twilio_handler_noop_when_tts_not_playing():
    handler = _raw_twilio_handler()
    handler._is_tts_playing = False

    await handler._on_stt_speech_started()

    assert handler._soft_duck_until_mono == 0.0


async def test_twilio_handler_noop_when_soft_duck_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "VOICE_SOFT_DUCK_ENABLED", False)
    handler = _raw_twilio_handler()
    handler._is_tts_playing = True

    await handler._on_stt_speech_started()

    assert handler._soft_duck_until_mono == 0.0


async def test_twilio_handler_soft_duck_defaults_resolved_from_settings():
    handler = _raw_twilio_handler()

    assert handler._soft_duck_enabled is True
    assert handler._soft_duck_ms == pytest.approx(400.0)
    assert handler._soft_duck_gain == pytest.approx(0.35)


async def test_twilio_handler_soft_duck_reads_live_settings_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "VOICE_SOFT_DUCK_MS", 250)
    monkeypatch.setattr(settings, "VOICE_SOFT_DUCK_GAIN", 0.5)

    handler = _raw_twilio_handler()

    assert handler._soft_duck_ms == pytest.approx(250.0)
    assert handler._soft_duck_gain == pytest.approx(0.5)


# ── 3. TtsStreamMixin gain scaling ──────────────────────────────────────────


def _make_mixin() -> TtsStreamMixin:
    mixin = TtsStreamMixin.__new__(TtsStreamMixin)
    mixin.agent = None  # short-circuits _resolve_voice_volume to _apply_soft_duck(1.0)
    mixin._soft_duck_until_mono = 0.0
    mixin._soft_duck_gain = 0.35
    return mixin


def test_apply_soft_duck_scales_gain_while_window_active(monkeypatch: pytest.MonkeyPatch):
    mixin = _make_mixin()
    fake_now = 1000.0
    monkeypatch.setattr("app.voice.tts_stream_mixin.time.monotonic", lambda: fake_now)
    mixin._soft_duck_until_mono = fake_now + 0.4  # window still open

    assert mixin._apply_soft_duck(1.0) == pytest.approx(0.35)


def test_apply_soft_duck_passthrough_after_window_elapses(monkeypatch: pytest.MonkeyPatch):
    mixin = _make_mixin()
    fake_now = 1000.0
    monkeypatch.setattr("app.voice.tts_stream_mixin.time.monotonic", lambda: fake_now)
    mixin._soft_duck_until_mono = fake_now - 0.001  # window just closed

    assert mixin._apply_soft_duck(1.0) == pytest.approx(1.0)


def test_apply_soft_duck_passthrough_when_never_set():
    mixin = _make_mixin()
    assert mixin._soft_duck_until_mono == 0.0

    assert mixin._apply_soft_duck(1.0) == pytest.approx(1.0)


def test_resolve_voice_volume_ducked_during_window_and_reverts_after(
    monkeypatch: pytest.MonkeyPatch,
):
    mixin = _make_mixin()
    current_time = {"value": 1000.0}
    monkeypatch.setattr(
        "app.voice.tts_stream_mixin.time.monotonic", lambda: current_time["value"]
    )

    mixin._soft_duck_until_mono = current_time["value"] + (400 / 1000.0)

    # Still inside the 400ms duck window -> reduced gain.
    current_time["value"] += 0.1
    assert mixin._resolve_voice_volume() == pytest.approx(0.35)

    # Window elapsed -> gain reverts to normal (1.0, no agent configured).
    current_time["value"] += 0.4
    assert mixin._resolve_voice_volume() == pytest.approx(1.0)


# ── 4. LiveKitBrowserCallHandler._on_stt_speech_started — observability only ─


async def test_livekit_handler_increments_counter_only_when_tts_playing():
    handler = _raw_livekit_handler()
    handler._is_tts_playing = True
    assert handler._soft_duck_signal_count == 0

    await handler._on_stt_speech_started()

    assert handler._soft_duck_signal_count == 1
    # Confirms no gain-pipeline attribute is created/used on this transport.
    assert not hasattr(handler, "_soft_duck_until_mono")
    assert not hasattr(handler, "_soft_duck_gain")


async def test_livekit_handler_noop_when_tts_not_playing():
    handler = _raw_livekit_handler()
    handler._is_tts_playing = False

    await handler._on_stt_speech_started()

    assert handler._soft_duck_signal_count == 0


async def test_livekit_handler_increments_across_multiple_calls():
    handler = _raw_livekit_handler()
    handler._is_tts_playing = True

    await handler._on_stt_speech_started()
    await handler._on_stt_speech_started()
    await handler._on_stt_speech_started()

    assert handler._soft_duck_signal_count == 3


# ── 5. Config defaults ──────────────────────────────────────────────────────


def test_soft_duck_config_defaults():
    assert settings.VOICE_SOFT_DUCK_ENABLED is True
    assert settings.VOICE_SOFT_DUCK_MS == 400
    assert settings.VOICE_SOFT_DUCK_GAIN == pytest.approx(0.35)


# ── 6. Regression: duck must apply WITHIN an in-flight chunk, not just the
#      next one (see _resolve_voice_volume_base()/_base_voice_gain_from_runtime()
#      + per-frame frame_gain_fn threading in app/voice/tts_stream_mixin.py and
#      app/utils/audio_utils.py::stream_mulaw_bytes_over_twilio()). ─────────────
#
# Before this fix, _resolve_voice_volume() (which applies the duck) was called
# ONCE per TTS chunk before that chunk's frame-streaming loop began, so a
# SpeechStarted event landing mid-chunk (the realistic "caller interrupts
# mid-sentence" case) never audibly ducked the chunk already in flight -- by
# the time the NEXT chunk resolved gain, the 400ms duck window had usually
# already expired.


# Mu-law 0x00 decodes to a large-magnitude (near-max) linear PCM sample --
# i.e. loud, clearly-non-silent content. (0xFF, by contrast, decodes to a
# zero/silence sample -- that's why it's used as the padding/priming byte
# elsewhere in this codebase.)
_LOUD_FRAME = bytes([0x00]) * MULAW_FRAME_BYTES


def _make_twilio_tts_handler():
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
    h._twilio_buffer_primed = True  # skip priming frames to simplify frame counting
    h._is_background_audio_enabled = lambda: False
    h._soft_duck_until_mono = 0.0
    h._soft_duck_gain = 0.35
    h.websocket = MagicMock()
    h.websocket.send_json = AsyncMock()
    return h


@pytest.mark.asyncio
async def test_streaming_branch_ducks_remaining_frames_of_same_inflight_chunk():
    """
    Exercises the REAL `_stream_tts_chunk` streaming branch (async for
    chunk_bytes in audio_iter) end to end -- not _apply_soft_duck() in
    isolation. A SpeechStarted event (simulated by setting
    `_soft_duck_until_mono`, exactly like
    BidirectionalStreamHandler._on_stt_speech_started does) fires partway
    through a single multi-frame chunk's own audio iterator. Frames sent
    before that point must be full volume; frames sent after -- still
    within the SAME chunk/_stream_tts_chunk() call -- must be ducked.
    """
    h = _make_twilio_tts_handler()

    async def audio_iter():
        # Frame 1: streamed out before any barge-in signal.
        yield _LOUD_FRAME
        # Simulate Deepgram SpeechStarted firing mid-utterance.
        h._soft_duck_until_mono = time.monotonic() + 0.4
        # Frames 2 and 3: still part of the SAME chunk/iterator, but now
        # inside the duck window -- must come out quieter.
        yield _LOUD_FRAME
        yield _LOUD_FRAME

    runtime = ResolvedTtsRuntime(
        adapter_slug="rime",
        voice_external_id="voice-1",
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )
    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=runtime
    ):
        await h._stream_tts_chunk(
            "A longer sentence with several words in it.",
            is_final=True,
            prefetched_bytes=audio_iter(),
        )

    sent_frames = [
        base64.b64decode(c.args[0]["media"]["payload"])
        for c in h.websocket.send_json.await_args_list
    ]
    # Drop any trailing 0xFF silence-drain frames appended for is_final=True.
    silence = bytes([0xFF]) * MULAW_FRAME_BYTES
    content_frames = [f for f in sent_frames if f != silence]

    assert len(content_frames) == 3
    first, second, third = content_frames

    # Pre-duck frame: untouched (gain == 1.0 short-circuits apply_volume_fade).
    assert first == _LOUD_FRAME
    # Post-duck frames (same chunk, same _stream_tts_chunk() call): scaled
    # down -- this is the exact behavior that was broken before the fix.
    assert second != _LOUD_FRAME
    assert third != _LOUD_FRAME


@pytest.mark.asyncio
async def test_stream_mulaw_bytes_over_twilio_reresolves_gain_every_frame():
    """
    Direct coverage of the shared low-level primitive
    (stream_mulaw_bytes_over_twilio's `frame_gain_fn`) used by the
    batch-fallback and prefix/suffix TTS paths in tts_stream_mixin.py: gain
    must be re-resolved on EVERY frame of a single `audio_bytes` buffer, not
    once for the whole buffer -- otherwise a duck window opening mid-buffer
    (as real wall-clock time elapses across the paced 20ms frame sends)
    would never be reflected in that buffer's remaining frames.
    """
    ws = MagicMock()
    ws.send_json = AsyncMock()

    audio_bytes = _LOUD_FRAME * 3  # 3 frames, all identical loud content

    # Gain flips from 1.0 -> 0.35 after the first frame has been resolved,
    # simulating a duck window opening mid-send (real wall-clock elapses
    # between paced frame sends in production).
    calls = {"n": 0}

    def frame_gain_fn():
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 0.35

    await stream_mulaw_bytes_over_twilio(
        websocket=ws,
        stream_sid="MZtest",
        audio_bytes=audio_bytes,
        pace_20ms=False,  # no real sleeping needed to prove the per-frame call
        frame_gain_fn=frame_gain_fn,
    )

    # frame_gain_fn must have been invoked once per frame, not once total.
    assert calls["n"] == 3

    sent_frames = [
        base64.b64decode(c.args[0]["media"]["payload"])
        for c in ws.send_json.await_args_list
    ]
    assert len(sent_frames) == 3
    assert sent_frames[0] == _LOUD_FRAME  # gain 1.0 -> untouched
    assert sent_frames[1] != _LOUD_FRAME  # gain 0.35 -> ducked
    assert sent_frames[2] != _LOUD_FRAME  # gain 0.35 -> ducked
