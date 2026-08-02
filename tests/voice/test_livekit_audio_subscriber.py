"""
Unit tests for the LiveKit caller-audio → STT ingestion path
(app/voice/audio_transcoder.py + app/voice/livekit_audio_subscriber.py).

Regression coverage for a production stall: the browser-based LiveKit
agent-join path (app/voice/livekit_browser_call_handler.py) reuses this
subscriber/resampler unmodified from the older Twilio+Google-STT LiveKit
bridge, but neither file had any test coverage before this change — the
resample→STT-feed handoff was only ever exercised with every LiveKit/STT
boundary mocked out (see test_livekit_browser_call_handler.py), so a defect
in the real per-frame processing chain was invisible to CI.

Bug: a single frame's ffmpeg (re)start failure (binary missing from PATH,
transient fork/exec error, etc.) raised out of
LiveKitAudioProcessor._ffmpeg_convert() uncaught, which propagated out of
LiveKitAudioSubscriber._subscribe_and_transcode()'s per-frame processing
call. Since that call sat directly inside the `async for audio_frame_event
in audio_stream:` loop with no per-frame try/except, the exception unwound
the entire loop (caught only by the outer broad `except Exception` around
the whole subscribe/transcode call), permanently ending STT ingestion for
the rest of the call after the very first frame that hit the failure —
matching the reported symptom exactly: "first frame received and logged,
then nothing else happens."

Fix: (1) LiveKitAudioProcessor._ffmpeg_convert now catches restart failures
per-frame and returns b"" (dropping just that frame) instead of raising;
(2) LiveKitAudioSubscriber's per-frame loop additionally wraps
processor.process_frame() in a try/except so any other unexpected
per-frame error also can't kill the whole subscription.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.audio_transcoder import LiveKitAudioProcessor
from app.voice.livekit_audio_subscriber import LiveKitAudioSubscriber

# A single browser-side LiveKit audio frame as reported in production:
# 48kHz mono, 960 bytes (480 samples * 2 bytes/sample = 10ms @ 48kHz).
_FRAME_48K_MONO = bytes(range(256)) * 3 + bytes(192)  # 960 bytes of non-zero data
assert len(_FRAME_48K_MONO) == 960


class _FakeAudioFrame:
    def __init__(self, data: bytes, sample_rate: int = 48000, num_channels: int = 1):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels


class _FakeAudioFrameEvent:
    def __init__(self, frame: _FakeAudioFrame):
        self.frame = frame


class _FakeAudioStream:
    """Minimal async-iterable stand-in for livekit.rtc.AudioStream."""

    def __init__(self, frames: list[_FakeAudioFrame]):
        self._frames = frames

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for f in self._frames:
            yield _FakeAudioFrameEvent(f)


# ─────────────────────────────────────────────────────────────────────────────
# LiveKitAudioProcessor: ffmpeg-restart failure must not raise out of
# process_frame — it should degrade to "drop this frame" and recover once
# ffmpeg becomes available again.
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveKitAudioProcessorResilience:
    @pytest.mark.asyncio
    async def test_ffmpeg_restart_failure_does_not_raise(self):
        processor = LiveKitAudioProcessor(output_sample_rate=16000)

        with patch.object(
            processor, "_restart_ffmpeg", new=AsyncMock(side_effect=RuntimeError("ffmpeg not found in PATH"))
        ):
            # Must return empty bytes, not propagate the RuntimeError.
            out = await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)

        assert out == b""

    @pytest.mark.asyncio
    async def test_recovers_once_ffmpeg_becomes_available(self):
        processor = LiveKitAudioProcessor(output_sample_rate=16000)

        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.stdin.write = MagicMock()
        fake_proc.stdin.drain = AsyncMock()
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.read = AsyncMock(return_value=b"\x01\x02" * 100)

        async def _fail_then_succeed(sample_rate, num_channels):
            processor._ffmpeg_process = fake_proc
            processor._ffmpeg_input_rate = sample_rate
            processor._ffmpeg_input_channels = num_channels

        with patch.object(
            processor, "_restart_ffmpeg", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            out1 = await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)
        assert out1 == b""

        # Simulate the restart backoff window having elapsed (the processor
        # rate-limits retry *attempts*, not just log lines, so a real caller
        # waits out this same window before the next successful restart).
        processor._last_restart_failure = None

        with patch.object(processor, "_restart_ffmpeg", new=_fail_then_succeed):
            out2 = await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)
        assert out2 != b""

    @pytest.mark.asyncio
    async def test_mid_call_ffmpeg_death_self_heals_on_next_frame(self):
        """
        A second instance of the same production symptom: ffmpeg starts fine
        for frame 1, then the process dies (broken pipe) partway through the
        call. Before the fix, self._ffmpeg_process was never cleared on this
        path, so every subsequent frame kept hitting the same dead process
        forever — permanent silence for the rest of the call, exactly like
        the original bug, just triggered by a mid-call crash instead of a
        failed start. After the fix, the dead process must be cleared so the
        next frame re-enters _restart_ffmpeg and recovers.
        """
        processor = LiveKitAudioProcessor(output_sample_rate=16000)

        dead_proc = MagicMock()
        dead_proc.stdin = MagicMock()
        dead_proc.stdin.write = MagicMock(side_effect=BrokenPipeError("pipe closed"))
        dead_proc.stdout = MagicMock()
        dead_proc.returncode = None
        dead_proc.wait = AsyncMock(return_value=None)
        dead_proc.kill = MagicMock()

        processor._ffmpeg_process = dead_proc
        processor._ffmpeg_input_rate = 48000
        processor._ffmpeg_input_channels = 1

        out1 = await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)
        assert out1 == b""
        # The dead process reference must be cleared, not left in place.
        assert processor._ffmpeg_process is None

        # Simulate the restart backoff window having elapsed.
        processor._last_restart_failure = None

        healthy_proc = MagicMock()
        healthy_proc.stdin = MagicMock()
        healthy_proc.stdin.write = MagicMock()
        healthy_proc.stdin.drain = AsyncMock()
        healthy_proc.stdout = MagicMock()
        healthy_proc.stdout.read = AsyncMock(return_value=b"\x01\x02" * 100)

        async def _restart(sample_rate, num_channels):
            processor._ffmpeg_process = healthy_proc
            processor._ffmpeg_input_rate = sample_rate
            processor._ffmpeg_input_channels = num_channels

        with patch.object(processor, "_restart_ffmpeg", new=_restart):
            out2 = await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)

        # Recovers on the very next frame instead of being stuck on the dead
        # process for the rest of the call.
        assert out2 != b""


# ─────────────────────────────────────────────────────────────────────────────
# LiveKitAudioSubscriber: a single bad frame must not abort ingestion for the
# rest of the call — this is the exact regression scenario from production.
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveKitAudioSubscriberFrameLoop:
    @pytest.mark.asyncio
    async def test_run_loop_survives_processing_error_end_to_end(self):
        """
        Full _subscribe_and_transcode() path (not just the extracted loop
        body): first frame's process_frame() raises, second frame succeeds —
        asserts SttPipeline.feed_audio_chunk (the "STT send" stage) is still
        called for the surviving frame, i.e. the subscription loop is not
        aborted by the first frame's error.
        """
        stt_pipeline = MagicMock()
        stt_pipeline.feed_audio_chunk = AsyncMock()

        subscriber = LiveKitAudioSubscriber(
            room_name="room_test", stt_pipeline=stt_pipeline, output_sample_rate=16000
        )

        frames = [_FakeAudioFrame(_FRAME_48K_MONO), _FakeAudioFrame(_FRAME_48K_MONO)]
        audio_stream = _FakeAudioStream(frames)

        fake_room = MagicMock()
        fake_room.connect = AsyncMock()
        fake_room.disconnect = AsyncMock()
        fake_room.on = MagicMock()

        fake_rtc = MagicMock()
        fake_rtc.Room = MagicMock(return_value=fake_room)
        fake_rtc.TrackKind.KIND_AUDIO = "audio"
        fake_rtc.AudioStream = MagicMock(return_value=audio_stream)

        fake_processor = MagicMock()
        fake_processor.process_frame = AsyncMock(
            side_effect=[RuntimeError("ffmpeg not found in PATH"), b"\x01\x02" * 50]
        )
        subscriber._processor = fake_processor

        fake_track = MagicMock()
        fake_track.kind = "audio"
        fake_track.sid = "TR_test"
        fake_participant = MagicMock()
        fake_participant.identity = "caller-room_test"

        async def _connect_and_fire_track_subscribed(*_args, **_kwargs):
            # Simulate LiveKit firing track_subscribed right after connect.
            for call in fake_room.on.call_args_list:
                if call.args and call.args[0] == "track_subscribed":
                    handler = call.args[1]
                    handler(fake_track, MagicMock(), fake_participant)
                    return

        fake_room.connect.side_effect = _connect_and_fire_track_subscribed

        with patch("app.core.config.settings.LIVEKIT_ENABLED", True), \
             patch.dict("sys.modules", {"livekit": MagicMock(rtc=fake_rtc)}):
            await asyncio.wait_for(
                subscriber._subscribe_and_transcode("ws://fake", "fake-token"), timeout=5.0
            )

        stt_pipeline.feed_audio_chunk.assert_awaited_once()
