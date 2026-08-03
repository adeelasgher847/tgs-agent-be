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

    @pytest.mark.asyncio
    async def test_occasional_read_timeout_does_not_reset_ffmpeg(self):
        """
        Regression: a single (or occasional) stdout-read timeout is a normal
        condition for a live/freshly-started ffmpeg process that hasn't
        buffered enough input yet to flush output for a tiny 10-20ms frame —
        it is NOT evidence the process is dead. A prior fix incorrectly
        treated every timeout the same as a broken pipe / dead process and
        reset (killed + forced a respawn) on every single one — which meant
        ffmpeg was killed before it ever got a chance to reach steady state,
        producing an infinite respawn loop where caller audio was NEVER
        successfully converted (observed in production: "ffmpeg read timed
        out — process likely stuck, resetting" logged continuously, every
        ~1.5s, for the whole call). A timeout must leave the same process in
        place so it can catch up on a later frame.
        """
        processor = LiveKitAudioProcessor(output_sample_rate=16000)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdout = MagicMock()
        proc.stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())

        processor._ffmpeg_process = proc
        processor._ffmpeg_input_rate = 48000
        processor._ffmpeg_input_channels = 1

        for _ in range(5):
            out = await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)
            assert out == b""

        # After several (but not too many) consecutive timeouts, the SAME
        # process must still be in place — not reset/killed.
        assert processor._ffmpeg_process is proc

    @pytest.mark.asyncio
    async def test_sustained_read_timeouts_eventually_reset_ffmpeg(self):
        """A process that never recovers across many consecutive timeouts is
        genuinely stuck and must still eventually self-heal."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdout = MagicMock()
        proc.stdout.read = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.returncode = None
        proc.wait = AsyncMock(return_value=None)
        proc.kill = MagicMock()

        processor._ffmpeg_process = proc
        processor._ffmpeg_input_rate = 48000
        processor._ffmpeg_input_channels = 1

        from app.voice.audio_transcoder import _MAX_CONSECUTIVE_READ_TIMEOUTS

        for _ in range(_MAX_CONSECUTIVE_READ_TIMEOUTS):
            await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)

        assert processor._ffmpeg_process is None


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


# ─────────────────────────────────────────────────────────────────────────────
# LiveKitAudioSubscriber: caller audio track arriving after the first 30s
# window must not be treated as a permanent failure — regression coverage for
# a production stall where the subscriber gave up and disconnected after
# exactly one 30s timeout, permanently killing STT ingestion for the rest of
# a call that continued for minutes afterward.
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveKitAudioSubscriberCallerTrackWaitLoop:
    @pytest.mark.asyncio
    async def test_stop_requested_shortly_after_start_ends_wait_promptly(self):
        """
        stop_event set shortly after start, before the track is ever found —
        must return False promptly rather than blocking for the full 30s
        internal timeout window.
        """
        subscriber = LiveKitAudioSubscriber(
            room_name="room_test", stt_pipeline=MagicMock(), output_sample_rate=16000
        )
        caller_track_found = asyncio.Event()
        room_disconnected = asyncio.Event()

        async def _stop_soon():
            await asyncio.sleep(0.05)
            subscriber._stop_event.set()

        stopper = asyncio.create_task(_stop_soon())
        result = await asyncio.wait_for(
            subscriber._wait_for_caller_track(caller_track_found, room_disconnected),
            timeout=5.0,
        )
        await stopper
        assert result is False

    @pytest.mark.asyncio
    async def test_caller_track_arriving_after_first_timeout_window_is_not_lost(self):
        """
        Core regression: the caller's track is NOT found within the first
        30s window, but arrives shortly after. Before the fix, the subscriber
        gave up permanently on the first timeout and disconnected — STT would
        never see this track no matter how long the call continued. After
        the fix, the wait loop keeps waiting (bounded by asyncio.wait's
        internal 30s timeout, looped) and must pick the track up as soon as
        it's set, without ever disconnecting.
        """
        subscriber = LiveKitAudioSubscriber(
            room_name="room_test", stt_pipeline=MagicMock(), output_sample_rate=16000
        )
        caller_track_found = asyncio.Event()
        room_disconnected = asyncio.Event()

        # Patch the loop's internal 30s timeout down to something fast so the
        # test doesn't actually wait 30 real seconds to exercise a second
        # iteration of the retry loop.
        async def _set_track_after_delay():
            await asyncio.sleep(0.2)
            caller_track_found.set()

        setter = asyncio.create_task(_set_track_after_delay())

        _real_asyncio_wait = asyncio.wait

        with patch("asyncio.wait") as _wrapped:
            # Force the internal per-iteration timeout way down so a single
            # "timeout" retry cycle happens well before the track is set at
            # t=0.2s, proving the loop survives at least one timeout without
            # giving up.
            async def _wait_with_short_timeout(tasks, timeout=None, return_when=None):
                return await _real_asyncio_wait(tasks, timeout=0.05, return_when=return_when)

            _wrapped.side_effect = _wait_with_short_timeout

            result = await asyncio.wait_for(
                subscriber._wait_for_caller_track(caller_track_found, room_disconnected),
                timeout=5.0,
            )

        await setter
        assert result is True
        # Must not have given up: no stop/disconnect should have been needed.
        assert not subscriber._stop_event.is_set()
        assert not room_disconnected.is_set()
        # The rate-limited timeout warning must have fired at least once,
        # proving the loop actually looped through a timeout before success.
        assert subscriber._caller_track_wait_timeout_logged is True

    @pytest.mark.asyncio
    async def test_stop_event_ends_the_wait_without_disconnect_flag(self):
        subscriber = LiveKitAudioSubscriber(
            room_name="room_test", stt_pipeline=MagicMock(), output_sample_rate=16000
        )
        caller_track_found = asyncio.Event()
        room_disconnected = asyncio.Event()
        subscriber._stop_event.set()

        result = await asyncio.wait_for(
            subscriber._wait_for_caller_track(caller_track_found, room_disconnected),
            timeout=2.0,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_room_disconnected_ends_the_wait(self):
        subscriber = LiveKitAudioSubscriber(
            room_name="room_test", stt_pipeline=MagicMock(), output_sample_rate=16000
        )
        caller_track_found = asyncio.Event()
        room_disconnected = asyncio.Event()
        room_disconnected.set()

        result = await asyncio.wait_for(
            subscriber._wait_for_caller_track(caller_track_found, room_disconnected),
            timeout=2.0,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_full_subscribe_loop_survives_late_caller_track_end_to_end(self):
        """
        Full _subscribe_and_transcode() path: caller track arrives late (after
        the on_track_subscribed callback fires from a delayed simulated
        LiveKit event), and STT ingestion must still proceed — the room must
        not have been disconnected in between.
        """
        stt_pipeline = MagicMock()
        stt_pipeline.feed_audio_chunk = AsyncMock()

        subscriber = LiveKitAudioSubscriber(
            room_name="room_test", stt_pipeline=stt_pipeline, output_sample_rate=16000
        )

        frames = [_FakeAudioFrame(_FRAME_48K_MONO)]
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
        fake_processor.process_frame = AsyncMock(return_value=b"\x01\x02" * 50)
        subscriber._processor = fake_processor

        fake_track = MagicMock()
        fake_track.kind = "audio"
        fake_track.sid = "TR_test"
        fake_participant = MagicMock()
        fake_participant.identity = "caller-room_test"

        async def _connect_and_fire_track_subscribed_late(*_args, **_kwargs):
            async def _fire_late():
                await asyncio.sleep(0.15)
                for call in fake_room.on.call_args_list:
                    if call.args and call.args[0] == "track_subscribed":
                        handler = call.args[1]
                        handler(fake_track, MagicMock(), fake_participant)
                        return

            asyncio.create_task(_fire_late())

        fake_room.connect.side_effect = _connect_and_fire_track_subscribed_late

        _real_asyncio_wait = asyncio.wait

        with patch("app.core.config.settings.LIVEKIT_ENABLED", True), \
             patch.dict("sys.modules", {"livekit": MagicMock(rtc=fake_rtc)}), \
             patch("asyncio.wait") as _wrapped:

            async def _wait_with_short_timeout(tasks, timeout=None, return_when=None):
                return await _real_asyncio_wait(tasks, timeout=0.05, return_when=return_when)

            _wrapped.side_effect = _wait_with_short_timeout

            await asyncio.wait_for(
                subscriber._subscribe_and_transcode("ws://fake", "fake-token"), timeout=5.0
            )

        stt_pipeline.feed_audio_chunk.assert_awaited_once()
        # disconnect() is called once at the very end via the outer `finally`,
        # not prematurely after the (survived) first timeout.
        fake_room.disconnect.assert_awaited_once()
