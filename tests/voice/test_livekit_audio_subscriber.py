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
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.voice.audio_transcoder import (
    _MAX_OUTPUT_QUEUE_CHUNKS,
    _MAX_WRITE_QUEUE_FRAMES,
    _STUCK_RESET_SECONDS,
    LiveKitAudioProcessor,
)
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


async def _drain_until_nonempty(
    processor: LiveKitAudioProcessor, frame: bytes, attempts: int = 50
) -> bytes:
    """Call process_frame() repeatedly, yielding to the event loop between
    calls, until the reader background task has produced output to drain.

    process_frame() no longer awaits ffmpeg I/O directly — it only enqueues a
    write and drains whatever the independent writer/reader tasks have
    already produced (see the module docstring in
    app/voice/audio_transcoder.py) — so a single call right after a (re)start
    can legitimately still see no output yet. This mirrors how a real caller
    feeding frames on a ~10-20ms cadence would observe output arriving once
    the background tasks catch up.
    """
    out = b""
    for _ in range(attempts):
        out = await processor.process_frame(frame, sample_rate=48000, num_channels=1)
        if out:
            return out
        await asyncio.sleep(0.01)
    return out


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
        """process_frame() never blocks on ffmpeg I/O directly anymore — a
        successful (re)start spins up independent writer/reader background
        tasks, so recovery is observed by polling the non-blocking drain
        until the reader task has had a chance to run, rather than by a
        single awaited call."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        try:
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

            # Simulate the restart backoff window having elapsed (the
            # processor rate-limits retry *attempts*, not just log lines, so
            # a real caller waits out this same window before the next
            # successful restart).
            processor._last_restart_failure = None

            with patch.object(processor, "_restart_ffmpeg", new=_fail_then_succeed):
                out2 = await _drain_until_nonempty(processor, _FRAME_48K_MONO)
            assert out2 != b""
            # Writer/reader background tasks must actually be up and running
            # against the newly (re)started process.
            assert processor._writer_task is not None and not processor._writer_task.done()
            assert processor._reader_task is not None and not processor._reader_task.done()
        finally:
            await processor.close()

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

        The write itself now happens on the background writer task rather
        than inline in process_frame(), so the broken pipe is observed by the
        writer task and must be given an event-loop turn to propagate before
        asserting the process reference was cleared.
        """
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        try:
            proc = MagicMock()
            proc.stdin = MagicMock()
            proc.stdin.write = MagicMock()
            proc.stdin.drain = AsyncMock()
            proc.stdout = MagicMock()
            proc.stdout.read = AsyncMock(return_value=b"\x01\x02" * 100)
            proc.returncode = None
            proc.wait = AsyncMock(return_value=None)
            proc.kill = MagicMock()

            async def _restart_healthy(sample_rate, num_channels):
                processor._ffmpeg_process = proc
                processor._ffmpeg_input_rate = sample_rate
                processor._ffmpeg_input_channels = num_channels

            with patch.object(processor, "_restart_ffmpeg", new=_restart_healthy):
                await _drain_until_nonempty(processor, _FRAME_48K_MONO)
            assert processor._ffmpeg_process is proc

            # Simulate the ffmpeg process dying mid-call: the next stdin
            # write raises a broken pipe on the writer task.
            proc.stdin.write = MagicMock(side_effect=BrokenPipeError("pipe closed"))

            await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)
            # Give the writer task a turn to observe the broken pipe and
            # mark the process broken.
            for _ in range(20):
                if processor._ffmpeg_process is None:
                    break
                await asyncio.sleep(0.01)

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
                out2 = await _drain_until_nonempty(processor, _FRAME_48K_MONO)

            # Recovers on a subsequent frame instead of being stuck on the
            # dead process for the rest of the call.
            assert out2 != b""
        finally:
            await processor.close()

    @pytest.mark.asyncio
    async def test_recent_output_gap_does_not_reset_ffmpeg(self):
        """
        Regression: a short gap since ffmpeg's last stdout flush (e.g.
        because the caller is mid-utterance and ffmpeg hasn't had a full
        frame's worth to flush yet) is a normal condition, not evidence the
        process is dead. This replaces the old per-call "N consecutive read
        timeouts" heuristic (removed along with the blocking read it used to
        guard) with the new time-based watchdog: it must only reset a
        process that has been stuck for >= _STUCK_RESET_SECONDS, not on
        every check.
        """
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        try:
            proc = MagicMock()
            processor._ffmpeg_process = proc
            now = time.monotonic()
            # Output flowing recently, writes actively flowing too — well
            # under the stuck threshold.
            processor._last_output_at = now - 0.5
            processor._last_write_at = now - 0.05

            processor._check_watchdog()

            assert processor._ffmpeg_process is proc
        finally:
            await processor.close()

    @pytest.mark.asyncio
    async def test_sustained_no_output_while_writing_resets_ffmpeg(self):
        """A process that has stopped producing output for longer than
        _STUCK_RESET_SECONDS while frames are still actively being written
        is genuinely stuck and must still eventually self-heal via the
        watchdog resetting it (clearing the process reference so the next
        frame re-enters _restart_ffmpeg)."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        try:
            proc = MagicMock()
            proc.returncode = None
            proc.wait = AsyncMock(return_value=None)
            proc.kill = MagicMock()
            processor._ffmpeg_process = proc
            now = time.monotonic()
            # No output for well over the stuck threshold, but writes are
            # still flowing recently — this is the "genuinely stuck" case,
            # not an idle gap.
            processor._last_output_at = now - (_STUCK_RESET_SECONDS + 1.0)
            processor._last_write_at = now - 0.05

            processor._check_watchdog()
            # _mark_broken() schedules an async task to kill/reap the stale
            # process — give it a turn to run.
            await asyncio.sleep(0.05)

            assert processor._ffmpeg_process is None
            proc.kill.assert_called_once()
        finally:
            await processor.close()

    @pytest.mark.asyncio
    async def test_watchdog_reset_with_live_tasks_spins_up_fresh_writer_reader_pair(self):
        """End-to-end coverage for the watchdog reset: unlike
        test_sustained_no_output_while_writing_resets_ffmpeg (which drives
        _check_watchdog() in isolation with no writer/reader tasks running
        at all), this starts a real writer/reader task pair against a live
        fake process, forces the "stuck" condition, invokes the watchdog,
        and then verifies a *subsequent* process_frame() call actually
        cancels/joins the now-stale tasks and replaces them with a genuinely
        fresh pair bound to a new process — not just that kill() was called
        on the raw process mock."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        try:
            proc = MagicMock()
            proc.stdin = MagicMock()
            proc.stdin.write = MagicMock()
            proc.stdin.drain = AsyncMock()
            proc.stdin.is_closing = MagicMock(return_value=False)
            proc.stdin.close = MagicMock()
            proc.stdin.wait_closed = AsyncMock(return_value=None)
            proc.stdout = MagicMock()
            proc.stdout.read = AsyncMock(return_value=b"\x01\x02" * 100)
            proc.returncode = None
            proc.wait = AsyncMock(return_value=None)
            proc.kill = MagicMock()

            async def _restart_healthy(sample_rate, num_channels):
                processor._ffmpeg_process = proc
                processor._ffmpeg_input_rate = sample_rate
                processor._ffmpeg_input_channels = num_channels

            with patch.object(processor, "_restart_ffmpeg", new=_restart_healthy):
                await _drain_until_nonempty(processor, _FRAME_48K_MONO)

            stale_writer_task = processor._writer_task
            stale_reader_task = processor._reader_task
            assert stale_writer_task is not None and not stale_writer_task.done()
            assert stale_reader_task is not None and not stale_reader_task.done()
            assert processor._ffmpeg_process is proc

            # Force the "genuinely stuck" state: no output for well over the
            # threshold while writes are still recent.
            now = time.monotonic()
            processor._last_output_at = now - (_STUCK_RESET_SECONDS + 1.0)
            processor._last_write_at = now - 0.05

            processor._check_watchdog()
            # _mark_broken() only clears the process reference and schedules
            # a fire-and-forget kill/reap — give it a turn to run.
            await asyncio.sleep(0.05)

            assert processor._ffmpeg_process is None
            proc.kill.assert_called_once()
            # The watchdog itself does not cancel the stale writer/reader
            # tasks (see _mark_broken's docstring — a task can't await-cancel
            # itself) — they're still alive at this point, just orphaned.
            assert processor._writer_task is stale_writer_task
            assert processor._reader_task is stale_reader_task
            assert not stale_writer_task.done()
            assert not stale_reader_task.done()

            # Simulate the restart backoff window having elapsed.
            processor._last_restart_failure = None

            new_proc = MagicMock()
            new_proc.stdin = MagicMock()
            new_proc.stdin.write = MagicMock()
            new_proc.stdin.drain = AsyncMock()
            new_proc.stdin.is_closing = MagicMock(return_value=False)
            new_proc.stdin.close = MagicMock()
            new_proc.stdin.wait_closed = AsyncMock(return_value=None)
            new_proc.stdout = MagicMock()
            new_proc.stdout.read = AsyncMock(return_value=b"\x03\x04" * 100)
            new_proc.returncode = None
            new_proc.wait = AsyncMock(return_value=None)
            new_proc.kill = MagicMock()

            async def _restart_new(sample_rate, num_channels):
                processor._ffmpeg_process = new_proc
                processor._ffmpeg_input_rate = sample_rate
                processor._ffmpeg_input_channels = num_channels

            with patch.object(processor, "_restart_ffmpeg", new=_restart_new):
                out = await _drain_until_nonempty(processor, _FRAME_48K_MONO)

            assert out != b""
            # The next process_frame() call must have cleanly torn down the
            # stale pair via _stop_pipeline_locked()...
            assert stale_writer_task.cancelled() or stale_writer_task.done()
            assert stale_reader_task.cancelled() or stale_reader_task.done()
            # ...and spun up a genuinely fresh pair bound to the new process.
            assert processor._writer_task is not None
            assert processor._writer_task is not stale_writer_task
            assert not processor._writer_task.done()
            assert processor._reader_task is not None
            assert processor._reader_task is not stale_reader_task
            assert not processor._reader_task.done()
            assert processor._ffmpeg_process is new_proc
        finally:
            await processor.close()

    @pytest.mark.asyncio
    async def test_enqueue_write_drops_oldest_keeps_newest_when_queue_full(self, caplog):
        """_enqueue_write's backpressure path: once the write queue is at its
        bound, the oldest queued frame must be evicted in favor of the
        newest one (never blocking/raising for the caller), the overflow
        warning must log once per outage (not once per dropped frame), and
        must reset once the queue accepts a write without overflowing again."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        processor._write_queue = asyncio.Queue(maxsize=_MAX_WRITE_QUEUE_FRAMES)

        frames = [f"frame-{i}".encode() for i in range(_MAX_WRITE_QUEUE_FRAMES)]
        for f in frames:
            processor._enqueue_write(f)
        assert processor._write_queue.qsize() == _MAX_WRITE_QUEUE_FRAMES
        assert processor._write_backpressure_logged is False

        with caplog.at_level(logging.WARNING):
            processor._enqueue_write(b"overflow-1")

        assert processor._write_queue.qsize() == _MAX_WRITE_QUEUE_FRAMES
        queued = []
        while not processor._write_queue.empty():
            queued.append(processor._write_queue.get_nowait())
        assert frames[0] not in queued  # oldest evicted
        assert queued[-1] == b"overflow-1"  # newest kept, at the back
        overflow_warnings = [r for r in caplog.records if "write queue full" in r.message]
        assert len(overflow_warnings) == 1
        assert processor._write_backpressure_logged is True

        # Re-fill to the bound and overflow again without a successful drain
        # in between — must NOT log a second time for the same outage.
        for f in frames:
            processor._write_queue.put_nowait(f)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            processor._enqueue_write(b"overflow-2")
        overflow_warnings = [r for r in caplog.records if "write queue full" in r.message]
        assert len(overflow_warnings) == 0

        # A write that doesn't hit the full-queue branch resets the flag so
        # a later, distinct outage logs again.
        processor._write_queue.get_nowait()  # make room
        processor._enqueue_write(b"normal-frame")
        assert processor._write_backpressure_logged is False

    @pytest.mark.asyncio
    async def test_push_output_nowait_drops_oldest_keeps_newest_when_queue_full(self, caplog):
        """Equivalent backpressure coverage for the reader-side output
        queue via _push_output_nowait: oldest chunk evicted, newest chunk
        kept, overflow warning rate-limited to once per outage, and reset
        once a push succeeds without overflowing."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)
        output_queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_OUTPUT_QUEUE_CHUNKS)
        processor._output_queue = output_queue

        chunks = [f"chunk-{i}".encode() for i in range(_MAX_OUTPUT_QUEUE_CHUNKS)]
        for c in chunks:
            processor._push_output_nowait(output_queue, c)
        assert output_queue.qsize() == _MAX_OUTPUT_QUEUE_CHUNKS
        assert processor._output_backpressure_logged is False

        with caplog.at_level(logging.WARNING):
            processor._push_output_nowait(output_queue, b"overflow-1")

        assert output_queue.qsize() == _MAX_OUTPUT_QUEUE_CHUNKS
        queued = []
        while not output_queue.empty():
            queued.append(output_queue.get_nowait())
        assert chunks[0] not in queued  # oldest evicted
        assert queued[-1] == b"overflow-1"  # newest kept, at the back
        overflow_warnings = [r for r in caplog.records if "output queue full" in r.message]
        assert len(overflow_warnings) == 1
        assert processor._output_backpressure_logged is True

        # Re-fill to the bound and overflow again without a successful drain
        # in between — must NOT log a second time for the same outage.
        for c in chunks:
            output_queue.put_nowait(c)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            processor._push_output_nowait(output_queue, b"overflow-2")
        overflow_warnings = [r for r in caplog.records if "output queue full" in r.message]
        assert len(overflow_warnings) == 0

        # A push that doesn't hit the full-queue branch resets the flag so a
        # later, distinct outage logs again.
        output_queue.get_nowait()  # make room
        processor._push_output_nowait(output_queue, b"normal-chunk")
        assert processor._output_backpressure_logged is False

    @pytest.mark.asyncio
    async def test_close_cancels_writer_and_reader_tasks_and_reaps_process(self):
        """Graceful shutdown: close() must cancel both background tasks
        (including unblocking a writer parked on write_queue.get() via the
        None sentinel) and reap the ffmpeg process, leaving the processor in
        a clean state that a subsequent process_frame() call could safely
        restart from."""
        processor = LiveKitAudioProcessor(output_sample_rate=16000)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdin.is_closing = MagicMock(return_value=False)
        proc.stdin.close = MagicMock()
        proc.stdin.wait_closed = AsyncMock(return_value=None)
        proc.stdout = MagicMock()
        # Reader task blocks "forever" (never returns) until cancelled —
        # exercises the task.cancel() path rather than a read that happens
        # to return on its own.
        stdout_read_started = asyncio.Event()

        async def _blocking_read(*_args, **_kwargs):
            stdout_read_started.set()
            await asyncio.sleep(3600)

        proc.stdout.read = _blocking_read
        proc.returncode = None
        proc.wait = AsyncMock(return_value=None)
        proc.kill = MagicMock()

        async def _restart(sample_rate, num_channels):
            processor._ffmpeg_process = proc
            processor._ffmpeg_input_rate = sample_rate
            processor._ffmpeg_input_channels = num_channels

        with patch.object(processor, "_restart_ffmpeg", new=_restart):
            await processor.process_frame(_FRAME_48K_MONO, sample_rate=48000, num_channels=1)

        writer_task = processor._writer_task
        reader_task = processor._reader_task
        assert writer_task is not None
        assert reader_task is not None
        await asyncio.wait_for(stdout_read_started.wait(), timeout=1.0)
        # The writer task has drained the one enqueued frame and is now
        # parked on write_queue.get() waiting for the next frame — nothing
        # left to feed it before shutdown.

        await processor.close()

        assert writer_task.cancelled() or writer_task.done()
        assert reader_task.cancelled() or reader_task.done()
        assert processor._writer_task is None
        assert processor._reader_task is None
        assert processor._write_queue is None
        assert processor._output_queue is None
        assert processor._ffmpeg_process is None

        # The stale process is reaped by a fire-and-forget task kicked off
        # at the end of _stop_pipeline_locked() — give it a turn to run
        # before asserting it actually killed the process.
        for _ in range(20):
            if proc.kill.called:
                break
            await asyncio.sleep(0.01)
        proc.kill.assert_called_once()


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
