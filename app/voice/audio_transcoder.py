"""
LiveKit / telephony audio → LINEAR16 PCM for Google STT.

LiveKit Python RTC ``AudioStream`` delivers decoded PCM (s16le), not raw Opus.
Use :class:`LiveKitAudioProcessor` per call: passthrough when already 16 kHz mono,
otherwise resample via ffmpeg child process (one per call).

Concurrency model (writer/reader decoupling)
---------------------------------------------
``process_frame()`` is called once per ~10-20ms LiveKit ``AudioFrame`` from
``LiveKitAudioSubscriber``'s tight ``async for`` loop. Earlier versions of
this class wrote each frame to ffmpeg's stdin and then *blocked that same
call* on ``asyncio.wait_for(proc.stdout.read(...), timeout=0.5)`` before the
loop could move on to the next frame — i.e. writing frame N+1 was gated on
reading frame N's output first, serialized inside a single coroutine call.
Even with prompt output flushing on the ffmpeg side (``-flush_packets 1``,
see below), this write-then-blocking-read pattern is structurally unsafe:
any slow/late read (resampler warm-up, transient CPU contention, a stuck
process before the watchdog below kicks in) directly stalls how fast
incoming LiveKit frames get drained, which is the real-time hot path feeding
STT.

To remove that coupling, writing to ffmpeg's stdin and reading its stdout
now run as two independent background ``asyncio.Task``s (``_writer_loop`` /
``_reader_loop``) per ffmpeg process, bridged by two bounded
``asyncio.Queue``s:

  LiveKit frame → process_frame() → _write_queue → _writer_loop → ffmpeg stdin
  ffmpeg stdout → _reader_loop → _output_queue → process_frame() → caller (STT)

``process_frame()`` itself never blocks on ffmpeg I/O: it enqueues the new
frame (non-blocking, bounded — see ``_MAX_WRITE_QUEUE_FRAMES``) and drains
whatever output has already accumulated (non-blocking — see
``_drain_output_nowait``), so LiveKit frames are consumed at real-time
cadence regardless of exactly when ffmpeg happens to flush.

Both queues are bounded (not unbounded buffers): if a queue is ever full —
meaning the pipeline is falling behind — the oldest entry is dropped in
favor of the newest one, since for a live conversation, fresher audio is more
valuable than stale backlog, and this bounds memory growth if ffmpeg were to
stall completely instead of blocking or growing forever.
"""
from __future__ import annotations

import asyncio
import shutil
import time

from app.core.logger import logger

_FFMPEG_OUTPUT_FORMAT = "s16le"
# Minimum gap between ffmpeg (re)start attempts once one has failed, so a
# sustained outage (binary missing, fork/exec exhaustion) doesn't retry on
# every ~20ms frame and pile onto whatever resource pressure caused it.
_RESTART_BACKOFF_SECONDS = 1.0
# Bounds on the writer-in / reader-out queues bridging process_frame() and
# the background writer/reader tasks. ~100 frames of ~10-20ms audio is 1-2s
# of buffered input — enough to absorb brief scheduling jitter without
# growing unbounded if ffmpeg genuinely stalls. Output chunks are typically
# much smaller than a whole frame's worth once flushed, so a slightly larger
# bound is used there.
_MAX_WRITE_QUEUE_FRAMES = 100
_MAX_OUTPUT_QUEUE_CHUNKS = 200
# If no output has arrived from ffmpeg for this long *while frames are still
# actively being written*, the process is treated as genuinely stuck (not
# just warming up) and is reset. Replaces the old "N consecutive per-call
# read timeouts" heuristic, which no longer applies once reads happen on a
# free-running background task rather than once per process_frame() call.
_STUCK_RESET_SECONDS = 5.0


class LiveKitAudioProcessor:
    """
    Convert LiveKit ``AudioFrame`` data to LINEAR16 mono bytes at ``output_sample_rate``.
    """

    def __init__(
        self,
        output_sample_rate: int = 16000,
        output_channels: int = 1,
    ) -> None:
        self._output_sample_rate = output_sample_rate
        self._output_channels = output_channels
        self._ffmpeg_process: asyncio.subprocess.Process | None = None
        self._ffmpeg_input_rate: int | None = None
        self._ffmpeg_input_channels: int | None = None
        self._first_frame_logged = False
        # Rate-limits the "ffmpeg (re)start failed" error to once per outage
        # (rather than once per dropped frame) so a sustained failure doesn't
        # flood the logs while still being fully visible.
        self._ffmpeg_start_failed_logged = False
        self._last_restart_failure: float | None = None

        # ── Writer/reader decoupling state ──────────────────────────────────
        self._restart_lock = asyncio.Lock()
        self._write_queue: asyncio.Queue | None = None
        self._output_queue: asyncio.Queue | None = None
        self._writer_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._last_output_at: float | None = None
        self._last_write_at: float | None = None
        # Rate-limits the backpressure-drop warnings similarly to the ffmpeg
        # start-failure log above.
        self._write_backpressure_logged = False
        self._output_backpressure_logged = False

    async def process_frame(
        self,
        raw_bytes: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> bytes:
        """Return LINEAR16 mono PCM at ``output_sample_rate`` Hz.

        Never blocks on ffmpeg subprocess I/O — see module docstring.
        """
        if not raw_bytes:
            return b""

        if not self._first_frame_logged:
            logger.info(
                "[LiveKitAudioProcessor] first frame sample_rate=%s channels=%s bytes=%s → target=%sHz",
                sample_rate,
                num_channels,
                len(raw_bytes),
                self._output_sample_rate,
            )
            self._first_frame_logged = True

        if (
            sample_rate == self._output_sample_rate
            and num_channels == self._output_channels
        ):
            logger.debug(
                "[LiveKitAudioProcessor] after resampler: passthrough bytes=%s (already %sHz/%sch)",
                len(raw_bytes), self._output_sample_rate, self._output_channels,
            )
            return raw_bytes

        out = await self._ffmpeg_convert(raw_bytes, sample_rate, num_channels)
        logger.debug(
            "[LiveKitAudioProcessor] after resampler: in_bytes=%s out_bytes=%s "
            "(%sHz/%sch → %sHz/%sch)",
            len(raw_bytes), len(out), sample_rate, num_channels,
            self._output_sample_rate, self._output_channels,
        )
        return out

    # ── Non-blocking convert: enqueue for the writer task, drain whatever
    #    the reader task has already produced ───────────────────────────────

    async def _ffmpeg_convert(
        self,
        pcm: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> bytes:
        """Resample / remix via a persistent ffmpeg child process whose stdin
        writes and stdout reads run on independent background tasks (lazy-
        started, one process per call). This call itself only enqueues
        ``pcm`` and drains already-available output — it never awaits the
        subprocess's pipes directly, so a slow/stalled ffmpeg cannot stall
        the caller's per-frame loop."""
        started = await self._ensure_running(sample_rate, num_channels)
        if not started:
            return b""

        self._enqueue_write(pcm)
        self._check_watchdog()
        return self._drain_output_nowait()

    # ── Writer-in / reader-out queue plumbing ───────────────────────────────

    def _enqueue_write(self, pcm: bytes) -> None:
        queue = self._write_queue
        if queue is None:
            return
        try:
            queue.put_nowait(pcm)
        except asyncio.QueueFull:
            # Falling behind — drop the oldest buffered frame in favor of
            # this newer one rather than blocking the caller or growing the
            # queue unbounded. Losing a stale ~10-20ms frame under sustained
            # backpressure is far less harmful than stalling real-time
            # ingestion for the whole call.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(pcm)
            except asyncio.QueueFull:
                pass
            if not self._write_backpressure_logged:
                logger.warning(
                    "[LiveKitAudioProcessor] write queue full — dropping oldest "
                    "buffered frame (ffmpeg falling behind real-time)"
                )
                self._write_backpressure_logged = True
        else:
            self._write_backpressure_logged = False
        self._last_write_at = time.monotonic()

    def _push_output_nowait(self, output_queue: asyncio.Queue, data: bytes) -> None:
        try:
            output_queue.put_nowait(data)
        except asyncio.QueueFull:
            try:
                output_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                output_queue.put_nowait(data)
            except asyncio.QueueFull:
                pass
            if not self._output_backpressure_logged:
                logger.warning(
                    "[LiveKitAudioProcessor] output queue full — dropping oldest "
                    "resampled chunk (consumer falling behind)"
                )
                self._output_backpressure_logged = True
        else:
            self._output_backpressure_logged = False
        self._last_output_at = time.monotonic()

    def _drain_output_nowait(self) -> bytes:
        queue = self._output_queue
        if queue is None:
            return b""
        chunks: list[bytes] = []
        while True:
            try:
                chunks.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return b"".join(chunks) if chunks else b""

    def _check_watchdog(self) -> None:
        """Detect a still-running-but-stuck ffmpeg process: writes are
        flowing but no output has arrived for a long time. Replaces the old
        per-call consecutive-timeout counter, which doesn't apply once reads
        happen on an independent background task."""
        if self._last_output_at is None or self._last_write_at is None:
            return
        now = time.monotonic()
        if now - self._last_output_at < _STUCK_RESET_SECONDS:
            return
        if now - self._last_write_at > _STUCK_RESET_SECONDS:
            # No recent writes either — this is just an idle gap (e.g. caller
            # silence), not a stuck process. Nothing to reset.
            return
        logger.warning(
            "[LiveKitAudioProcessor] no ffmpeg output for %.1fs despite active "
            "writes — process likely stuck, resetting",
            now - self._last_output_at,
        )
        self._mark_broken(now, self._ffmpeg_process)

    # ── Writer / reader background tasks ────────────────────────────────────

    async def _writer_loop(self, proc: asyncio.subprocess.Process, write_queue: asyncio.Queue) -> None:
        try:
            while True:
                pcm = await write_queue.get()
                if pcm is None:  # shutdown sentinel
                    return
                if proc.stdin is None:
                    return
                try:
                    proc.stdin.write(pcm)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError) as exc:
                    logger.warning("[LiveKitAudioProcessor] ffmpeg stdin write failed (pipe broken): %s", exc)
                    self._mark_broken(time.monotonic(), proc)
                    return
                except Exception as exc:
                    logger.warning("[LiveKitAudioProcessor] ffmpeg stdin write error: %s", exc)
                    self._mark_broken(time.monotonic(), proc)
                    return
        except asyncio.CancelledError:
            raise

    async def _reader_loop(self, proc: asyncio.subprocess.Process, output_queue: asyncio.Queue) -> None:
        try:
            while True:
                if proc.stdout is None:
                    return
                try:
                    data = await proc.stdout.read(65536)
                except (BrokenPipeError, ConnectionResetError, ValueError) as exc:
                    logger.warning("[LiveKitAudioProcessor] ffmpeg stdout read failed: %s", exc)
                    self._mark_broken(time.monotonic(), proc)
                    return
                except Exception as exc:
                    logger.warning("[LiveKitAudioProcessor] ffmpeg stdout read error: %s", exc)
                    self._mark_broken(time.monotonic(), proc)
                    return

                if not data:
                    # EOF — ffmpeg exited.
                    logger.warning(
                        "[LiveKitAudioProcessor] ffmpeg stdout closed (EOF) — "
                        "process ended, will restart on next frame"
                    )
                    self._mark_broken(time.monotonic(), proc)
                    return

                self._push_output_nowait(output_queue, data)
                # Defensive cooperative yield: a real subprocess pipe read
                # naturally cedes control to the event loop whenever no data
                # is immediately available, but guard against this loop ever
                # monopolizing the loop (e.g. if the transport ever returns a
                # burst of already-buffered reads back-to-back) so other
                # tasks — including this call's own writer task and the
                # caller's process_frame() drain — always get a turn.
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise

    def _mark_broken(self, now: float, stale_proc: asyncio.subprocess.Process | None) -> None:
        """Fast, non-blocking: called from the writer/reader tasks themselves
        (or the watchdog) when the process is known to be dead/stuck. Only
        clears the process reference so the next ``_ensure_running()`` call
        knows to (re)start — the writer/reader tasks calling this are already
        returning on their own, so actual task cancellation/join happens
        later in ``_stop_pipeline_locked`` rather than here (a task can't
        await-cancel itself)."""
        if self._ffmpeg_process is stale_proc:
            self._ffmpeg_process = None
            self._ffmpeg_input_rate = None
            self._ffmpeg_input_channels = None
        self._last_restart_failure = now
        if stale_proc is not None:
            asyncio.create_task(self._kill_stale_process(stale_proc))

    @staticmethod
    async def _kill_stale_process(proc: asyncio.subprocess.Process) -> None:
        try:
            if proc.returncode is None:
                proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except Exception as exc:
            logger.debug("[LiveKitAudioProcessor] failed to reap stale ffmpeg process: %s", exc)

    # ── Process + task lifecycle ─────────────────────────────────────────────

    async def _ensure_running(self, sample_rate: int, num_channels: int) -> bool:
        """Ensure an ffmpeg process (+ its writer/reader tasks) matching
        ``sample_rate``/``num_channels`` is running. Returns False if a
        (re)start attempt is currently backed off or failed."""

        def _matches() -> bool:
            return (
                self._ffmpeg_process is not None
                and self._ffmpeg_input_rate == sample_rate
                and self._ffmpeg_input_channels == num_channels
                and self._writer_task is not None and not self._writer_task.done()
                and self._reader_task is not None and not self._reader_task.done()
            )

        if _matches():
            return True

        async with self._restart_lock:
            # Re-check inside the lock — another call may have already
            # completed the (re)start while this one waited for the lock.
            if _matches():
                return True

            now = time.monotonic()
            if (
                self._last_restart_failure is not None
                and now - self._last_restart_failure < _RESTART_BACKOFF_SECONDS
            ):
                # Still within backoff from the last failed (re)start attempt —
                # drop this frame without retrying, so a sustained outage
                # doesn't re-attempt subprocess creation on every ~20ms frame.
                return False

            await self._stop_pipeline_locked()

            try:
                await self._restart_ffmpeg(sample_rate, num_channels)
            except Exception as exc:
                # A single frame's ffmpeg-(re)start failure (e.g. binary missing
                # from PATH in this environment, transient fork/exec error) must
                # not kill the whole LiveKitAudioSubscriber loop for the rest of
                # the call — that would silently end STT ingestion for the
                # entire conversation after only the frames processed so far.
                # Log once loudly, drop this frame, and let a later frame retry
                # (subject to the backoff above).
                if not self._ffmpeg_start_failed_logged:
                    logger.error(
                        "[LiveKitAudioProcessor] ffmpeg (re)start failed — "
                        "dropping frames until it recovers: %s",
                        exc,
                        exc_info=True,
                    )
                    self._ffmpeg_start_failed_logged = True
                self._last_restart_failure = now
                return False

            self._ffmpeg_start_failed_logged = False
            self._last_restart_failure = None

            proc = self._ffmpeg_process
            if proc is None or proc.stdin is None or proc.stdout is None:
                return False

            write_queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_WRITE_QUEUE_FRAMES)
            output_queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_OUTPUT_QUEUE_CHUNKS)
            self._write_queue = write_queue
            self._output_queue = output_queue
            self._writer_task = asyncio.create_task(self._writer_loop(proc, write_queue))
            self._reader_task = asyncio.create_task(self._reader_loop(proc, output_queue))
            self._last_output_at = time.monotonic()
            self._last_write_at = None
            return True

    async def _stop_pipeline_locked(self, *, wait_for_kill: bool = False) -> None:
        """Stop the writer/reader tasks and the ffmpeg process they're bound
        to. Must be called while holding ``self._restart_lock``.

        By default the final process reap runs as a detached background task
        so a mid-call restart (``_ensure_running``) never blocks frame
        processing on ``proc.wait()``. ``close()`` passes
        ``wait_for_kill=True`` instead, since callers of ``close()`` (e.g.
        ``LiveKitAudioSubscriber.stop()`` during call/app teardown) rely on
        the ffmpeg child actually being gone — not just scheduled for
        killing — by the time ``close()`` returns, to avoid leaking orphaned
        ffmpeg processes across many concurrent calls."""
        for task in (self._writer_task, self._reader_task):
            if task is not None and not task.done():
                task.cancel()
        # Also unblock a writer task parked on write_queue.get() so it can
        # observe cancellation promptly instead of waiting on a queue nobody
        # will feed again.
        if self._write_queue is not None:
            try:
                self._write_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        tasks = [t for t in (self._writer_task, self._reader_task) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._writer_task = None
        self._reader_task = None
        self._write_queue = None
        self._output_queue = None
        self._last_output_at = None
        self._last_write_at = None

        stale_proc = self._ffmpeg_process
        self._ffmpeg_process = None
        self._ffmpeg_input_rate = None
        self._ffmpeg_input_channels = None
        if stale_proc is None:
            return

        try:
            if stale_proc.stdin and not stale_proc.stdin.is_closing():
                stale_proc.stdin.close()
                await asyncio.wait_for(stale_proc.stdin.wait_closed(), timeout=1.0)
        except Exception as exc:
            logger.debug("[LiveKitAudioProcessor] failed to close stale ffmpeg stdin: %s", exc)
        if wait_for_kill:
            await self._kill_stale_process(stale_proc)
        else:
            asyncio.create_task(self._kill_stale_process(stale_proc))

    async def _restart_ffmpeg(self, sample_rate: int, num_channels: int) -> None:
        """Spawn a fresh ffmpeg process. Caller is responsible for stopping
        any prior process/tasks first (see ``_stop_pipeline_locked``)."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "ffmpeg not found in PATH. "
                "Install ffmpeg (apt-get install ffmpeg / brew install ffmpeg)."
            )

        cmd = [
            ffmpeg_bin,
            "-loglevel",
            "error",
            "-f",
            _FFMPEG_OUTPUT_FORMAT,
            "-ar",
            str(sample_rate),
            "-ac",
            str(num_channels),
            "-i",
            "pipe:0",
            "-ar",
            str(self._output_sample_rate),
            "-ac",
            str(self._output_channels),
            "-f",
            _FFMPEG_OUTPUT_FORMAT,
            # Force ffmpeg to flush its output AVIOContext after every write
            # instead of buffering up to its default ~32KB before releasing
            # anything to the pipe. Without this, real-time PCM fed in small
            # (~10-20ms) chunks gets released downstream in large multi-second
            # -delayed bursts instead of a steady near-real-time stream —
            # this starves STT endpointing (e.g. Deepgram) of the cadence it
            # needs to ever finalize a transcript. `-fflags nobuffer` /
            # `-flags low_delay` were evaluated and found to be no-ops here:
            # they govern demuxer/decoder latency, not applicable to this
            # raw s16le pipe-to-pipe (no container, no decoding) resample.
            "-flush_packets",
            "1",
            "pipe:1",
        ]
        self._ffmpeg_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._ffmpeg_input_rate = sample_rate
        self._ffmpeg_input_channels = num_channels
        logger.debug(
            "[LiveKitAudioProcessor] ffmpeg resampler pid=%s (%sHz/%sch→%sHz)",
            self._ffmpeg_process.pid,
            sample_rate,
            num_channels,
            self._output_sample_rate,
        )

    async def close(self) -> None:
        async with self._restart_lock:
            await self._stop_pipeline_locked(wait_for_kill=True)

    async def __aenter__(self) -> "LiveKitAudioProcessor":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


# Backward-compatible alias used by older imports/tests.
OpusToLinear16Transcoder = LiveKitAudioProcessor
