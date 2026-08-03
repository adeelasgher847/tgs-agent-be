"""
LiveKit / telephony audio → LINEAR16 PCM for Google STT.

LiveKit Python RTC ``AudioStream`` delivers decoded PCM (s16le), not raw Opus.
Use :class:`LiveKitAudioProcessor` per call: passthrough when already 16 kHz mono,
otherwise resample via ffmpeg child process (one per call).
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
# A single stdout-read timeout is normal/expected — a freshly (re)started
# ffmpeg process needs a moment to initialize and buffer enough input before
# it flushes its first output block, especially under container CPU
# contention, and each caller-audio frame is only ~10-20ms of PCM. Only treat
# it as "the process is actually stuck" (and reset it) after this many
# *consecutive* timeouts, so we don't kill-and-respawn ffmpeg before it ever
# gets a chance to reach steady state.
_MAX_CONSECUTIVE_READ_TIMEOUTS = 15


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
        self._consecutive_read_timeouts = 0

    async def process_frame(
        self,
        raw_bytes: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> bytes:
        """Return LINEAR16 mono PCM at ``output_sample_rate`` Hz."""
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

    async def _ffmpeg_convert(
        self,
        pcm: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> bytes:
        """Resample / remix via ffmpeg child process (lazy-started, per call)."""
        if (
            self._ffmpeg_process is None
            or self._ffmpeg_input_rate != sample_rate
            or self._ffmpeg_input_channels != num_channels
        ):
            now = time.monotonic()
            if (
                self._last_restart_failure is not None
                and now - self._last_restart_failure < _RESTART_BACKOFF_SECONDS
            ):
                # Still within backoff from the last failed (re)start attempt —
                # drop this frame without retrying, so a sustained outage
                # doesn't re-attempt subprocess creation on every ~20ms frame.
                return b""
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
                return b""
            else:
                self._ffmpeg_start_failed_logged = False
                self._last_restart_failure = None

        proc = self._ffmpeg_process
        if proc is None or proc.stdin is None or proc.stdout is None:
            return b""

        try:
            proc.stdin.write(pcm)
            await proc.stdin.drain()
            out = await asyncio.wait_for(proc.stdout.read(65536), timeout=0.5)
            self._consecutive_read_timeouts = 0
            return out or b""
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.warning("[LiveKitAudioProcessor] ffmpeg pipe broken: %s", exc)
            self._consecutive_read_timeouts = 0
            await self._reset_after_failure()
            return b""
        except asyncio.TimeoutError:
            # A single (or occasional) timeout is normal — the process just
            # hasn't buffered enough input yet to flush output for this
            # frame; drop this frame only, keep the same process running so
            # it can reach steady state. Only reset once enough consecutive
            # timeouts have piled up to indicate the process is genuinely
            # stuck rather than merely warming up / catching up.
            self._consecutive_read_timeouts += 1
            if self._consecutive_read_timeouts >= _MAX_CONSECUTIVE_READ_TIMEOUTS:
                logger.warning(
                    "[LiveKitAudioProcessor] ffmpeg read timed out %d times in a "
                    "row — process likely stuck, resetting",
                    self._consecutive_read_timeouts,
                )
                self._consecutive_read_timeouts = 0
                await self._reset_after_failure()
            return b""
        except Exception as exc:
            logger.warning("[LiveKitAudioProcessor] ffmpeg convert error: %s", exc)
            self._consecutive_read_timeouts = 0
            await self._reset_after_failure()
            return b""

    async def _reset_after_failure(self) -> None:
        """Clear the (dead/stuck) ffmpeg process so the next frame re-enters
        the restart path instead of repeatedly hitting the same broken proc
        for the rest of the call, and reap the old process in the background
        so a merely-stuck (not dead) proc doesn't leak as an orphan."""
        stale_proc = self._ffmpeg_process
        self._ffmpeg_process = None
        self._ffmpeg_input_rate = None
        self._ffmpeg_input_channels = None
        self._last_restart_failure = time.monotonic()
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

    async def _restart_ffmpeg(self, sample_rate: int, num_channels: int) -> None:
        await self.close()
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
        if self._ffmpeg_process is None:
            return
        try:
            if self._ffmpeg_process.stdin and not self._ffmpeg_process.stdin.is_closing():
                self._ffmpeg_process.stdin.close()
                await self._ffmpeg_process.stdin.wait_closed()
        except Exception as exc:
            logger.debug("Failed to close ffmpeg stdin: %s", exc)
        try:
            await asyncio.wait_for(self._ffmpeg_process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            self._ffmpeg_process.kill()
        self._ffmpeg_process = None
        self._ffmpeg_input_rate = None
        self._ffmpeg_input_channels = None

    async def __aenter__(self) -> "LiveKitAudioProcessor":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


# Backward-compatible alias used by older imports/tests.
OpusToLinear16Transcoder = LiveKitAudioProcessor
