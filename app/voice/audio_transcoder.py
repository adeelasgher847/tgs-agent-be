"""
LiveKit / telephony audio → LINEAR16 PCM for Google STT.

LiveKit Python RTC ``AudioStream`` delivers decoded PCM (s16le), not raw Opus.
Use :class:`LiveKitAudioProcessor` per call: passthrough when already 16 kHz mono,
otherwise resample via ffmpeg child process (one per call).
"""
from __future__ import annotations

import asyncio
import shutil
from typing import Optional

from app.core.logger import logger

_FFMPEG_OUTPUT_FORMAT = "s16le"


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
        self._ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self._ffmpeg_input_rate: Optional[int] = None
        self._ffmpeg_input_channels: Optional[int] = None
        self._first_frame_logged = False

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
            return raw_bytes

        return await self._ffmpeg_convert(raw_bytes, sample_rate, num_channels)

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
            await self._restart_ffmpeg(sample_rate, num_channels)

        proc = self._ffmpeg_process
        if proc is None or proc.stdin is None or proc.stdout is None:
            return b""

        try:
            proc.stdin.write(pcm)
            await proc.stdin.drain()
            out = await asyncio.wait_for(proc.stdout.read(65536), timeout=0.5)
            return out or b""
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.warning("[LiveKitAudioProcessor] ffmpeg pipe broken: %s", exc)
            return b""
        except asyncio.TimeoutError:
            return b""
        except Exception as exc:
            logger.warning("[LiveKitAudioProcessor] ffmpeg convert error: %s", exc)
            return b""

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
        except Exception:
            pass
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
