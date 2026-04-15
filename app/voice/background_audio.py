import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Optional

from app.core.logger import logger
from app.utils.audio_utils import (
    decode_background_audio_from_base64,
    get_background_audio_chunk,
    mix_audio_with_background,
    iter_mulaw_20ms_frames,
    MULAW_FRAME_BYTES,
)


@dataclass
class BackgroundAudioState:
    """Holds background audio configuration and runtime state."""

    enabled: bool = False
    volume: float = 0.6  # 60% volume (-4.4dB)
    mulaw_bytes: Optional[bytes] = None
    length: int = 0
    offset: int = 0
    task: Optional[asyncio.Task] = None


class BackgroundAudioManager:
    """
    Manages background / ambient audio for a single call.

    Responsibilities:
    - Loading/decoding MULAW background audio from base64 (async, non-blocking).
    - Starting/stopping a continuous background loop with 3s delayed start.
    - Optional mixing of main TTS audio with background frames.
    """

    def __init__(self, websocket, get_stream_sid, is_speaking_flag):
        self.websocket = websocket
        self._get_stream_sid = get_stream_sid
        self._is_speaking_flag = is_speaking_flag  # callable or lambda reading handler.is_speaking
        self.state = BackgroundAudioState()

    # -------- Loading / enabling -----------------------------------------

    async def load_from_base64_async(self) -> None:
        """
        Load background audio asynchronously to avoid blocking initialization.
        Uses the shared decode_background_audio_from_base64 helper.
        """
        try:
            loop = asyncio.get_event_loop()
            bg_audio_bytes, bg_audio_len = await loop.run_in_executor(
                None,
                decode_background_audio_from_base64,
            )

            if bg_audio_bytes and bg_audio_len > 0:
                self.state.mulaw_bytes = bg_audio_bytes
                self.state.length = bg_audio_len
                self.state.enabled = True
        except Exception as e:
            # Continue without background audio - call won't crash
            logger.warning(f"[BG] Failed to load background audio: {e}")

    async def _stream_loop(self) -> None:
        """
        Continuously stream background audio in a loop.
        Pauses when TTS is speaking to avoid conflicts.
        Uses pacing with drift correction for smooth playback.
        """
        if not self.state.mulaw_bytes or self.state.length == 0:
            return

        send_interval = 0.02  # 20ms per frame
        frame_bytes = MULAW_FRAME_BYTES
        first = True
        next_send = time.perf_counter()

        try:
            while True:
                stream_sid = self._get_stream_sid()
                if not stream_sid:
                    await asyncio.sleep(0.1)
                    continue

                # PAUSE background audio when TTS is speaking (no noise when AI speaks)
                if self._is_speaking_flag():
                    first = True
                    next_send = time.perf_counter()
                    await asyncio.sleep(0.01)
                    continue

                bg_chunk = get_background_audio_chunk(
                    self.state.offset,
                    frame_bytes,
                    self.state.mulaw_bytes,
                    self.state.length,
                )
                self.state.offset = (self.state.offset + frame_bytes) % self.state.length

                payload = base64.b64encode(bg_chunk).decode("utf-8")
                await self.websocket.send_json(
                    {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload},
                    }
                )

                if not first:
                    next_send += send_interval
                    now = time.perf_counter()
                    sleep_dur = next_send - now
                    if sleep_dur > 0:
                        await asyncio.sleep(sleep_dur)
                    elif sleep_dur < -0.03:
                        next_send = time.perf_counter()
                else:
                    first = False
                    next_send = time.perf_counter() + send_interval
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[BG] Error in background audio loop: {e}")

    async def start_loop_if_enabled(self, delay_seconds: float = 3.0) -> None:
        """
        Start the background loop after an optional delay, if enabled and not already running.
        Mirrors the existing 3-second post-pickup delay.
        """
        try:
            if not self.state.enabled or not self.state.mulaw_bytes or self.state.task:
                return

            # Delay before starting to give the call time to establish
            await asyncio.sleep(delay_seconds)

            if not self.state.enabled or not self.state.mulaw_bytes or self.state.task:
                return

            self.state.task = asyncio.create_task(self._stream_loop())
        except Exception as e:
            logger.error(f"[BG] Error starting background audio loop: {e}", exc_info=True)

    async def stop_loop(self) -> None:
        """
        Stop the background loop if running.
        """
        try:
            if self.state.task:
                self.state.task.cancel()
                try:
                    await self.state.task
                except asyncio.CancelledError:
                    pass
                finally:
                    self.state.task = None
        except Exception:
            # Never raise on shutdown
            pass

    # -------- Mixing with main TTS ---------------------------------------

    def mix_with_background(self, audio_bytes: bytes) -> bytes:
        """
        Mix main TTS audio with the configured background audio in-frame.
        Uses iter_mulaw_20ms_frames + mix_audio_with_background.
        """
        if not self.state.mulaw_bytes or self.state.length == 0:
            return audio_bytes

        mixed_frames = []
        for frame in iter_mulaw_20ms_frames(audio_bytes):
            mixed = mix_audio_with_background(
                tts_audio=frame,
                bg_audio=self.state.mulaw_bytes,
                bg_length=self.state.length,
                bg_offset=self.state.offset,
                volume_level=self.state.volume,
            )
            self.state.offset = (self.state.offset + len(frame)) % self.state.length
            mixed_frames.append(mixed)

        return b"".join(mixed_frames)

