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
    apply_volume_fade,
)

# After user_level (0..1 from background_volume slider), scale the bed for:
# - idle: agent hears office ambiance between speech
# - speak: ducked so voice + bed rarely clips (0.2/0.5 = −8 dB relative to idle)
BG_GAIN_IDLE = 0.5
BG_GAIN_SPEAK = 0.2
# Per 20ms frame smoothing toward the target (≈3–4 frames to settle, low zipper noise)
RAMP_SMOOTH = 0.32


@dataclass
class BackgroundAudioState:
    """Holds background audio configuration and runtime state."""

    enabled: bool = False
    user_level: float = 0.5
    ramp_gain: float = 0.0
    mulaw_bytes: Optional[bytes] = None
    length: int = 0
    offset: int = 0
    task: Optional[asyncio.Task] = None


class BackgroundAudioManager:
    """
    Manages background ambient audio for a single call.

    Uses embedded base64 audio decode on startup and streams looped 20ms frames.
    While the agent speaks, the background send loop is paused (single Twilio media
    stream); TTS frames mix the same looped bed at a ducked gain (see BG_GAIN_*).
    Gain changes are smoothed each 20ms frame to avoid clicks and level steps.
    """

    def __init__(self, websocket, get_stream_sid, is_speaking_flag):
        self.websocket = websocket
        self._get_stream_sid = get_stream_sid
        self._is_speaking_flag = is_speaking_flag
        self.state = BackgroundAudioState()

    def set_user_level(self, level: float) -> None:
        """Slider 0..1 (from background_volume 0–100)."""
        try:
            v = float(level)
        except (TypeError, ValueError):
            v = 0.5
        self.state.user_level = max(0.0, min(1.0, v))

    def _ramp_toward(self, target: float) -> None:
        s = self.state
        s.ramp_gain += (target - s.ramp_gain) * RAMP_SMOOTH
        if abs(s.ramp_gain - target) < 0.008:
            s.ramp_gain = target

    def mix_tts_frame(self, tts_20ms: bytes) -> bytes:
        """Mix one 20ms mu-law frame with ducked background; advances offset."""
        if not tts_20ms or not self.state.mulaw_bytes or self.state.length == 0:
            return tts_20ms
        frame = tts_20ms
        if len(frame) < MULAW_FRAME_BYTES:
            frame = frame + bytes([0xFF]) * (MULAW_FRAME_BYTES - len(frame))
        elif len(frame) > MULAW_FRAME_BYTES:
            frame = frame[:MULAW_FRAME_BYTES]
        target = self.state.user_level * BG_GAIN_SPEAK
        self._ramp_toward(target)
        g = self.state.ramp_gain
        mixed = mix_audio_with_background(
            tts_audio=frame,
            bg_audio=self.state.mulaw_bytes,
            bg_length=self.state.length,
            bg_offset=self.state.offset,
            volume_level=g,
        )
        self.state.offset = (self.state.offset + MULAW_FRAME_BYTES) % self.state.length
        return mixed

    async def load_from_base64_async(self) -> None:
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
            logger.warning(f"[BG] Failed to load background audio: {e}")

    async def _stream_loop(self) -> None:
        if not self.state.mulaw_bytes or self.state.length == 0:
            return

        send_interval = 0.02
        frame_bytes = MULAW_FRAME_BYTES
        first = True
        next_send = time.perf_counter()

        try:
            while True:
                stream_sid = self._get_stream_sid()
                if not stream_sid:
                    await asyncio.sleep(0.1)
                    continue

                if self._is_speaking_flag():
                    first = True
                    next_send = time.perf_counter()
                    await asyncio.sleep(0.01)
                    continue

                target_idle = self.state.user_level * BG_GAIN_IDLE
                self._ramp_toward(target_idle)
                eff = self.state.ramp_gain
                bg_chunk = get_background_audio_chunk(
                    self.state.offset,
                    frame_bytes,
                    self.state.mulaw_bytes,
                    self.state.length,
                )
                self.state.offset = (self.state.offset + frame_bytes) % self.state.length

                if eff <= 0.0:
                    out_chunk = bytes([0xFF]) * frame_bytes
                else:
                    out_chunk = apply_volume_fade(bg_chunk, eff)

                payload = base64.b64encode(out_chunk).decode("utf-8")
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
        try:
            if not self.state.enabled or not self.state.mulaw_bytes or self.state.task:
                return
            await asyncio.sleep(delay_seconds)
            if not self.state.enabled or not self.state.mulaw_bytes or self.state.task:
                return
            self.state.task = asyncio.create_task(self._stream_loop())
        except Exception as e:
            logger.error(f"[BG] Error starting background audio loop: {e}", exc_info=True)

    async def stop_loop(self) -> None:
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
            pass

    def mix_with_background(self, audio_bytes: bytes) -> bytes:
        if not self.state.mulaw_bytes or self.state.length == 0:
            return audio_bytes
        return b"".join(self.mix_tts_frame(f) for f in iter_mulaw_20ms_frames(audio_bytes))
