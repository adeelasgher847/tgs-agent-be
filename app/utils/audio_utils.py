"""
Audio processing utilities for bidirectional streaming.
Handles MULAW audio conversion, crossfading, and streaming.
"""

import base64
import asyncio
import time
import sys
import math
import subprocess
import tempfile
import os
from typing import Optional, Iterable

from app.core.logger import logger
from app.utils.audio_constants import BACKGROUND_AUDIO_BASE64

# Real-time TTS MULAW streaming constants
MULAW_SAMPLE_RATE_HZ = 8000  # Twilio-friendly
BYTES_PER_SECOND = MULAW_SAMPLE_RATE_HZ  # 8-bit mu-law => 1 byte per sample
CHUNK_DURATION_SEC = 0.02  # 20ms
MULAW_FRAME_BYTES = int(BYTES_PER_SECOND * CHUNK_DURATION_SEC)  # 160 bytes

ULAW_BIAS = 0x84
ULAW_CLIP = 32635

# Cache for decoded background audio
_background_audio_mulaw_cache = None
_background_audio_length_cache = 0


def decode_background_audio_from_base64() -> tuple[bytes, int]:
    """Decode embedded base64 MP3 and convert to MULAW 8kHz."""
    global _background_audio_mulaw_cache, _background_audio_length_cache

    if _background_audio_mulaw_cache is not None:
        return _background_audio_mulaw_cache, _background_audio_length_cache

    if not BACKGROUND_AUDIO_BASE64:
        logger.warning("No embedded background audio configured.")
        return b"", 0

    mp3_path = None
    try:
        mp3_bytes = base64.b64decode(BACKGROUND_AUDIO_BASE64)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
            mp3_file.write(mp3_bytes)
            mp3_path = mp3_file.name

        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                mp3_path,
                "-ar",
                "8000",
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
            check=True,
            input=None,
        )

        pcm_data = result.stdout
        if not pcm_data:
            return b"", 0

        linear_samples = []
        for i in range(0, len(pcm_data), 2):
            if i + 1 < len(pcm_data):
                linear_samples.append(int.from_bytes(pcm_data[i : i + 2], byteorder="little", signed=True))

        mulaw_bytes = bytes(linear_to_ulaw_sample(sample) for sample in linear_samples)
        _background_audio_mulaw_cache = mulaw_bytes
        _background_audio_length_cache = len(mulaw_bytes)
        return mulaw_bytes, len(mulaw_bytes)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        logger.error(f"FFmpeg conversion failed: {error_msg}")
        return b"", 0
    except FileNotFoundError:
        logger.error("FFmpeg not found; embedded background audio disabled.")
        return b"", 0
    except Exception as e:
        logger.error(f"Failed to decode embedded background audio: {e}")
        return b"", 0
    finally:
        if mp3_path and os.path.exists(mp3_path):
            try:
                os.unlink(mp3_path)
            except Exception:
                pass


def get_background_audio_chunk(offset: int, length: int, bg_audio: bytes, bg_length: int) -> bytes:
    if not bg_audio or bg_length == 0:
        return bytes([0xFF]) * length
    chunk = bytearray()
    for i in range(length):
        idx = (offset + i) % bg_length
        chunk.append(bg_audio[idx])
    return bytes(chunk)


def mix_audio_with_background(
    tts_audio: bytes,
    bg_audio: bytes,
    bg_length: int,
    bg_offset: int,
    volume_level: float = 0.3,
) -> bytes:
    if not bg_audio or bg_length == 0 or not tts_audio:
        return tts_audio
    try:
        tts_linear = [ulaw_to_linear_sample(b) for b in tts_audio]
        bg_chunk = get_background_audio_chunk(bg_offset, len(tts_audio), bg_audio, bg_length)
        bg_linear = [ulaw_to_linear_sample(b) for b in bg_chunk]
        mixed = []
        for i, sample in enumerate(tts_linear):
            m = sample + int(bg_linear[i] * volume_level)
            mixed.append(max(-32768, min(32767, m)))
        return bytes(linear_to_ulaw_sample(s) for s in mixed)
    except Exception as e:
        logger.warning(f"Background mix failed; using dry voice. err={e}")
        return tts_audio


def apply_volume_fade(audio_bytes: bytes, volume: float) -> bytes:
    """
    Apply volume level to MULAW audio bytes.
    
    Args:
        audio_bytes: MULAW audio bytes
        volume: Volume level (0.0-1.0, where 1.0 = 100%)
    
    Returns:
        Volume-adjusted MULAW audio bytes
    """
    if not audio_bytes or len(audio_bytes) == 0:
        return audio_bytes
    
    if volume <= 0.0:
        # Silence (return mu-law silence)
        return bytes([0xFF]) * len(audio_bytes)
    
    if volume >= 1.0:
        # No change
        return audio_bytes
    
    try:
        linear_samples = [ulaw_to_linear_sample(b) for b in audio_bytes]
        adjusted_samples = [max(-32768, min(32767, int(s * volume))) for s in linear_samples]
        return bytes([linear_to_ulaw_sample(s) for s in adjusted_samples])
    except Exception as e:
        logger.warning(f"⚠️ Volume adjustment failed: {e}")
        return audio_bytes


def apply_micro_fade_in(audio_bytes: bytes, duration_ms: float = 25.0) -> bytes:
    """
    Apply a micro linear fade-in to the start of MULAW audio to eliminate clicks/pops.
    
    Args:
        audio_bytes: MULAW audio bytes
        duration_ms: Duration of fade in milliseconds (default 25ms for smoother start)
        
    Returns:
        Audio bytes with micro fade-in applied
    """
    if not audio_bytes or len(audio_bytes) == 0:
        return audio_bytes
        
    try:
        # Calculate samples to fade (8kHz sample rate)
        num_fade_samples = int((duration_ms / 1000.0) * MULAW_SAMPLE_RATE_HZ)
        num_fade_samples = min(num_fade_samples, len(audio_bytes))
        
        if num_fade_samples <= 0:
            return audio_bytes
            
        # Convert only the part to fade to linear
        fade_part = audio_bytes[:num_fade_samples]
        remaining_part = audio_bytes[num_fade_samples:]
        
        linear_samples = [ulaw_to_linear_sample(b) for b in fade_part]
        
        faded_samples = []
        for i, sample in enumerate(linear_samples):
            # Linear ramp from 0.0 to 1.0
            volume = i / num_fade_samples
            faded_samples.append(int(sample * volume))
            
        faded_part_mulaw = bytes([linear_to_ulaw_sample(s) for s in faded_samples])
        
        return faded_part_mulaw + remaining_part
        
    except Exception as e:
        logger.warning(f"⚠️ Micro fade-in failed: {e}")
        return audio_bytes


def ulaw_to_linear_sample(ulaw_byte: int) -> int:
    """
    Convert a single mu-law encoded byte to 16-bit linear PCM.
    """
    ulaw_byte = (~ulaw_byte) & 0xFF
    sign = ulaw_byte & 0x80
    exponent = (ulaw_byte >> 4) & 0x07
    mantissa = ulaw_byte & 0x0F
    sample = ((mantissa << 3) + ULAW_BIAS) << exponent
    return -sample if sign else sample


def linear_to_ulaw_sample(sample: int) -> int:
    """
    Convert a 16-bit linear PCM sample to mu-law encoded byte.
    """
    if sample > ULAW_CLIP:
        sample = ULAW_CLIP
    elif sample < -ULAW_CLIP:
        sample = -ULAW_CLIP

    sign = 0x80 if sample < 0 else 0
    if sign:
        sample = -sample

    sample += ULAW_BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        mask >>= 1
        exponent -= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw_byte


def linear_samples_to_ulaw_bytes(samples: Iterable[int]) -> bytes:
    """Encode an iterable of 16-bit linear PCM samples to mu-law bytes."""
    return bytes(linear_to_ulaw_sample(int(sample)) for sample in samples)


def strip_wav_header(audio_bytes: bytes) -> bytes:
    """
    Return the PCM payload if `audio_bytes` contains a RIFF/WAVE header.

    If the payload doesn't look like WAV data, return the bytes unchanged.
    """
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return audio_bytes
    idx = audio_bytes.find(b"data")
    if idx == -1 or idx + 8 > len(audio_bytes):
        return audio_bytes
    data_len = int.from_bytes(audio_bytes[idx + 4:idx + 8], "little", signed=False)
    start = idx + 8
    end = min(len(audio_bytes), start + data_len) if data_len > 0 else len(audio_bytes)
    return audio_bytes[start:end]


def pcm16le_bytes_to_linear_samples(audio_bytes: bytes) -> list[int]:
    """Decode raw little-endian 16-bit PCM bytes into signed linear samples."""
    payload = strip_wav_header(audio_bytes)
    usable = len(payload) - (len(payload) % 2)
    out: list[int] = []
    for i in range(0, usable, 2):
        out.append(int.from_bytes(payload[i:i + 2], "little", signed=True))
    return out


def downsample_linear_samples(samples: list[int], src_rate_hz: int, dst_rate_hz: int) -> list[int]:
    """
    Downsample linear PCM samples using simple box averaging.

    This is intentionally simple and fast. For 16k -> 8k (our ElevenLabs
    background use-case), averaging each adjacent pair removes enough high
    frequency energy to avoid harsh aliasing on phone calls.
    """
    if not samples or src_rate_hz == dst_rate_hz:
        return list(samples)
    if src_rate_hz <= 0 or dst_rate_hz <= 0 or src_rate_hz % dst_rate_hz != 0:
        raise ValueError(f"Unsupported resample ratio: {src_rate_hz} -> {dst_rate_hz}")
    factor = src_rate_hz // dst_rate_hz
    usable = len(samples) - (len(samples) % factor)
    out: list[int] = []
    for i in range(0, usable, factor):
        chunk = samples[i:i + factor]
        out.append(int(sum(chunk) / factor))
    return out


class PCM16KStreamDownsampler:
    """
    Incrementally convert PCM16 LE 16kHz chunks to linear PCM 8kHz samples.

    Handles:
    - partial byte pairs across HTTP chunks
    - a one-time WAV header at the start of the stream
    - 16k -> 8k box-average downsampling
    """

    __slots__ = ("_buf", "_header_done")

    def __init__(self):
        self._buf = bytearray()
        self._header_done = False

    def _strip_header_if_ready(self) -> bool:
        if self._header_done:
            return True
        if len(self._buf) < 12:
            return False
        if self._buf[:4] != b"RIFF" or self._buf[8:12] != b"WAVE":
            self._header_done = True
            return True
        idx = self._buf.find(b"data")
        if idx == -1 or idx + 8 > len(self._buf):
            return False
        del self._buf[:idx + 8]
        self._header_done = True
        return True

    def feed(self, chunk: bytes) -> list[int]:
        if chunk:
            self._buf.extend(chunk)
        if not self._strip_header_if_ready():
            return []
        usable = len(self._buf) - (len(self._buf) % 4)  # two int16 samples -> one 8k sample
        out: list[int] = []
        for i in range(0, usable, 4):
            s1 = int.from_bytes(self._buf[i:i + 2], "little", signed=True)
            s2 = int.from_bytes(self._buf[i + 2:i + 4], "little", signed=True)
            out.append((s1 + s2) // 2)
        if usable:
            del self._buf[:usable]
        return out

    def flush(self) -> list[int]:
        if not self._strip_header_if_ready():
            return []
        usable = len(self._buf) - (len(self._buf) % 4)
        out: list[int] = []
        for i in range(0, usable, 4):
            s1 = int.from_bytes(self._buf[i:i + 2], "little", signed=True)
            s2 = int.from_bytes(self._buf[i + 2:i + 4], "little", signed=True)
            out.append((s1 + s2) // 2)
        self._buf.clear()
        return out


def iter_mulaw_20ms_frames(audio_bytes: bytes) -> Iterable[bytes]:
    """
    Yield 20ms mu-law frames (160 bytes at 8kHz).
    Pads the final frame with mu-law silence (0xFF) if needed.
    """
    if not audio_bytes:
        return
    total_len = len(audio_bytes)
    full_frames = total_len // MULAW_FRAME_BYTES
    remainder = total_len % MULAW_FRAME_BYTES

    offset = 0
    for _ in range(full_frames):
        yield audio_bytes[offset:offset + MULAW_FRAME_BYTES]
        offset += MULAW_FRAME_BYTES

    if remainder:
        last = bytearray(audio_bytes[offset:])
        last.extend(b'\xFF' * (MULAW_FRAME_BYTES - remainder))  # mu-law silence pad
        yield bytes(last)


async def stream_mulaw_bytes_over_twilio(
    websocket,
    stream_sid: str,
    audio_bytes: bytes,
    pace_20ms: bool = True,
    cancel: Optional[asyncio.Event] = None,
    prime_frames: int = 0,
):
    """
    Send mu-law audio to Twilio as 20ms 'media' frames.
    - Sends first frame immediately (early playback).
    - Optionally pace subsequent frames by ~20ms to match realtime.
    """
    first = True
    send_interval = 0.02  # 20ms
    next_send = time.perf_counter()
    # Optional: prime Twilio jitter buffer with mu-law silence frames
    if prime_frames and prime_frames > 0:
        silent_frame = bytes([0xFF]) * MULAW_FRAME_BYTES
        for _ in range(prime_frames):
            if cancel and cancel.is_set():
                break
            payload = base64.b64encode(silent_frame).decode("utf-8")
            try:
                await websocket.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload}
                })
            except RuntimeError:
                # WebSocket already closed (hangup). Stop sending immediately.
                return
            # do not pace priming frames to quickly fill buffer
    for frame in iter_mulaw_20ms_frames(audio_bytes):
        if cancel and cancel.is_set():
            break
        payload = base64.b64encode(frame).decode("utf-8")
        try:
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
        except RuntimeError:
            return
        if not pace_20ms:
            continue
        if first:
            first = False
            next_send = time.perf_counter() + send_interval
            continue
        # Precise pacing with drift correction
        next_send += send_interval
        now = time.perf_counter()
        sleep_dur = next_send - now
        if sleep_dur > 0:
            await asyncio.sleep(sleep_dur)
        elif sleep_dur < -0.03:
            # We're late by >30ms; reset schedule to avoid cumulative jitter
            next_send = time.perf_counter()


def crossfade_mulaw_segments(prev_tail: bytes, next_head: bytes, overlap_bytes: int = None) -> bytes:
    """
    Crossfade two adjacent mu-law segments to eliminate clicks at boundaries.
    Python 3.13+ compatible (no audioop dependency).
    
    Args:
        prev_tail: Last portion of previous chunk
        next_head: Complete next chunk
        overlap_bytes: Overlap size (default: 160 bytes = 20ms at 8kHz)
        
    Returns:
        Blended audio bytes
    """
    if not prev_tail and not next_head:
        return b""

    if overlap_bytes is None:
        overlap_bytes = MULAW_FRAME_BYTES  # default 20ms

    if not prev_tail or not next_head:
        return (prev_tail or b"") + (next_head or b"")

    overlap = min(len(prev_tail), len(next_head), overlap_bytes)
    if overlap <= 0:
        return (prev_tail or b"") + (next_head or b"")

    try:
        prev_overlap = prev_tail[-overlap:]
        next_overlap = next_head[:overlap]

        prev_lin = [ulaw_to_linear_sample(b) for b in prev_overlap]
        next_lin = [ulaw_to_linear_sample(b) for b in next_overlap]

        n = min(len(prev_lin), len(next_lin))
        if n == 0:
            return (prev_tail or b"") + (next_head or b"")
        mixed = []

        # S-curve crossfade for equal-loudness (no volume dip)
        for i in range(n):
            progress = i / n
            fade_out = math.cos(progress * math.pi / 2)  # 1.0 → 0.0 (smooth curve)
            fade_in = math.sin(progress * math.pi / 2)   # 0.0 → 1.0 (smooth curve)
            mixed_sample = int(prev_lin[i] * fade_out + next_lin[i] * fade_in)
            mixed_sample = max(-32768, min(32767, mixed_sample))
            mixed.append(linear_to_ulaw_sample(mixed_sample))

        return prev_tail[:-overlap] + bytes(mixed) + next_head[overlap:]

    except Exception as e:
        logger.warning(f"⚠️ Crossfade failed, using direct join: {e}")
        return (prev_tail or b"") + (next_head or b"")


def build_crossfade_bridge(prev_tail: bytes, next_head: bytes, overlap_bytes: int = None) -> bytes:
    """
    Build a dedicated overlap bridge between two mu-law segments.
    The bridge contains the blended overlap region only, intended to be sent
    between consecutive chunks to avoid audible clicks ("tak-tak").
    """
    if not prev_tail or not next_head:
        return b""

    if overlap_bytes is None:
        overlap_bytes = 400  # 50ms at 8kHz for smooth transitions

    overlap = min(len(prev_tail), len(next_head), max(1, overlap_bytes))
    if overlap <= 0:
        return b""

    prev_overlap = prev_tail[-overlap:]
    next_overlap = next_head[:overlap]

    prev_lin = [ulaw_to_linear_sample(b) for b in prev_overlap]
    next_lin = [ulaw_to_linear_sample(b) for b in next_overlap]

    n = min(len(prev_lin), len(next_lin))
    if n == 0:
        return b""

    # S-curve crossfade for equal-loudness (no volume dip)
    bridge_samples = []
    for i in range(n):
        progress = i / n
        fade_out = math.cos(progress * math.pi / 2)  # Smooth S-curve
        fade_in = math.sin(progress * math.pi / 2)   # Smooth S-curve
        mixed_sample = int(prev_lin[i] * fade_out + next_lin[i] * fade_in)
        mixed_sample = max(-32768, min(32767, mixed_sample))
        bridge_samples.append(linear_to_ulaw_sample(mixed_sample))

    return bytes(bridge_samples)


def add_ambient_noise_to_mulaw(audio_bytes: bytes, noise_level: float = 0.02) -> bytes:
    """
    Add a subtle synthetic office-like ambience bed to mu-law audio.
    Kept lightweight for telephony latency constraints.
    """
    import random

    if not audio_bytes:
        return audio_bytes
    try:
        linear_audio = [ulaw_to_linear_sample(b) for b in audio_bytes]
        num_samples = len(linear_audio)

        sample_rate = 8000.0
        hvac_freq = 120.0
        hvac_phase_step = 2 * math.pi * hvac_freq / sample_rate
        hvac_phase = random.uniform(0, 2 * math.pi)
        pink_state = [0.0] * 7

        noise_samples = []
        for _ in range(num_samples):
            hvac_phase += hvac_phase_step
            if hvac_phase > 2 * math.pi:
                hvac_phase -= 2 * math.pi
            hvac = math.sin(hvac_phase) * 0.6

            white = random.uniform(-1.0, 1.0)
            pink_state[0] = 0.99886 * pink_state[0] + white * 0.0555179
            pink_state[1] = 0.99332 * pink_state[1] + white * 0.0750759
            pink_state[2] = 0.96900 * pink_state[2] + white * 0.1538520
            pink_state[3] = 0.86650 * pink_state[3] + white * 0.3104856
            pink_state[4] = 0.55000 * pink_state[4] + white * 0.5329522
            pink_state[5] = -0.7616 * pink_state[5] - white * 0.0168980
            pink = sum(pink_state) * 0.1

            total_noise = hvac + (pink * 0.5)
            noise_scaled = int(total_noise * 32767 * noise_level)
            noise_samples.append(max(-32768, min(32767, noise_scaled)))

        mixed_linear = []
        for i in range(num_samples):
            mixed = linear_audio[i] + noise_samples[i]
            mixed_linear.append(max(-32768, min(32767, mixed)))

        return bytes(linear_to_ulaw_sample(sample) for sample in mixed_linear)
    except Exception as e:
        logger.warning(f"Ambient noise mix failed, returning dry audio: {e}")
        return audio_bytes

