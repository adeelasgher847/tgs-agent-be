"""
Audio processing utilities for bidirectional streaming.
Handles MULAW audio conversion, mixing, crossfading, and streaming.
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
    """
    Decode base64 MP3 and convert to MULAW format using FFmpeg.
    Returns (mulaw_bytes, length_in_bytes).
    Cached after first load.
    
    Uses FFmpeg subprocess for reliable MP3 to PCM conversion (Python 3.13 compatible).
    """
    global _background_audio_mulaw_cache, _background_audio_length_cache
    
    if _background_audio_mulaw_cache is not None:
        return _background_audio_mulaw_cache, _background_audio_length_cache
    
    if not BACKGROUND_AUDIO_BASE64:
        print("⚠️ No background audio configured (BACKGROUND_AUDIO_BASE64 is empty)")
        return b'', 0
    
    try:
        import subprocess
        import tempfile
        import os
    except ImportError as import_error:
        print(f"❌ Failed to import required modules: {import_error}")
        sys.stdout.flush()
        return b'', 0
    
    mp3_path = None
    try:
        # Decode base64 MP3
        mp3_bytes = base64.b64decode(BACKGROUND_AUDIO_BASE64)
        
        # Create temporary MP3 file for FFmpeg
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
            mp3_file.write(mp3_bytes)
            mp3_path = mp3_file.name
        
        # Convert MP3 to raw PCM using FFmpeg
        result = subprocess.run(
            [
                'ffmpeg',
                '-nostdin',
                '-loglevel', 'error',
                '-i', mp3_path,
                '-ar', '8000',      # Sample rate 8kHz
                '-ac', '1',          # Mono channel
                '-f', 's16le',       # 16-bit little-endian PCM
                '-'                   # Output to stdout
            ],
            capture_output=True,
            check=True,
            input=None
        )
        
        pcm_data = result.stdout
        
        if not pcm_data or len(pcm_data) == 0:
            print("⚠️ FFmpeg conversion produced empty output")
            sys.stdout.flush()
            return b'', 0
        
        # Convert linear PCM samples to MULAW
        linear_samples = []
        for i in range(0, len(pcm_data), 2):
            if i + 1 < len(pcm_data):
                sample = int.from_bytes(pcm_data[i:i+2], byteorder='little', signed=True)
                linear_samples.append(sample)
        
        if not linear_samples:
            print("⚠️ No audio samples extracted from PCM data")
            sys.stdout.flush()
            return b'', 0
        
        mulaw_bytes = bytes([linear_to_ulaw_sample(sample) for sample in linear_samples])
        
        _background_audio_mulaw_cache = mulaw_bytes
        _background_audio_length_cache = len(mulaw_bytes)
        
        print(f"✅ Decoded background audio using FFmpeg: {len(mulaw_bytes)} bytes MULAW ({len(mulaw_bytes)/8000:.2f}s)")
        sys.stdout.flush()
        return mulaw_bytes, len(mulaw_bytes)
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        print(f"❌ FFmpeg conversion failed: {error_msg}")
        print(f"⚠️ Make sure FFmpeg is installed: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        sys.stdout.flush()
        return b'', 0
    except FileNotFoundError:
        print(f"❌ FFmpeg not found. Please install FFmpeg: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        sys.stdout.flush()
        return b'', 0
    except Exception as e:
        print(f"❌ Failed to decode background audio: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return b'', 0
    finally:
        # Clean up temporary MP3 file
        if mp3_path and os.path.exists(mp3_path):
            try:
                os.unlink(mp3_path)
            except Exception:
                pass  # Ignore cleanup errors


def get_background_audio_chunk(offset: int, length: int, bg_audio: bytes, bg_length: int) -> bytes:
    """
    Get a chunk of background audio from the looped buffer.
    
    Args:
        offset: Starting byte offset in the loop
        length: Number of bytes to get
        bg_audio: Background audio MULAW bytes
        bg_length: Length of background audio
        
    Returns:
        MULAW audio chunk (looped if needed)
    """
    if not bg_audio or bg_length == 0:
        return bytes([0xFF]) * length  # Silence if no background
    
    chunk = bytearray()
    for i in range(length):
        index = (offset + i) % bg_length
        chunk.append(bg_audio[index])
    
    return bytes(chunk)


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
        # Convert to linear, apply volume, convert back
        linear_samples = [ulaw_to_linear_sample(b) for b in audio_bytes]
        faded_linear = [int(sample * volume) for sample in linear_samples]
        # Clamp to valid range
        faded_linear = [max(-32768, min(32767, sample)) for sample in faded_linear]
        faded_mulaw = bytes([linear_to_ulaw_sample(sample) for sample in faded_linear])
        return faded_mulaw
    except Exception as e:
        print(f"⚠️ Volume fade failed: {e}, using original audio")
        return audio_bytes


def mix_audio_with_background(tts_audio: bytes, bg_audio: bytes, bg_length: int, bg_offset: int, volume_level: float = 0.3) -> bytes:
    """
    Mix TTS audio with background audio at current offset.
    
    Args:
        tts_audio: TTS MULAW audio bytes
        bg_audio: Background audio MULAW bytes
        bg_length: Length of background audio
        bg_offset: Current offset in background loop
        volume_level: Background volume (0.0-1.0, default 0.3 = -10dB)
        
    Returns:
        Mixed MULAW audio
    """
    if not bg_audio or bg_length == 0:
        return tts_audio
    
    if not tts_audio or len(tts_audio) == 0:
        return tts_audio
    
    try:
        tts_linear = [ulaw_to_linear_sample(b) for b in tts_audio]
        num_samples = len(tts_linear)
        
        bg_chunk = get_background_audio_chunk(bg_offset, num_samples, bg_audio, bg_length)
        bg_linear = [ulaw_to_linear_sample(b) for b in bg_chunk]
        
        mixed_linear = []
        for i in range(num_samples):
            mixed = tts_linear[i] + int(bg_linear[i] * volume_level)
            mixed = max(-32768, min(32767, mixed))
            mixed_linear.append(mixed)
        
        mixed_mulaw = bytes([linear_to_ulaw_sample(sample) for sample in mixed_linear])
        return mixed_mulaw
        
    except Exception as e:
        print(f"⚠️ Audio mixing failed: {e}, using clean TTS")
        sys.stdout.flush()
        return tts_audio


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
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
            # do not pace priming frames to quickly fill buffer
    for frame in iter_mulaw_20ms_frames(audio_bytes):
        if cancel and cancel.is_set():
            break
        payload = base64.b64encode(frame).decode("utf-8")
        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        })
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
        print(f"⚠️ Crossfade failed, using direct join: {e}")
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
    Add realistic office environment noise with minimal latency.
    Uses optimized layered approach: HVAC rumble + keyboard typing + conversations.
    Python 3.13+ compatible (no audioop dependency).
    
    Args:
        audio_bytes: MULAW audio bytes (8kHz)
        noise_level: Noise volume (0.01-0.05 recommended, default 0.02 = -34dB)
        
    Returns:
        MULAW audio with realistic office background noise mixed in
    """
    import random
    
    if not audio_bytes or len(audio_bytes) == 0:
        return audio_bytes
    
    try:
        # Convert MULAW to linear
        linear_audio = [ulaw_to_linear_sample(b) for b in audio_bytes]
        num_samples = len(linear_audio)
        
        # Pre-calculate constants for speed
        sample_rate = 8000.0
        hvac_freq = 120.0  # Fixed HVAC frequency (faster than random)
        hvac_phase_step = 2 * math.pi * hvac_freq / sample_rate
        
        # Initialize states (reused across samples)
        hvac_phase = random.uniform(0, 2 * math.pi)
        pink_state = [0.0] * 7
        
        # Keyboard typing state (intermittent)
        keyboard_counter = 0
        keyboard_active = False
        keyboard_phase = 0
        
        noise_samples = []
        
        for i in range(num_samples):
            total_noise = 0
            
            # Layer 1: HVAC rumble (low-frequency, constant) - FAST: just phase increment
            hvac_phase += hvac_phase_step
            if hvac_phase > 2 * math.pi:
                hvac_phase -= 2 * math.pi
            hvac = math.sin(hvac_phase) * 0.6  # 60% of noise
            total_noise += hvac
            
            # Layer 2: Keyboard typing (intermittent, every 2-3 seconds) - FAST: counter-based
            keyboard_counter += 1
            if not keyboard_active:
                if keyboard_counter > 16000:  # ~2 seconds at 8kHz
                    keyboard_active = True
                    keyboard_counter = 0
                    keyboard_phase = 0
            else:
                if keyboard_counter < 800:  # 0.1 second burst
                    keyboard_phase += 0.5  # Fast typing
                    typing = math.sin(keyboard_phase) * 0.5 * (1.0 - keyboard_counter / 800.0)
                    total_noise += typing
                else:
                    keyboard_active = False
                    keyboard_counter = 0
            
            # Layer 3: Distant conversations (pink noise - already optimized)
            white = random.uniform(-1.0, 1.0)
            pink_state[0] = 0.99886 * pink_state[0] + white * 0.0555179
            pink_state[1] = 0.99332 * pink_state[1] + white * 0.0750759
            pink_state[2] = 0.96900 * pink_state[2] + white * 0.1538520
            pink_state[3] = 0.86650 * pink_state[3] + white * 0.3104856
            pink_state[4] = 0.55000 * pink_state[4] + white * 0.5329522
            pink_state[5] = -0.7616 * pink_state[5] - white * 0.0168980
            pink = sum(pink_state) * 0.1  # Muffled conversations
            total_noise += pink * 0.5
            
            # Scale and clamp
            noise_scaled = int(total_noise * 32767 * noise_level)
            noise_scaled = max(-32768, min(32767, noise_scaled))
            noise_samples.append(noise_scaled)
        
        # Mix noise with original audio
        mixed_linear = []
        for i in range(num_samples):
            mixed = linear_audio[i] + noise_samples[i]
            mixed = max(-32768, min(32767, mixed))
            mixed_linear.append(mixed)
        
        # Convert back to MULAW
        mixed_mulaw = bytes([linear_to_ulaw_sample(sample) for sample in mixed_linear])
        
        return mixed_mulaw
        
    except Exception as e:
        print(f"⚠️ Office noise mixing failed: {e}, using clean audio")
        return audio_bytes

