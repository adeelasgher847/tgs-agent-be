"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Optimized for ultra-low latency (<3s response time)

ULTRA-AGGRESSIVE INTERIM PROCESSING:
- Processes interim STT results with 40% confidence
- Starts LLM generation immediately (100ms throttle)
- Minimal latency for fastest possible response

PARALLEL TTS PIPELINE (Vapi-style):
- User Speech → STT Interim → LLM Chunk 1 → TTS Chunk 1 Playing
                             ↓ LLM Chunk 2 → TTS Chunk 2 Generating (parallel)
                             ↓ LLM Chunk 3 → TTS Chunk 3 Queued
- TTS generation and playback happen in parallel
- Significantly reduces total response time

NATURAL CONVERSATION FEATURES (Vapi-Style):
1. SSML Support:
   - <prosody> tags with varied rate (95%, 98%, 100%, 102%, 105%)
   - <prosody> tags with varied pitch (-1st, 0st, +1st, +2st)
   - <break> tags for natural pauses (150-200ms)
   - <audio> tags for breath sounds (DISABLED by default - can cause distortion)
   - Example: <prosody rate="95%" pitch="+1st">Alright.</prosody><break time="180ms"/>

2. Micro-Pauses & Hesitation Fillers:
   - Hesitation patterns: "Hmm <break time='120ms'/> I think..."
   - "Uh", "Well", "Let me see" WITH breaks (15% chance)
   - SPOKEN boundary fillers at EVERY 10-word mid-chunk break (100%):
     * "<break time='80-100ms'/><prosody>uhh/umm/uh/hmm</prosody>" pattern
     * ALWAYS added between chunks to eliminate tak-tak distortion
     * No silent gaps - natural spoken connectors keep audio flowing
   - Example: <speak>Hmm <break time="120ms"/> I think I can help with that.</speak>

3. Backchannels:
   - "mm-hmm", "I see", "okay", "right", "yeah", "got it"
   - Triggered during long user monologues (5-7+ seconds)
   - 30% random chance for naturalness

4. Turn-Taking & Barge-In:
   - ENABLED - Agent stops immediately when user starts speaking
   - Detection: 2+ words (Google interim confidence is unreliable, often 0.00)
   - Checked FIRST before interim gating (highest priority!)
   - TTS queue cleared (prevents old audio from resuming)
   - Waits for final transcript before responding (no partial interruptions)

5. Persona & Variability:
   - Subtle prosody variations (95%-105% rate, ±1 semitone pitch)
   - Randomized breath/pause durations
   - Consistent voice persona from agent configuration

SMART CHUNKING WITH OVERLAP:
- 10 words per chunk for balanced quality + performance
- Major punctuation (. ! ? ;) triggers immediate chunk
- Minor punctuation (, : —) with 5+ words triggers chunk
- OVERLAP TECHNIQUE: Last 2 words of chunk 1 + filler + first words of chunk 2
  Example: Chunk 1: "Hello how are you doing today I'm doing" + SAVE("great thank")
           Chunk 2: "great thank" + "uhh" + "you for asking how"
  Result: Seamless transition with spoken fillers, no tak-tak distortion!

6. Ambient Background Noise:
   - DISABLED by default (caused distortion on some systems)
   - Can be enabled via self._use_ambient_noise = True
   - Subtle pink noise mixed with TTS audio (-46dB if enabled)
   - Note: Use with caution, may cause audio artifacts on certain setups

CACHING & LOW-LATENCY STRATEGIES:
1. Auto-Greeting on Connect:
   - Agent speaks FIRST when call connects (no waiting for user!)
   - Uses agent's first_message or default: "hello how are you"
   - Bypasses LLM entirely for instant greeting (<200ms)
   - Eliminates awkward silence at call start

2. Pre-cached Common Phrases:
   - 36+ common phrases pre-generated at startup
   - Greetings, acknowledgements, confirmations cached
   - <50ms response time for cached phrases (vs 500-2900ms generation)
   - Instant "Hello", "Got it", "Thank you" responses

3. Quick Acknowledgement Pattern:
   - Instant "Got it" from cache for 5+ word queries
   - Then full response streams in parallel
   - User gets immediate feedback while response generates
   - Example: "Got it" (50ms) → "checking that now..." (1500ms)

4. Adaptive Max Tokens:
   - Yes/No queries: 15 tokens (ultra-fast)
   - Short queries (1-3 words): 25 tokens (fast)
   - Medium queries (4-7 words): 35 tokens (balanced)
   - Complex queries: Full configured tokens
   - 30-60% faster LLM generation for simple queries

4. TTS Client Pre-warming:
   - Google TTS client initialized at startup
   - Avoids first-call penalty (~500ms saved)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import json
import base64
import asyncio
from typing import Optional, Dict, Iterable
import time
from datetime import datetime, timezone
import uuid
import sys
import math
import re

# Google built-in endpointing (VAD) will be used via streaming_recognize

from app.services.google_stt_service import google_stt_service
from app.services.google_tts_service import google_tts_service
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.services.transcript_service import transcript_service
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.services.groq_service import groq_service
from app.core.config import settings
from app.routers.tts_audio import audio_cache, generate_cache_key
from app.routers.general_websocket import broadcast_call_status_update
from app.middleware.tts_preprocessing_middleware import preprocess_for_tts, quick_clean
from app.utils.audio_constants import BACKGROUND_AUDIO_BASE64

# Real-time TTS MULAW streaming constants
MULAW_SAMPLE_RATE_HZ = 8000  # Twilio-friendly
BYTES_PER_SECOND = MULAW_SAMPLE_RATE_HZ  # 8-bit mu-law => 1 byte per sample
CHUNK_DURATION_SEC = 0.02  # 20ms
MULAW_FRAME_BYTES = int(BYTES_PER_SECOND * CHUNK_DURATION_SEC)  # 160 bytes

ULAW_BIAS = 0x84
ULAW_CLIP = 32635

router = APIRouter()


# Cache for decoded background audio
_background_audio_mulaw_cache = None
_background_audio_length_cache = 0


def decode_background_audio_from_base64() -> tuple[bytes, int]:
    """
    Decode base64 MP3 and convert to MULAW format.
    Returns (mulaw_bytes, length_in_bytes).
    Cached after first load.
    """
    global _background_audio_mulaw_cache, _background_audio_length_cache
    
    if _background_audio_mulaw_cache is not None:
        return _background_audio_mulaw_cache, _background_audio_length_cache
    
    if not BACKGROUND_AUDIO_BASE64:
        print("⚠️ No background audio configured (BACKGROUND_AUDIO_BASE64 is empty)")
        return b'', 0
    
    try:
        from io import BytesIO
        from pydub import AudioSegment
    except ImportError as import_error:
        error_msg = str(import_error)
        if "audioop" in error_msg:
            print(f"⚠️ Python 3.13+ detected: audioop module was removed")
            print(f"⚠️ Please install audioop-lts: pip install audioop-lts")
            print(f"⚠️ Background audio will be disabled until audioop-lts is installed")
        else:
            print(f"❌ Failed to import pydub: {import_error}")
        sys.stdout.flush()
        return b'', 0
    
    try:
        mp3_bytes = base64.b64decode(BACKGROUND_AUDIO_BASE64)
        audio = AudioSegment.from_mp3(BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(8000)
        audio = audio.set_channels(1)
        raw_audio = audio.raw_data
        linear_samples = []
        for i in range(0, len(raw_audio), 2):
            sample = int.from_bytes(raw_audio[i:i+2], byteorder='little', signed=True)
            linear_samples.append(sample)
        mulaw_bytes = bytes([linear_to_ulaw_sample(sample) for sample in linear_samples])
        
        _background_audio_mulaw_cache = mulaw_bytes
        _background_audio_length_cache = len(mulaw_bytes)
        
        print(f"✅ Decoded background audio from base64: {len(mulaw_bytes)} bytes MULAW ({len(mulaw_bytes)/8000:.2f}s)")
        sys.stdout.flush()
        return mulaw_bytes, len(mulaw_bytes)
        
    except Exception as e:
        print(f"❌ Failed to decode background audio: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return b'', 0


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


def strip_ssml_tags(text: str) -> str:
    """
    Remove all SSML tags from text, keeping only the actual text content.
    Used for saving clean text to transcript.
    Handles both complete and incomplete SSML tags.
    """
    if not text:
        return ""
    
    import re
    # Remove complete SSML tags (<tag>content</tag> or <tag/>)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove incomplete SSML tags (tags without closing >, like <break time="150ms)
    text = re.sub(r'<[^>]*', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()




def add_natural_ssml(text: str, use_ssml: bool = True, add_breaths: bool = True, add_fillers: bool = True, add_boundary_pause: bool = False) -> str:
    """
    Add SSML tags and natural speech elements for human-like delivery (Vapi-style).
    
    Features:
    1. SSML <prosody> tags with varied rate (95-105%) and pitch (±2st)
    2. <break> tags for natural pauses (100-200ms)
    3. <audio> tags for breath sounds between sentences (Google hosted)
    4. Hesitation fillers WITH breaks: "Hmm <break time='120ms'/> text"
    5. Boundary fillers with breaks for smooth chunk transitions
    
    Args:
        text: Input text
        use_ssml: Enable SSML tags
        add_breaths: Add breath pauses between sentences
        add_fillers: Add occasional fillers (uh, hmm)
        add_boundary_pause: Add natural pause/filler at chunk boundary
        
    Returns:
        Enhanced text with SSML and natural elements
    """
    import re
    import random
    
    if not text or not text.strip():
        return ""
    
    cleaned = text.strip()
    
    # If SSML not requested, just clean and return
    if not use_ssml:
        return clean_text_for_tts(cleaned)
    
    # Add hesitation fillers WITH breaks (Vapi-style - exact implementation)
    if add_fillers and random.random() < 0.15:
        # Hesitation patterns with breaks for natural pauses
        hesitations = [
            'Hmm <break time="120ms"/> ',
            'Uh <break time="100ms"/> ',
            'Well <break time="150ms"/> ',
            'Let me see <break time="180ms"/> ',
            'Umm <break time="130ms"/> ',
        ]
        cleaned = random.choice(hesitations) + cleaned
    
    # Wrap in SSML speak tag
    ssml = '<speak>'
    
    # Split into sentences for breath insertion
    sentences = re.split(r'([.!?;])', cleaned)
    
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i+1] if i+1 < len(sentences) else ""
        
        if not sentence:
            continue
        
        # Add prosody variation (Vapi-style - subtle speed and pitch changes)
        rate_variation = random.choice(["95%", "98%", "100%", "102%", "105%"])  # Vapi-style range
        pitch_variation = random.choice(["-1st", "0st", "+1st", "+2st"])  # Include +2st like Vapi
        
        ssml += f'<prosody rate="{rate_variation}" pitch="{pitch_variation}">'
        ssml += sentence + punct
        ssml += '</prosody>'
        
        # Add natural breath/pause after sentences (Vapi-style with audio!)
        if add_breaths and punct in ['.', '!', '?', ';']:
            # Vary break duration for naturalness
            break_time = random.choice(["150ms", "180ms", "200ms"])
            ssml += f'<break time="{break_time}"/>'
            
            # Add breath audio file for natural pauses (low volume, subtle)
            # Only after major sentence endings, not too frequently
            if random.random() < 0.15:  # 15% chance - very subtle
                BREATH_AUDIO = "https://actions.google.com/sounds/v1/human_voices/breath.ogg"
                ssml += f'<audio src="{BREATH_AUDIO}" soundLevel="-10dB"/>'  # Quieter breath
    
    # Add remaining text (if any)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        ssml += sentences[-1]
    
    # Add boundary filler for smooth chunk transitions - ALWAYS (100%)
    # Critical: This eliminates tak-tak distortion between chunks
    if add_boundary_pause:
        # ALWAYS add boundary connector (100% - not random!) for seamless audio
        # Boundary fillers with breaks for natural thinking pauses
        boundary_fillers = [
            ' <break time="80ms"/><prosody rate="90%" pitch="-1st">uhh</prosody>',
            ' <break time="90ms"/><prosody rate="88%" pitch="-2st">umm</prosody>',
            ' <break time="70ms"/><prosody rate="92%" pitch="0st">uh</prosody>',
            ' <break time="100ms"/><prosody rate="85%" pitch="-1st">hmm</prosody>',
        ]
        chosen_filler = random.choice(boundary_fillers)
        ssml += chosen_filler
        print(f"🔗 Added boundary filler: {chosen_filler[:50]}")
        sys.stdout.flush()
    
    ssml += '</speak>'
    
    return ssml


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
    import math
    
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
            hvac = math.sin(hvac_phase) * 0.6  # 60% of noise (increased from 0.4 for better audibility)
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
                    typing = math.sin(keyboard_phase) * 0.5 * (1.0 - keyboard_counter / 800.0)  # Increased from 0.3 to 0.5
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
            total_noise += pink * 0.5  # Increased from 0.3 to 0.5 for better audibility
            
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


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS to prevent reading punctuation marks aloud.
    Removes duplicate punctuation and normalizes spacing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text safe for TTS
    """
    import re
    if not text or not text.strip():
        return ""
    
    cleaned = text.strip()
    
    # Remove multiple punctuation (e.g., "!!!" -> "!", "..." -> ".")
    cleaned = re.sub(r'([.!?;,—:])\1+', r'\1', cleaned)
    
    # Remove standalone punctuation marks that get read as words
    cleaned = re.sub(r'\s+([.!?;,—:])\s+', r'\1 ', cleaned)
    
    # Ensure proper spacing after punctuation
    cleaned = re.sub(r'([.!?;,—:])([A-Za-z])', r'\1 \2', cleaned)
    
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()


def smart_chunk_text(text: str, max_words: int = 15) -> tuple[str, str]:
    """
    Smart text chunking that splits at natural pauses for smoother speech.
    Prefers splitting at sentence boundaries to maintain natural flow.
    
    Args:
        text: Text to split
        max_words: Maximum words in prefix chunk
        
    Returns:
        (prefix, suffix) tuple
    """
    if not text or not text.strip():
        return "", ""
    
    text = text.strip()
    words = text.split()
    
    # If text is short enough, return as-is
    if len(words) <= max_words:
        return text, ""
    
    # Try to split at sentence boundaries (., !, ?)
    sentence_endings = ['. ', '! ', '? ']
    best_split = None
    
    for ending in sentence_endings:
        parts = text.split(ending)
        if len(parts) > 1:
            prefix_candidate = parts[0] + ending.strip()
            prefix_words = len(prefix_candidate.split())
            
            # Use this split if it's within our word limit
            if prefix_words <= max_words and prefix_words > max_words * 0.5:
                best_split = (prefix_candidate, text[len(prefix_candidate):].strip())
                break
    
    # If no good sentence split, try comma split
    if not best_split and ', ' in text:
        parts = text.split(', ', 1)
        prefix_candidate = parts[0] + ','
        prefix_words = len(prefix_candidate.split())
        
        if prefix_words <= max_words and prefix_words > 5:
            best_split = (prefix_candidate, parts[1].strip())
    
    # Fallback: split at word count
    if not best_split:
        prefix = " ".join(words[:max_words])
        suffix = " ".join(words[max_words:])
        best_split = (prefix, suffix)
    
    return best_split


async def generate_mulaw_tts(text: str, lang: str = "en", voice: str = "female", use_chirp3_hd: bool = True, speaking_rate: float = 0.95, use_ssml: bool = False, add_office_bg: bool = False) -> bytes:
    """
    Generate mu-law (8kHz) TTS audio using Chirp 3: HD model.
    Optimized for word-by-word streaming with caching.
    
    Args:
        text: Text or SSML to convert to speech
        lang: Language code
        voice: Voice type (male/female)
        use_chirp3_hd: Use Chirp 3 HD model
        speaking_rate: Speech rate
        use_ssml: Whether text contains SSML markup
        add_office_bg: Add office background noise to audio (mixed at audio level)
    
    Note: Google TTS natively supports SSML. Text starting with <speak> is auto-detected.
    """
    # Skip empty text
    if not text or not text.strip():
        return b''
    
    # Cache key aligned with existing cache strategy (include ssml and office_bg flags)
    cache_key = generate_cache_key(text.strip(), lang, voice, use_chirp3_hd, "mulaw") + ("_ssml" if use_ssml else "") + ("_officebg" if add_office_bg else "")

    if cache_key in audio_cache:
        return audio_cache[cache_key]

    # Use 8kHz MULAW for Twilio with Chirp 3: HD model - Optimized for small chunks
    # Google TTS auto-detects SSML if text starts with <speak>
    # Let SSML control prosody (use defaults when SSML present, don't override)
    audio_content = google_tts_service.text_to_speech(
        text=text.strip(),
        language=lang,
        voice_type=voice,
        speaking_rate=1.0 if use_ssml else speaking_rate,  # Use 1.0 (default) for SSML to respect prosody tags
        pitch=0.0,  # Always 0, let SSML <prosody pitch> handle variations
        output_format="mulaw",
        use_chirp3_hd=use_chirp3_hd
    )

    # Mix office background noise if enabled (NO DOWNLOAD - generates programmatically!)
    if add_office_bg:
        audio_content = add_ambient_noise_to_mulaw(
            audio_content, 
            noise_level=0.06  # Office background noise (~-24dB) - audible but not distracting
        )
        print(f"🔊 Added office background noise to TTS audio (noise_level: 0.06)")

    # Cache for instant reuse (especially useful for repeated words/phrases)
    audio_cache[cache_key] = audio_content
    return audio_content


def build_streaming_twiml(call_session_id: str, agent_id: str) -> str:
    """
    Replace <Play> with <Start><Stream> to enable realtime MULAW streaming.
    Configure Twilio edge/region via settings.TWILIO_EDGE if available.
    """
    from twilio.twiml.voice_response import VoiceResponse, Start, Stream, Parameter
    
    # Your public WebSocket endpoint that handles Twilio Media Streams:
    # Example: wss://your-domain.com/api/v1/stream/ws/bidirectional/{callSessionId}/{agentId}
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/stream/ws/bidirectional/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS for Twilio

    edge = getattr(settings, "TWILIO_EDGE", None)  # e.g., "ashburn", "singapore", "dublin"
    vr = VoiceResponse()
    with vr.start() as s:
        stream = Stream(url=ws_url, name=f"tts-stream-{agent_id}")
        # Forward region hint to your WS (for observability); set real Twilio edge via account/call config.
        if edge:
            stream.parameter(Parameter(name="edge", value=edge))
        s.append(stream)

    return str(vr)


def build_tts_only_twiml(call_session_id: str, agent_id: str, record_callback_url: str) -> str:
    """
    Build TwiML for TTS-only WebSocket streaming + Recording for next input
    
    Flow:
    1. Connect to TTS-only WebSocket
    2. WebSocket auto-plays pending TTS from metadata
    3. After playback, start recording for next user input
    
    Args:
        call_session_id: Call session UUID
        agent_id: Agent UUID
        record_callback_url: URL for recording callback
    
    Returns:
        TwiML string
    """
    from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
    
    # Build TTS-only WebSocket URL
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/voice/ws/tts-only/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS
    
    vr = VoiceResponse()
    
    # Connect to TTS-only WebSocket for streaming playback
    connect = Connect()
    stream = Stream(url=ws_url, name=f"tts-only-{agent_id}")
    connect.append(stream)
    vr.append(connect)
    
    # After TTS playback, start recording for next user input
    vr.record(
        action=record_callback_url,
        method='POST',
        timeout=3,  # Fast detection
        max_length=120,
        play_beep=False,
        trim='do-not-trim',
        recording_status_callback=record_callback_url,
        recording_status_callback_method='POST',
        transcribe=False
    )
    
    return str(vr)


class BidirectionalStreamHandler:
    """Handles real-time bidirectional voice streaming"""
    
    def __init__(
        self,
        websocket: WebSocket,
        call_session_id: str,
        agent_id: str,
        db: Session
    ):
        self.websocket = websocket
        self.call_session_id = call_session_id
        self.agent_id = agent_id
        self.db = db
        
        # STT (Input) state - Google streaming_recognize with built-in VAD
        self.stream_sid = None
        self.call_sid = None
        self.current_speech = ""
        self._stt_session = None
        self._stt_task = None
        # Ultra-aggressive interim processing state (40% confidence)
        self._last_interim_text = ""
        self._last_interim_sent_ts = 0.0
        self._min_interim_words = 1  # speak sooner on shorter interim
        self._min_interim_confidence = 0.40  # ULTRA-AGGRESSIVE: process 40% confidence
        self._min_interim_interval_sec = 0.10  # ULTRA-AGGRESSIVE: 100ms throttle (was 200ms)
        
        # TTS (Output) state - Parallel Pipeline
        self.tts_queue = asyncio.Queue()     # Queue for parallel TTS processing
        self.is_speaking = False
        self._tts_cancel = asyncio.Event()   # barge-in cancel signal
        self._tts_lock = asyncio.Lock()      # serialize TTS streams
        self._tts_worker_task = None         # Background TTS worker
        self._tts_generation_tasks = []      # Track parallel TTS generation
        self._prev_tts_tail = b""            # Last streamed audio tail for crossfade bridge
        self._tts_overlap_bytes = 400        # 50ms overlap at 8kHz (Vapi's approach for smooth transitions)
        self._twilio_buffer_primed = False   # Track if jitter buffer has been primed
        
        # Natural conversation state (backchannels & persona)
        self._user_speech_duration = 0.0    # Track user monologue duration
        self._last_backchannel_time = 0.0   # Prevent frequent backchannels
        self._last_user_speech_start = 0.0  # Track when user started speaking
        self._backchannel_phrases = ["mm-hmm", "I see", "okay", "right", "yeah", "got it"]
        self._use_ssml = True                # Enable SSML by default
        
        # Session data
        self.call_session = None
        self.agent = None
        self._load_session_data()
        
        # User pickup detection (VAPI-style: actual user audio = user picked up)
        self._user_picked_up = False
        self._first_media_received = False
        self._skip_audio_until = None  # Timestamp until which to skip audio (system messages)
        self._audio_level_samples = []  # Track audio levels to detect actual user audio
        self._min_audio_level_threshold = 100  # Minimum audio level to consider as user audio (not silence/system noise)
        self._audio_samples_needed = 10  # Need 10 consecutive non-silent samples (200ms) to confirm user audio
        
        # Background audio state (embedded MP3)
        self._bg_audio_task = None
        self._bg_audio_offset = 0
        self._bg_audio_mulaw = None
        self._bg_audio_length = 0
        self._bg_audio_volume = 0.6  # 60% volume (-4.4dB) - increased for better audibility
        self._use_background_audio = False
        
        # Load and start background audio if available
        bg_audio_bytes, bg_audio_len = decode_background_audio_from_base64()
        if bg_audio_bytes and bg_audio_len > 0:
            self._bg_audio_mulaw = bg_audio_bytes
            self._bg_audio_length = bg_audio_len
            self._use_background_audio = True
            print(f"✅ Background audio loaded: {bg_audio_len} bytes ({bg_audio_len/8000:.2f}s)")
            sys.stdout.flush()
        
        # Pre-warm Google TTS client to avoid first-call penalty
        try:
            google_tts_service.get_client()
        except Exception as e:
            print(f"⚠️ TTS pre-warm failed: {e}")
            sys.stdout.flush()

        # Start parallel TTS pipeline worker
        self._tts_worker_task = asyncio.create_task(self._tts_pipeline_worker())
        
        # Pre-cache common phrases in background for instant responses
        asyncio.create_task(self._precache_common_phrases())

        print(f"✅ Bidirectional stream handler initialized (Google streaming STT + Streaming TTS)")
        print(f"⚡ ULTRA-AGGRESSIVE MODE: 40% confidence, 100ms throttle")
        print(f"🔄 PARALLEL TTS PIPELINE: Started background worker")
        print(f"🔥 PRE-CACHING: Common phrases loading in background...")
        sys.stdout.flush()
    
    def _load_session_data(self):
        """Load call session and agent data"""
        try:
            session_uuid = uuid.UUID(self.call_session_id)
            self.call_session = call_session_service.get_call_session_by_id(self.db, session_uuid)
            
            if self.call_session and self.agent_id:
                agent_uuid = uuid.UUID(self.agent_id)
                self.agent = agent_service.get_agent_by_id(
                    self.db,
                    agent_uuid,
                    self.call_session.tenant_id
                )
                print(f"✅ Loaded agent: {self.agent.name if self.agent else 'Unknown'}")
                sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ Error loading session data: {e}")
            sys.stdout.flush()
    
    async def _precache_common_phrases(self):
        """
        Pre-generate and cache common phrases for instant playback.
        Runs in background during initialization.
        """
        try:
            # Common phrases for instant responses (greetings, confirmations, acknowledgements)
            common_phrases = [
                # Greetings
                "Hello",
                "Hi there",
                "Hi",
                "Good morning",
                "Good afternoon",
                "Good evening",
                
                # Acknowledgements (Quick feedback)
                "Got it",
                "I see",
                "Okay",
                "Sure",
                "Alright",
                "Perfect",
                "Great",
                "Understood",
                
                # Confirmations
                "Yes",
                "No",
                "Absolutely",
                "Of course",
                
                # Thinking/Processing
                "Let me check that",
                "One moment please",
                "Just a second",
                "Let me see",
                
                # Transitions
                "Thank you",
                "Thanks",
                "You're welcome",
                
                # Closings
                "Goodbye",
                "Have a great day",
                "Thank you for calling",
                "Talk to you later",
            ]
            
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            
            print(f"🔥 Pre-caching {len(common_phrases)} common phrases for instant playback...")
            sys.stdout.flush()
            
            cached_count = 0
            for phrase in common_phrases:
                try:
                    # Generate and cache (async, non-blocking)
                    await generate_mulaw_tts(
                        text=phrase,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,  # Changed from 0.95 to 1.0
                        use_ssml=False
                    )
                    cached_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to cache '{phrase}': {e}")
                    continue
            
            print(f"✅ Pre-cache complete: {cached_count}/{len(common_phrases)} phrases cached ({len(audio_cache)} total items)")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ Pre-cache error: {e}")
            sys.stdout.flush()
    
    async def _tts_pipeline_worker(self):
        """
        Background worker for parallel TTS pipeline (Vapi-style).
        
        Processes TTS chunks from queue while new chunks are being generated:
        - LLM Chunk 1 → TTS generates → plays
        - LLM Chunk 2 → TTS generates (parallel) → queued
        - LLM Chunk 3 → TTS generates (parallel) → queued
        """
        print("🔄 TTS Pipeline Worker: Started")
        sys.stdout.flush()
        
        try:
            while True:
                # Get next TTS task from queue
                task = await self.tts_queue.get()
                
                # Check for shutdown signal
                if task is None:
                    print("🔄 TTS Pipeline Worker: Shutdown signal received")
                    sys.stdout.flush()
                    break
                
                # Check if cancelled (barge-in)
                if self._tts_cancel.is_set():
                    print("🛑 TTS Pipeline Worker: Skipping chunk due to barge-in")
                    sys.stdout.flush()
                    self.tts_queue.task_done()
                    continue
                
                try:
                    text = task.get("text", "")
                    chunk_id = task.get("chunk_id", "unknown")
                    use_ssml = task.get("use_ssml", False)
                    is_backchannel = task.get("is_backchannel", False)
                    is_final = task.get("is_final", False)
                    
                    if not text or not text.strip():
                        self.tts_queue.task_done()
                        continue
                    
                    if is_backchannel:
                        print(f"🗣️ TTS Pipeline: Processing backchannel: '{text}'")
                    else:
                        print(f"🔄 TTS Pipeline: Processing chunk {chunk_id}: '{text[:30]}...'")
                    sys.stdout.flush()
                    
                    # Generate and stream TTS (this is the blocking part)
                    await self._stream_tts_chunk(text, use_ssml=use_ssml, is_final=is_final)
                    
                    print(f"✅ TTS Pipeline: Completed chunk {chunk_id}")
                    sys.stdout.flush()
                    
                except Exception as e:
                    print(f"❌ TTS Pipeline Worker: Error processing chunk: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                finally:
                    self.tts_queue.task_done()
        
        except Exception as e:
            print(f"❌ TTS Pipeline Worker: Fatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    
    
    async def handle_media_message(self, message: dict):
        """Handle incoming audio from Twilio and feed to Google streaming STT"""
        try:
            import time
            
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode audio (MULAW from Twilio)
            audio_data = base64.b64decode(payload)
            
            # ✅ DETECT ACTUAL USER AUDIO (not Twilio system messages/music)
            if not self._first_media_received:
                self._first_media_received = True
                print(f"📡 First media packet received - analyzing for actual user audio...")
                sys.stdout.flush()
            
            # Calculate audio level (RMS) to detect actual user audio vs silence/system noise
            if not self._user_picked_up:
                # Convert MULAW to linear and calculate RMS (Root Mean Square) audio level
                audio_level = 0
                if len(audio_data) > 0:
                    linear_samples = [ulaw_to_linear_sample(b) for b in audio_data]
                    # Calculate RMS
                    sum_squares = sum(s * s for s in linear_samples)
                    audio_level = int((sum_squares / len(linear_samples)) ** 0.5)
                
                # Track audio levels
                self._audio_level_samples.append(audio_level)
                if len(self._audio_level_samples) > self._audio_samples_needed * 2:
                    self._audio_level_samples.pop(0)  # Keep last 20 samples
                
                # Check if we have enough samples and enough non-silent audio (actual user audio)
                if len(self._audio_level_samples) >= self._audio_samples_needed:
                    non_silent_count = sum(1 for level in self._audio_level_samples[-self._audio_samples_needed:] if level > self._min_audio_level_threshold)
                    
                    if non_silent_count >= self._audio_samples_needed:
                    # Actual user audio detected! Set in-progress status
                    self._user_picked_up = True
                    await self._handle_user_pickup()  # User actually picked up with real audio!
                    
                    # Skip first few seconds for STT (system messages might still be there)
                    self._skip_audio_until = time.time() + 3.0
                    print(f"✅ Actual user audio detected (avg level: {sum(self._audio_level_samples)/len(self._audio_level_samples):.0f}) - Setting in-progress status")
                    sys.stdout.flush()
                else:
                    # Still waiting for actual user audio (might be Twilio system messages/music)
                    avg_level = sum(self._audio_level_samples) / len(self._audio_level_samples) if self._audio_level_samples else 0
                    if len(self._audio_level_samples) % 50 == 0:  # Log every 50 packets
                        print(f"⏳ Waiting for user audio... (current level: {avg_level:.0f}, need {self._audio_samples_needed} non-silent samples)")
                        sys.stdout.flush()
                    return  # Don't process until we have actual user audio
            
            # Skip audio if still in grace period (system messages)
            if self._skip_audio_until and time.time() < self._skip_audio_until:
                return  # Don't send to STT - this is likely system message/ringing

            # (Removed first-media DB marker for outbound gating)
            # Lazily create a streaming session
            if self._stt_session is None:
                self._stt_session = google_stt_service.create_streaming_session(
                    language_code=(self.agent.language + "-US") if getattr(self.agent, "language", None) == "en" else None,
                    encoding="MULAW",
                    sample_rate=8000,
                    interim_results=True,
                    single_utterance=False,
                )

                async def consume_results():
                    try:
                        # Start underlying blocking stream in executor
                        await self._stt_session.start()
                    except Exception as e:
                        print(f"❌ STT streaming start error: {e}")
                        sys.stdout.flush()
                    # Drain any remaining
                
                # Start the session in background and concurrently read results
                async def reader_loop():
                    while True:
                        result = await self._stt_session.get_result()
                        if not result:
                            continue
                        if result.get("error"):
                            print(f"❌ STT error: {result['error']}")
                            sys.stdout.flush()
                            continue
                        transcript = (result.get("transcript") or "").strip()
                        if not transcript:
                            continue
                        is_final = bool(result.get("is_final"))
                        confidence = float(result.get("confidence") or 0.0)
                        if is_final:
                            print(f"📝 Final STT: '{transcript}' ({confidence:.2f})")
                            sys.stdout.flush()
                            await self._process_transcript(transcript, confidence)
                        else:
                            # Process interim for ultra-low latency (Vapi-like)
                            await self._maybe_process_interim(transcript, confidence)

                # kick off background readers
                self._stt_task = asyncio.create_task(reader_loop())
                asyncio.create_task(consume_results())

            # Push audio to Google
            self._stt_session.push_audio(audio_data)
        
        except Exception as e:
            print(f"❌ Error handling media: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    # Removed chunk-based STT processing; relying on Google streaming endpointing
    
    async def _process_transcript(self, transcript: str, confidence: float):
        """Process a transcript (final result)"""
        try:
            if not transcript or confidence < 0.3:
                print(f"⚠️ Skipping low-confidence transcript (confidence: {confidence:.2f})")
                sys.stdout.flush()
                return
            
            # Reset user speech timer (user finished speaking)
            self._last_user_speech_start = 0.0
            
            # Reset interim state (user finished, ready for new response)
            self._tts_cancel.clear()
            self._last_interim_text = ""
            print(f"✅ User finished speaking - ready to respond")
            sys.stdout.flush()
            
            # Add to transcript
            await self._add_to_transcript("client", transcript, "speech", confidence)
            
            # Generate and stream response
            await self.generate_and_stream_response(transcript, confidence)
            
        except Exception as e:
            print(f"❌ Error processing transcript: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    async def _maybe_inject_backchannel(self, transcript: str):
        """
        Inject backchannel responses during long user monologues.
        Triggered after 5-7 seconds of continuous user speech.
        """
        import random
        
        now = time.time()
        
        # Track user speech duration
        if not self._last_user_speech_start:
            self._last_user_speech_start = now
        
        speech_duration = now - self._last_user_speech_start
        time_since_last_backchannel = now - self._last_backchannel_time
        
        # Inject backchannel if:
        # 1. User has been speaking for 5-7+ seconds
        # 2. At least 3 seconds since last backchannel
        # 3. We're not currently speaking
        # 4. Random chance (30%) for naturalness
        should_backchannel = (
            speech_duration >= random.uniform(5.0, 7.0) and
            time_since_last_backchannel >= 3.0 and
            not self.is_speaking and
            random.random() < 0.3
        )
        
        if should_backchannel:
            backchannel = random.choice(self._backchannel_phrases)
            print(f"🗣️ Injecting backchannel: '{backchannel}' (user spoke for {speech_duration:.1f}s)")
            sys.stdout.flush()
            
            # Queue backchannel with minimal processing
            await self.tts_queue.put({
                "text": backchannel,
                "chunk_id": "backchannel",
                "is_backchannel": True,
                "is_final": True,
                "use_ssml": False
            })
            
            self._last_backchannel_time = now

    async def _maybe_process_interim(self, transcript: str, confidence: float):
        """
        ULTRA-AGGRESSIVE interim processing for minimal latency.
        Processes interim STT results with 40% confidence to start LLM generation ASAP.
        Also tracks user speech for backchannel injection.
        """
        try:
            if not transcript:
                return
            
            # Check for backchannel opportunity during long user speech
            await self._maybe_inject_backchannel(transcript)
            
            # Calculate word count for checks
            word_count = len(transcript.split())
            
            # ✅ BARGE-IN CHECK FIRST - Highest priority! Stop agent immediately!
            # Detection: 2+ words, agent currently speaking
            # NOTE: We check this BEFORE interim gate because barge-in should work
            # even with low confidence (Google interim often returns 0.00 confidence)
            if self.is_speaking and word_count >= 2:
                if not self._tts_cancel.is_set():
                    print(f"🛑 BARGE-IN DETECTED!")
                    print(f"   User: '{transcript[:60]}...' (words: {word_count}, confidence: {confidence:.2f})")
                    print(f"   Stopping TTS and clearing all buffers...")
                    sys.stdout.flush()
                    
                    # Set cancel flag to stop streaming
                    self._tts_cancel.set()
                    
                    # 🆕 CLEAR TTS QUEUE - Remove ALL pending audio chunks!
                    # This prevents old audio from resuming after user finishes speaking
                    cleared_count = 0
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                            cleared_count += 1
                        except asyncio.QueueEmpty:
                            break
                    
                    if cleared_count > 0:
                        print(f"🗑️ Cleared {cleared_count} pending TTS chunk(s) from queue")
                        sys.stdout.flush()
                    
                    # Mark agent as no longer speaking
                    self.is_speaking = False
                    
                    print(f"🎧 Agent stopped - now listening to user...")
                    sys.stdout.flush()
                
                # Don't process interim during barge-in - wait for final transcript
                return
            
            # Basic gating: confidence and minimum words (for LLM generation only)
            # NOTE: This comes AFTER barge-in check so that interruption works even with low confidence
            if confidence < self._min_interim_confidence or word_count < self._min_interim_words:
                # Still log interim for observability
                print(f"⌛ Interim (gated) [{confidence:.2f}]: '{transcript[:60]}...'")
                sys.stdout.flush()
                return
            
            # Ultra-aggressive throttling: only 100ms between triggers
            now = asyncio.get_event_loop().time()
            if (now - self._last_interim_sent_ts) < self._min_interim_interval_sec:
                print(f"⌛ Interim (throttled): '{transcript[:60]}...'")
                sys.stdout.flush()
                return
            
            # ULTRA-AGGRESSIVE: Process even small advances (no minimum word requirement)
            # This ensures we start LLM generation as soon as possible
            if self._last_interim_text and transcript.startswith(self._last_interim_text):
                advanced = transcript[len(self._last_interim_text):].strip()
                # Skip only if there's literally no new content
                if not advanced:
                    print(f"⌛ Interim (no-advance): '{transcript[:60]}...'")
                    sys.stdout.flush()
                    return
                # Process even single character advances for ultra-low latency
                print(f"⚡ ULTRA-AGGRESSIVE: Processing advance '{advanced}' (total: '{transcript[:60]}')")
                sys.stdout.flush()
            
            # Passed heuristics → process immediately to start LLM generation
            print(f"⚡⚡ ULTRA-AGGRESSIVE INTERIM [{confidence:.2f}]: '{transcript[:60]}'")
            sys.stdout.flush()
            self._last_interim_text = transcript
            self._last_interim_sent_ts = now
            await self.generate_and_stream_response(transcript, confidence)
        except Exception as e:
            print(f"❌ Error in interim processing: {e}")
            sys.stdout.flush()
    
    async def _send_quick_acknowledgement(self, user_text: str):
        """
        Send instant acknowledgement for longer queries while generating full response.
        Uses pre-cached phrases for <50ms latency.
        """
        import random
        
        # Only for longer queries (5+ words) - short queries get direct response
        if len(user_text.split()) >= 5:
            # Quick acknowledgements (these should be pre-cached = instant!)
            acks = ["Got it", "I see", "Okay", "Sure", "Let me check that"]
            ack = random.choice(acks)
            
            print(f"⚡ Sending instant acknowledgement: '{ack}' (from cache)")
            sys.stdout.flush()
            
            # Queue cached acknowledgement (should be instant!)
            await self.tts_queue.put({
                "text": ack,
                "chunk_id": "quick_ack",
                "use_ssml": False,
                "is_acknowledgement": True,
                "is_final": False
            })
    
    async def generate_and_stream_response(self, user_text: str, confidence: float, is_greeting: bool = False):
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        
        Args:
            user_text: User's input text (empty for greeting)
            confidence: STT confidence score
            is_greeting: If True, uses agent's first_message instead of calling LLM
        """
        try:
            from datetime import datetime, timezone
            import json
            
            # 👋 HANDLE AUTO-GREETING - Skip LLM, use pre-defined greeting
            if is_greeting:
                # Get greeting from agent or use default
                if self.agent and hasattr(self.agent, 'first_message') and self.agent.first_message:
                    greeting_text = self.agent.first_message
                else:
                    greeting_text = "hello how are you"
                
                print(f"👋 Using auto-greeting (no LLM): '{greeting_text}'")
                sys.stdout.flush()
                
                # Add greeting to transcript
                await self._add_to_transcript("agent", greeting_text, "greeting")
                
                # Queue greeting TTS directly (skip LLM!)
                await self.tts_queue.put({
                    "text": greeting_text,
                    "chunk_id": "greeting",
                    "use_ssml": self._use_ssml,
                    "is_final": True
                })
                
                print(f"✅ Auto-greeting queued for TTS")
                sys.stdout.flush()
                return  # Done! No LLM needed for greeting
            
            # Reset cancel flag for new response generation
            self._tts_cancel.clear()
            # Reset crossfade state so new response starts clean
            self._prev_tts_tail = b""
            
            # Send quick acknowledgement for longer queries (instant from cache!)
            await self._send_quick_acknowledgement(user_text)
            
            print(f"🤖 Generating streaming response for: '{user_text}'")
            sys.stdout.flush()
            
            # Build conversation context from transcript
            conversation_history = []
            if self.call_session and self.call_session.call_transcript:
                try:
                    conversation_history = json.loads(self.call_session.call_transcript) if isinstance(self.call_session.call_transcript, str) else self.call_session.call_transcript
                except:
                    conversation_history = []
            
            # Build history text - FULL HISTORY (VAPI-style) - handle different formats
            history_text = ""
            if conversation_history:
                try:
                    history_lines = []
                    # Use FULL history instead of last 6 messages (VAPI approach)
                    for msg in conversation_history:  # Full history, not [-6:]
                        if isinstance(msg, dict):
                            # Handle both 'content' and 'message' keys
                            role = msg.get('role', 'unknown')
                            content = msg.get('content') or msg.get('message', '')
                            message_type = msg.get('message_type', '')
                            
                            # Filter: Only include client and agent messages (skip system/greeting/status messages)
                            if content and role in ['client', 'agent'] and message_type not in ['greeting', 'system', 'status']:
                                history_lines.append(f"{role.capitalize()}: {content}")
                    history_text = "\n".join(history_lines)
                    print(f"📝 Full conversation history: {len(history_lines)} messages ({len(history_text)} chars)")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"⚠️ Error building history text: {e}")
                    sys.stdout.flush()
                    history_text = ""
            
            # Build system prompt with agent personality + history
            agent_name = self.agent.name if self.agent and self.agent.name else "AI Assistant"
            agent_language = self.agent.language if self.agent and self.agent.language else "en"
            
            # Base prompt for phone conversations
            base_prompt = f"""You are {agent_name}, a real person having a phone conversation.

Guidelines:
- Speak naturally and conversationally
- Keep responses brief (1-2 sentences)
- Remember previous conversation - don't repeat questions
- Be friendly and engaging
- Talk like a real human, not a robot

Previous conversation:
{history_text}

IMPORTANT: 
- Use the conversation history above. Don't ask questions you already asked. Continue the conversation naturally.
- If you have completed all your objectives/questions from the conversation, naturally end the conversation with a friendly closing (e.g., "Thank you for calling, have a great day!" or similar). 
- DO NOT restart the conversation, repeat questions, or ask questions again once everything is complete. Once all objectives are finished, end the call gracefully.
- Example: <speak><prosody rate="100%" pitch="0st">Hello, how can I help you?</prosody><break time="150ms"/><prosody rate="98%" pitch="-1st">Let me check that for you.</prosody></speak>"""
            
            # Use agent's custom system prompt if available, otherwise use base prompt
            if self.agent and self.agent.system_prompt:
                # Agent has custom system prompt - use it with context
                system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation

Custom instructions:
{self.agent.system_prompt}

Previous conversation:
{history_text}

Guidelines:
- Keep responses brief (1-2 sentences) for phone conversations
- Use the conversation history to provide relevant responses
- Don't repeat information you've already shared
- Talk naturally like a real person

IMPORTANT: 
- Follow your custom instructions above while maintaining natural conversation flow.
- If you have completed all objectives/questions from your custom instructions, naturally end the conversation with a friendly closing. 
- DO NOT restart the conversation, repeat questions, or ask new questions once all objectives are complete. Once everything is finished, end the call gracefully."""
            elif self.agent and self.agent.model and self.agent.model.system_prompt:
                # Model has system prompt - use it
                system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally

Model instructions:
{self.agent.model.system_prompt}

Previous conversation:
{history_text}

Guidelines:
- Keep responses brief (1-2 sentences) for phone conversations
- Use the conversation history to provide relevant responses

IMPORTANT: 
- Follow the model instructions above.
- If you have completed all objectives/questions, naturally end the conversation with a friendly closing. 
- DO NOT restart the conversation, repeat questions, or ask questions again once everything is complete. Once all objectives are finished, end the call gracefully."""
            else:
                # Use base prompt
                system_prompt = base_prompt
            
            # Log which system prompt is being used
            if self.agent and self.agent.system_prompt:
                print(f"📝 Using agent's custom system prompt")
            elif self.agent and self.agent.model and self.agent.model.system_prompt:
                print(f"📝 Using model's system prompt")
            else:
                print(f"📝 Using default base prompt")
            sys.stdout.flush()
            
            # Get agent's configured model and provider
            llm_service = None
            model_name = "gemini-1.5-flash"  # Default fallback
            api_key = None
            temperature = 0.5
            max_tokens = 100
            
            if self.agent and self.agent.model:
                model_name = self.agent.model.model_name
                
                # Decrypt API key if available
                if self.agent.model.api_key:
                    try:
                        from app.core.security import decrypt_api_key
                        api_key = decrypt_api_key(self.agent.model.api_key)
                        print(f"🔑 Using decrypted model-specific API key")
                    except Exception as e:
                        print(f"⚠️ Failed to decrypt API key: {e}")
                        api_key = None  # Will fallback to settings.OPENAI_API_KEY or settings.GOOGLE_API_KEY
                else:
                    api_key = None  # Will use global key from .env
                
                # Use agent-specific config if available
                if self.agent.agent_temperature is not None:
                    temperature = self.agent.agent_temperature / 100.0  # Convert 0-100 to 0-1
                elif self.agent.model.temperature is not None:
                    temperature = self.agent.model.temperature / 100.0
                
                if self.agent.agent_max_tokens:
                    max_tokens = self.agent.agent_max_tokens
                elif self.agent.model.max_tokens:
                    max_tokens = self.agent.model.max_tokens
                
                # Use configured max_tokens exactly as set by user (no adaptive adjustments)
                print(f"📝 Using configured max_tokens: {max_tokens}")
                sys.stdout.flush()
                
                # Select service based on provider
                if self.agent.provider:
                    provider_name = self.agent.provider.name.lower()
                    if "openai" in provider_name:
                        llm_service = openai_service
                        print(f"🤖 Using OpenAI: {model_name}")
                    elif "gemini" in provider_name or "google" in provider_name:
                        llm_service = gemini_service
                        print(f"🤖 Using Gemini: {model_name}")
                    elif "groq" in provider_name:
                        llm_service = groq_service
                        print(f"🚀 Using Groq: {model_name}")
                    else:
                        # Default to Gemini
                        llm_service = gemini_service
                        print(f"⚠️ Unknown provider '{provider_name}', defaulting to Gemini")
                else:
                    llm_service = gemini_service
                    print(f"⚠️ No provider configured, defaulting to Gemini")
            else:
                # Fallback to Gemini
                llm_service = gemini_service
                print(f"⚠️ No model configured for agent, using default Gemini")
            
            sys.stdout.flush()
            
            # Stream LLM output and QUEUE for PARALLEL TTS PIPELINE (Vapi-style)
            chunk_counter = 0
            
            async def try_stream(service, model: str, api_key_override: str = None) -> str:
                nonlocal chunk_counter
                response_accum = ""

                # SIMPLE: Collect FULL response first (no chunking during LLM streaming)
                async for chunk in service.stream_text(
                    prompt=user_text,
                    system_prompt=system_prompt,
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key_override
                ):
                    if not chunk:
                        continue
                    # If barge-in requested, stop generating
                    if self._tts_cancel.is_set():
                        print("🛑 Barge-in: aborting current LLM stream")
                        sys.stdout.flush()
                        break

                    response_accum += chunk

                # Now generate TTS for FULL response (no chunks, no distortion!)
                full_response = response_accum.strip()
                
                if full_response and not self._tts_cancel.is_set():
                    print(f"📝 Full LLM response ready ({len(full_response.split())} words): '{full_response[:60]}...'")
                    sys.stdout.flush()
                    
                    # Preprocess with SSML using middleware (version 1.0.1 style - simple and reliable)
                    if self._use_ssml:
                        # Strip any SSML tags that LLM might have generated (we use middleware for SSML)
                        clean_text = strip_ssml_tags(full_response)
                        enhanced_text = preprocess_for_tts(clean_text)
                        print(f"🔄 Using middleware SSML generation (version 1.0.1 style)")
                        sys.stdout.flush()
                    else:
                        enhanced_text = quick_clean(full_response)
                    
                    # Queue SINGLE TTS chunk (complete response)
                    chunk_counter += 1
                    await self.tts_queue.put({
                        "text": enhanced_text,
                        "chunk_id": chunk_counter,
                        "use_ssml": self._use_ssml,
                        "is_final": True
                    })
                    print(f"🔄 Queued FULL TTS response (no chunks): '{full_response[:50]}...'")
                    sys.stdout.flush()

                return response_accum.strip()

            final_text = None
            try:
                # Use agent's configured model and service
                final_text = await try_stream(llm_service, model_name, api_key)
            except Exception as e1:
                print(f"⚠️ Streaming with {model_name} failed: {e1}")
                sys.stdout.flush()
                # Fallback: try alternate service
                try:
                    if llm_service == openai_service:
                        # Fallback to Gemini
                        print("⚠️ Falling back to Gemini gemini-1.5-flash")
                        sys.stdout.flush()
                        final_text = await try_stream(gemini_service, "gemini-1.5-flash", None)
                    else:
                        # Fallback to OpenAI
                        print("⚠️ Falling back to OpenAI gpt-3.5-turbo")
                        sys.stdout.flush()
                        final_text = await try_stream(openai_service, "gpt-3.5-turbo", None)
                except Exception as e2:
                    print(f"⚠️ Streaming with fallback model failed: {e2}")
                    sys.stdout.flush()
                    # Last fallback: non-streaming fast response via VoiceLoggingService
                    try:
                        final_text = await VoiceLoggingService.generate_agent_response(
                            speech_text=user_text,
                            confidence=confidence,
                            agent=self.agent,
                            db=self.db,
                            call_session_id=self.call_session.id if self.call_session else None
                        )
                        # Queue fallback response
                        if final_text and not self._tts_cancel.is_set():
                            chunk_counter += 1
                            await self.tts_queue.put({
                                "text": final_text,
                                "chunk_id": chunk_counter,
                                "use_ssml": self._use_ssml,
                                "is_final": True
                            })
                            print(f"🔄 Queued fallback TTS chunk {chunk_counter}")
                            sys.stdout.flush()
                    except Exception as e3:
                        print(f"⚠️ All fallbacks failed: {e3}")
                        sys.stdout.flush()
                        # Ultimate fallback response
                        final_text = "I apologize, I'm having trouble responding right now. Could you please repeat that?"
                        chunk_counter += 1
                        await self.tts_queue.put({
                            "text": final_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._use_ssml,
                            "is_final": True
                        })

            if final_text:
                await self._add_to_transcript("agent", final_text, "agent_response")
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            sys.stdout.flush()
    
    async def _stream_tts_chunk(self, text: str, use_ssml: bool = False, is_final: bool = False):
        """
        Generate and stream a single TTS chunk (used by parallel pipeline worker).
        Simplified version without the complex prefix/suffix splitting.
        Note: Does NOT clear cancel flag - respects barge-in for entire queue.
        
        Args:
            text: Text or SSML to convert to speech
            use_ssml: Whether text contains SSML markup
        """
        try:
            from datetime import datetime, timezone
            
            if not text or not text.strip():
                return
            
            # Check if already cancelled before acquiring lock
            if self._tts_cancel.is_set():
                print(f"🛑 Skipping TTS chunk due to barge-in: '{text[:30]}...'")
                sys.stdout.flush()
                return
            
            async with self._tts_lock:
                self.is_speaking = True
                try:
                    lang = self.agent.language if self.agent and self.agent.language else "en"
                    voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
                    clean = text.strip()
                    
                    if use_ssml:
                        print(f"🎵 Generating TTS with SSML for chunk: '{clean[:40]}...'")
                    else:
                        print(f"🎵 Generating TTS for chunk: '{clean[:40]}...'")
                    sys.stdout.flush()
                    
                    # Generate TTS audio (Google TTS auto-detects SSML)
                    # Note: add_office_bg=False because mixing is handled during streaming
                    tts_start = datetime.now(timezone.utc)
                    audio_bytes = await generate_mulaw_tts(
                        text=clean,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,  # Changed from 0.95 to 1.0 for normal speed
                        use_ssml=use_ssml,
                        add_office_bg=False  # Background mixing handled separately
                    )
                    tts_gen_time = (datetime.now(timezone.utc) - tts_start).total_seconds()
                    print(f"⏱️ TTS generation: {tts_gen_time:.3f}s for '{clean[:20]}...'")
                    sys.stdout.flush()
                    
                    if self._tts_cancel.is_set():
                        self._prev_tts_tail = b""
                        return
                    
                    # Stream TTS CLEAN (no background mixing when AI is speaking)
                    # Background audio loop will automatically pause when is_speaking=True
                    if audio_bytes and not self._tts_cancel.is_set():
                        prime_frames = 0 if self._twilio_buffer_primed else 1
                        
                        # Always stream clean TTS - background loop handles background separately
                        # Background loop pauses automatically when is_speaking=True
                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=audio_bytes,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=prime_frames,
                        )
                        self._twilio_buffer_primed = True
                        print(f"✅ Streamed complete response: {len(audio_bytes)} bytes ({len(audio_bytes)/8000:.2f}s audio)")
                        sys.stdout.flush()
                finally:
                    if self._tts_cancel.is_set():
                        self._prev_tts_tail = b""
                    self.is_speaking = False
        
        except Exception as e:
            print(f"❌ Error streaming TTS chunk: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    async def _stream_background_audio_loop(self):
        """
        Continuously stream background audio in a loop.
        PAUSES when TTS is speaking to avoid conflicts.
        Uses proper pacing with drift correction for smooth playback.
        """
        if not self._bg_audio_mulaw or self._bg_audio_length == 0:
            return
        
        print(f"🔄 Background audio loop started (continuous streaming, pauses during TTS)")
        sys.stdout.flush()
        
        send_interval = 0.02  # 20ms per frame
        frame_bytes = MULAW_FRAME_BYTES  # 160 bytes at 8kHz
        first = True
        next_send = time.perf_counter()
        
        try:
            while True:
                if not self.stream_sid:
                    await asyncio.sleep(0.1)
                    continue
                
                # PAUSE background audio when TTS is speaking (no noise when AI speaks)
                if self.is_speaking:
                    # Reset timing when pausing so resume is smooth
                    first = True
                    next_send = time.perf_counter()
                    await asyncio.sleep(0.1)  # Check every 100ms
                    continue
                
                bg_chunk = get_background_audio_chunk(
                    self._bg_audio_offset,
                    frame_bytes,
                    self._bg_audio_mulaw,
                    self._bg_audio_length
                )
                
                self._bg_audio_offset = (self._bg_audio_offset + frame_bytes) % self._bg_audio_length
                
                payload = base64.b64encode(bg_chunk).decode("utf-8")
                await self.websocket.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload}
                })
                
                # Proper pacing with drift correction (same as TTS streaming)
                if not first:
                    next_send += send_interval
                    now = time.perf_counter()
                    sleep_dur = next_send - now
                    if sleep_dur > 0:
                        await asyncio.sleep(sleep_dur)
                    elif sleep_dur < -0.03:
                        # We're late by >30ms; reset schedule to avoid cumulative jitter
                        next_send = time.perf_counter()
                else:
                    first = False
                    next_send = time.perf_counter() + send_interval
                
        except asyncio.CancelledError:
            print(f"🛑 Background audio streaming cancelled")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Background audio streaming error: {e}")
            sys.stdout.flush()
    
    async def _stream_mulaw_with_background(
        self,
        audio_bytes: bytes,
        cancel: Optional[asyncio.Event] = None
    ):
        """
        Stream TTS audio mixed with continuous background audio.
        Uses proper pacing with drift correction to prevent stuttering.
        """
        send_interval = 0.02  # 20ms
        first = True
        next_send = time.perf_counter()
        
        for frame in iter_mulaw_20ms_frames(audio_bytes):
            if cancel and cancel.is_set():
                break
            
            if self._bg_audio_mulaw and self._bg_audio_length > 0:
                mixed_frame = mix_audio_with_background(
                    tts_audio=frame,
                    bg_audio=self._bg_audio_mulaw,
                    bg_length=self._bg_audio_length,
                    bg_offset=self._bg_audio_offset,
                    volume_level=self._bg_audio_volume
                )
                self._bg_audio_offset = (self._bg_audio_offset + len(frame)) % self._bg_audio_length
            else:
                mixed_frame = frame
            
            payload = base64.b64encode(mixed_frame).decode("utf-8")
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload}
            })
            
            # Proper pacing with drift correction (same as stream_mulaw_bytes_over_twilio)
            if not first:
                next_send += send_interval
                now = time.perf_counter()
                sleep_dur = next_send - now
                if sleep_dur > 0:
                    await asyncio.sleep(sleep_dur)
                elif sleep_dur < -0.03:
                    # We're late by >30ms; reset schedule to avoid cumulative jitter
                    next_send = time.perf_counter()
            else:
                first = False
                next_send = time.perf_counter() + send_interval
    
    async def stream_tts_response(self, text: str):
        """Fast-first TTS with barge-in: cancellable streaming with prefix-first strategy.
        
        Enhanced with sentence-aware chunking for natural pauses.
        """
        try:
            from datetime import datetime, timezone
            
            if not text or not text.strip():
                return
            async with self._tts_lock:
                self._tts_cancel.clear()
                self.is_speaking = True
                try:
                    lang = self.agent.language if self.agent and self.agent.language else "en"
                    voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
                    clean = text.strip()
                    print(f"🎵 Streaming TTS chunk: '{clean[:30]}...'")
                    sys.stdout.flush()

                    # Smart chunking at sentence boundaries (10 words for natural flow)
                    prefix, suffix = smart_chunk_text(clean, max_words=10)

                    # Begin generating suffix in parallel (if any)
                    suffix_task = asyncio.create_task(
                        generate_mulaw_tts(text=suffix, lang=lang, voice=voice, use_chirp3_hd=True, speaking_rate=1.0, add_office_bg=True)
                    ) if suffix else None

                    # Generate prefix audio immediately
                    tts_start = datetime.now(timezone.utc)
                    prefix_audio = await generate_mulaw_tts(text=prefix, lang=lang, voice=voice, use_chirp3_hd=True, speaking_rate=1.0, add_office_bg=True)
                    tts_gen_time = (datetime.now(timezone.utc) - tts_start).total_seconds()
                    print(f"⏱️ TTS(first) latency: {tts_gen_time:.3f}s for '{prefix[:20]}...'")
                    sys.stdout.flush()

                    # Hold back 50ms for crossfade with next chunk (smooth transitions)
                    overlap_bytes = 400  # 50ms at 8kHz
                    if len(prefix_audio) > overlap_bytes:
                        prefix_main = prefix_audio[:-overlap_bytes]
                        prefix_tail = prefix_audio[-overlap_bytes:]
                    else:
                        prefix_main = prefix_audio
                        prefix_tail = b""
                    
                    # Stream main part immediately
                    if prefix_main:
                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=prefix_main,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=1,  # Smooth start with 20ms buffer
                        )

                    # Stream remainder when ready and not cancelled
                    if suffix_task and not self._tts_cancel.is_set():
                        try:
                            suffix_audio = await suffix_task
                        except Exception as e:
                            print(f"⚠️ TTS remainder generation failed: {e}")
                            sys.stdout.flush()
                            suffix_audio = b""
                        
                        if not self._tts_cancel.is_set():
                            if suffix_audio:
                                # Crossfade boundary to eliminate clicks
                                if prefix_tail and len(suffix_audio) > overlap_bytes:
                                    merged = crossfade_mulaw_segments(prefix_tail, suffix_audio, overlap_bytes)
                                else:
                                    merged = (prefix_tail or b"") + suffix_audio
                                
                                await stream_mulaw_bytes_over_twilio(
                                    websocket=self.websocket,
                                    stream_sid=self.stream_sid,
                                    audio_bytes=merged,
                                    pace_20ms=True,
                                    cancel=self._tts_cancel,
                                    prime_frames=0,
                                )
                                print(f"⚡ Streamed remainder with crossfade ({len(merged)} bytes)")
                                sys.stdout.flush()
                            else:
                                # No suffix - flush held tail
                                if prefix_tail:
                                    await stream_mulaw_bytes_over_twilio(
                                        websocket=self.websocket,
                                        stream_sid=self.stream_sid,
                                        audio_bytes=prefix_tail,
                                        pace_20ms=True,
                                        cancel=self._tts_cancel,
                                        prime_frames=0,
                                    )
                finally:
                    self.is_speaking = False
        
        except Exception as e:
            print(f"❌ Error streaming TTS chunk '{text[:20]}...': {e}")
            sys.stdout.flush()
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences for streaming
        NOTE: This function is now deprecated with word-by-word streaming
        Kept for potential fallback or future use
        """
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def send_audio_to_twilio(self, audio_data: bytes):
        """Send audio chunk to Twilio for immediate playback (legacy method)"""
        try:
            # Use new 20ms chunked streaming method
            await stream_mulaw_bytes_over_twilio(
                websocket=self.websocket,
                stream_sid=self.stream_sid,
                audio_bytes=audio_data,
                pace_20ms=True,
            )
            
            print(f"📤 Sent {len(audio_data)} bytes to Twilio (20ms chunks)")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Error sending audio: {e}")
            sys.stdout.flush()
    
    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: Optional[float] = None
    ):
        """Add message to transcript (SSML tags are automatically stripped)"""
        try:
            if not self.call_session:
                return
            
            # Strip SSML tags before saving to transcript (keep only clean text)
            clean_message = strip_ssml_tags(message)
            
            await transcript_service.add_and_broadcast_message(
                db=self.db,
                call_session_id=self.call_session.id,
                role=role,
                message=clean_message,  # Save clean text without SSML
                message_type=message_type,
                agent_id=self.agent.id if self.agent else None,
                user_id=self.call_session.user_id,
                confidence=confidence
            )
            
            # Update legacy field
            conversation = transcript_service.get_conversation_array(self.db, self.call_session.id)
            self.call_session.call_transcript = conversation
            self.db.commit()
        
        except Exception as e:
            print(f"❌ Error adding to transcript: {e}")
            sys.stdout.flush()
    
    async def handle_start_message(self, message: dict):
        """Handle stream start - Just WebSocket connection (NOT user pickup!)"""
        try:
            self.stream_sid = message.get("streamSid")
            start = message.get("start", {})
            self.call_sid = start.get("callSid")
            
            print("=" * 80)
            print(f"🔌 WEBSOCKET CONNECTED")
            print(f"Stream SID: {self.stream_sid}")
            print(f"Call SID: {self.call_sid}")
            print(f"Agent: {self.agent.name if self.agent else 'Unknown'}")
            print(f"⏳ Waiting for user to pick up (first media packet)...")
            print("=" * 80)
            sys.stdout.flush()
            
            # DO NOT start credit monitoring or greeting here!
            # Wait for first media packet (user actually picks up - VAPI-style)
        
        except Exception as e:
            print(f"❌ Error handling start: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    async def _handle_user_pickup(self):
        """Handle user pickup - called on first media packet (VAPI-style)"""
        try:
            if self._user_picked_up:
                return  # Already handled
            
            self._user_picked_up = True
            
            print("=" * 80)
            print(f"🎉 FIRST MEDIA PACKET - USER PICKED UP!")
            print(f"✅ Audio stream active - User actually answered")
            print("=" * 80)
            sys.stdout.flush()
            
            # Update call session status to "in-progress" (user accepted call)
            if self.call_session:
                try:
                    if self.call_session.status != "in-progress":
                        old_status = self.call_session.status
                        self.call_session.status = "in-progress"
                        
                        # Set start time when user actually picks up
                        if not self.call_session.start_time:
                            self.call_session.start_time = datetime.now(timezone.utc)
                        
                        self.db.commit()
                        print(f"✅ Updated DB status: '{old_status}' → 'in-progress' (user picked up)")
                    
                    # Broadcast "in-progress" event (user accepted call)
                    await broadcast_call_status_update(
                        call_session_id=str(self.call_session.id),
                        status="in-progress",
                        metadata={
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "connected",
                            "event": "first_media_packet"
                        }
                    )
                    print(f"✅ Broadcasted 'in-progress' status (user picked up)")
                    
                    # 🎵 START BACKGROUND AUDIO LOOP - User picked up, start streaming
                    if self._use_background_audio and self._bg_audio_mulaw and not self._bg_audio_task:
                        self._bg_audio_task = asyncio.create_task(self._stream_background_audio_loop())
                        print(f"🔄 Started background audio streaming loop")
                        sys.stdout.flush()
                    
                    # 🎯 START CREDIT MONITORING - User actually picked up!
                    try:
                        from app.services.credit_service import CreditService
                        credit_service = CreditService()
                        
                        # Check if monitoring already started (avoid duplicates)
                        if str(self.call_session.id) not in credit_service._active_monitors:
                            asyncio.create_task(credit_service.start_credit_monitoring(
                                db=self.db,
                                call_session_id=self.call_session.id,
                                tenant_id=self.call_session.tenant_id,
                                agent_id=self.call_session.agent_id
                            ))
                            print(f"✅ Started credit monitoring for session {self.call_session.id}")
                            print(f"🔍 DEBUG: Credits will deduct every 30s while status is 'in-progress'")
                        else:
                            print(f"ℹ️ Credit monitoring already active for session {self.call_session.id}")
                    except Exception as e:
                        print(f"❌ Failed to start credit monitoring: {e}")
                        import traceback
                        traceback.print_exc()
                        
                except Exception as e:
                    print(f"❌ Failed to handle user pickup: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ⚠️ AUTO-GREETING DISABLED - Agent waits for user to speak first
            # Auto-greeting disabled - waiting for user to speak first
            print(f"ℹ️ Greeting disabled - waiting for user to speak first")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Error handling user pickup: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    async def handle_stop_message(self, message: dict):
        """Handle stream stop"""
        try:
            print("=" * 80)
            print(f"🛑 BIDIRECTIONAL STREAM STOPPED")
            print(f"Stream SID: {self.stream_sid}")
            print("=" * 80)
            sys.stdout.flush()
            
            # Stop TTS pipeline worker
            try:
                if self._tts_worker_task:
                    await self.tts_queue.put(None)  # Shutdown signal
                    await asyncio.wait_for(self._tts_worker_task, timeout=2.0)
                    print("✅ TTS pipeline worker stopped")
                    sys.stdout.flush()
            except asyncio.TimeoutError:
                print("⚠️ TTS pipeline worker shutdown timeout")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ Error stopping TTS worker: {e}")
            sys.stdout.flush()
            
            # Stop background audio streaming
            try:
                if self._bg_audio_task:
                    self._bg_audio_task.cancel()
                    try:
                        await self._bg_audio_task
                    except asyncio.CancelledError:
                        pass
                    print("✅ Background audio streaming stopped")
                    sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ Error stopping background audio: {e}")
                sys.stdout.flush()
            
            # Close STT session
            try:
                if self._stt_session:
                    self._stt_session.finish()
                if self._stt_task:
                    await asyncio.sleep(0)  # yield
            except Exception:
                pass
        
        except Exception as e:
            print(f"❌ Error handling stop: {e}")
            sys.stdout.flush()


@router.websocket("/ws/bidirectional/{callSessionId}/{agentId}")
async def bidirectional_stream_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    Bidirectional WebSocket for real-time voice AI
    
    Handles:
    - Incoming audio (STT) from Twilio
    - Outgoing audio (TTS) to Twilio
    - Real-time streaming for ultra-low latency
    
    Target: <3 seconds response time
    """
    print("=" * 80)
    print(f"🎙️ Bidirectional WebSocket Connection")
    print(f"Call Session: {callSessionId}")
    print(f"Agent: {agentId}")
    print("=" * 80)
    sys.stdout.flush()
    
    # Accept connection
    try:
        await websocket.accept()
        print(f"✅ WebSocket accepted")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Failed to accept WebSocket: {e}")
        sys.stdout.flush()
        return
    
    # Get database session
    from app.db.session import SessionLocal
    db = SessionLocal()
    
    # Create handler
    handler = BidirectionalStreamHandler(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db
    )
    
    media_count = 0
    
    try:
        while True:
            # Receive message from Twilio
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event = message.get("event")
            
            if event == "connected":
                print("✅ Twilio connected")
                sys.stdout.flush()
            
            elif event == "start":
                await handler.handle_start_message(message)
            
            elif event == "media":
                media_count += 1
                if media_count % 100 == 0:
                    print(f"📦 Processed {media_count} media packets")
                    sys.stdout.flush()
                await handler.handle_media_message(message)
            
            elif event == "stop":
                await handler.handle_stop_message(message)
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected")
        sys.stdout.flush()
    
    except Exception as e:
        print(f"❌ Error in bidirectional stream: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    finally:
        db.close()
        print(f"🔚 Bidirectional stream closed")
        sys.stdout.flush()


@router.websocket("/ws/tts-only/{callSessionId}/{agentId}")
async def tts_only_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    TTS-ONLY WebSocket for streaming audio playback
    
    Used with recording-based STT:
    - Recording callback sends TTS text via custom event
    - WebSocket streams audio in 20ms MULAW chunks
    - No STT handling (recording handles that)
    
    Flow:
    1. Connect to WebSocket
    2. Receive custom {"event": "play_tts", "text": "...", "lang": "en", "voice": "female"}
    3. Generate MULAW TTS
    4. Stream in 20ms chunks
    5. Send {"event": "tts_complete"} when done
    """
    print("=" * 80)
    print(f"🎵 TTS-ONLY WebSocket Connection")
    print(f"Call Session: {callSessionId}")
    print(f"Agent: {agentId}")
    print("=" * 80)
    sys.stdout.flush()
    
    try:
        await websocket.accept()
        print(f"✅ WebSocket accepted (TTS-only mode)")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Failed to accept WebSocket: {e}")
        sys.stdout.flush()
        return
    
    # Get database session
    from app.db.session import SessionLocal
    db = SessionLocal()
    
    # Get agent info for voice settings
    agent = None
    call_session = None
    stream_sid = None
    
    try:
        session_uuid = uuid.UUID(callSessionId)
        call_session = call_session_service.get_call_session_by_id(db, session_uuid)
        
        if call_session and agentId:
            agent_uuid = uuid.UUID(agentId)
            agent = agent_service.get_agent_by_id(db, agent_uuid, call_session.tenant_id)
            if agent:
                print(f"✅ Agent: {agent.name}")
                sys.stdout.flush()
    except Exception as e:
        print(f"⚠️ Error loading agent: {e}")
        sys.stdout.flush()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event = message.get("event")
            
            if event == "connected":
                print("✅ Twilio connected to TTS-only stream")
                sys.stdout.flush()
            
            elif event == "start":
                stream_sid = message.get("streamSid")
                print(f"🎙️ TTS Stream started - SID: {stream_sid}")
                sys.stdout.flush()
                
                # Auto-retrieve and play pending TTS from call session metadata
                if call_session and call_session.call_metadata:
                    pending_tts = call_session.call_metadata.get("pending_tts")
                    if pending_tts:
                        text = pending_tts.get("text", "")
                        lang = pending_tts.get("lang", agent.language if agent else "en")
                        voice = pending_tts.get("voice", agent.voice_type if agent else "female")
                        
                        if text:
                            print(f"🎵 Auto-playing pending TTS: '{text[:50]}...'")
                            sys.stdout.flush()
                            
                            # Generate MULAW TTS with Chirp 3: HD
                            audio_bytes = await generate_mulaw_tts(
                                text=text,
                                lang=lang,
                                voice=voice,
                                use_chirp3_hd=True,
                                speaking_rate=1.0,  # Added for normal speed
                                add_office_bg=True  # Add office background noise
                            )
                            
                            # Stream in 20ms chunks
                            await stream_mulaw_bytes_over_twilio(
                                websocket=websocket,
                                stream_sid=stream_sid,
                                audio_bytes=audio_bytes,
                                pace_20ms=True
                            )
                            
                            # Clear pending TTS
                            call_session.call_metadata.pop("pending_tts", None)
                            db.commit()
                            
                            print(f"✅ Auto-playback complete, cleared pending TTS")
                            sys.stdout.flush()
            
            elif event == "play_tts":
                # Custom event to trigger TTS playback
                text = message.get("text", "")
                lang = message.get("lang", agent.language if agent else "en")
                voice = message.get("voice", agent.voice_type if agent else "female")
                
                if text and stream_sid:
                    print(f"🎵 Playing TTS: '{text[:50]}...'")
                    sys.stdout.flush()
                    
                    # Generate MULAW TTS with Chirp 3: HD
                    audio_bytes = await generate_mulaw_tts(
                        text=text,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,  # Added for normal speed
                        add_office_bg=True  # Add office background noise
                    )
                    
                    # Stream in 20ms chunks
                    await stream_mulaw_bytes_over_twilio(
                        websocket=websocket,
                        stream_sid=stream_sid,
                        audio_bytes=audio_bytes,
                        pace_20ms=True
                    )
                    
                    # Send completion event
                    await websocket.send_json({
                        "event": "tts_complete",
                        "text_length": len(text),
                        "audio_bytes": len(audio_bytes)
                    })
                    print(f"✅ TTS playback complete")
                    sys.stdout.flush()
            
            elif event == "media":
                # Ignore incoming media (we're TTS-only)
                pass
            
            elif event == "stop":
                print("🛑 TTS stream stopped")
                sys.stdout.flush()
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected (TTS-only)")
        sys.stdout.flush()
    
    except Exception as e:
        print(f"❌ Error in TTS-only stream: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    finally:
        db.close()
        print(f"🔚 TTS-only stream closed")
        sys.stdout.flush()