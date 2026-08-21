"""
Pure audio-conversion helpers bridging Twilio's MULAW/8kHz telephony audio
to/from Gemini Live's required PCM16 format (16kHz mono in, 24kHz mono out —
confirmed live against the real API, see ``app/core/llm_models.py``'s
``GEMINI_LIVE_MODELS`` docstring).

This module has ZERO knowledge of Twilio/LiveKit/call-session state — it
only converts byte buffers. Functions here are intentionally synchronous,
pure, and side-effect-free (no async, no I/O, no class state) so they're
trivially unit-testable and safe to call from any transport's hot audio
path without blocking an event loop.

Design notes / tradeoffs
-------------------------
- **mu-law codec**: reuses this repo's existing pure-Python
  ``ulaw_to_linear_sample`` / ``linear_samples_to_ulaw_bytes`` helpers
  (``app/utils/audio_utils.py``), already used by ``livekit_browser_call_handler.py``'s
  ``_LiveKitAgentAudioPublisher.publish_mulaw`` and the TTS mixing/fade code
  in the same module. Considered stdlib ``audioop``/``audioop-lts``
  (``audioop.ulaw2lin``/``lin2ulaw``) instead — it would be faster (C
  implementation) and is available in this repo's Docker builder
  (``python:3.11-slim``, confirmed via ``python3 --version`` -> 3.11.2, so
  stdlib ``audioop`` has not yet been removed as of 3.13). But ``audioop``
  is deprecated (removal scheduled for a future CPython) and, per a repo
  grep, is not used anywhere else in this codebase, which otherwise
  consistently favors the pure-Python codec in ``audio_utils.py`` for MULAW
  math. Reusing that existing, already-tested helper avoids introducing a
  second mu-law implementation and a stdlib-deprecation dependency for a
  small, non-bottleneck buffer size (20ms Twilio frames are only 160 bytes).
  If this ever becomes a measured hot-path bottleneck, swapping the per-byte
  loop below for ``audioop.ulaw2lin``/``lin2ulaw`` is a drop-in, isolated
  change confined to this module.
- **Resampling (8k<->16k, 24k->8k)**: ``app/voice/audio_transcoder.py``'s
  ``LiveKitAudioProcessor`` is a generic PCM ffmpeg-subprocess resampler,
  but it's built as a persistent per-call subprocess with async reader/
  writer loops — the wrong shape for a stateless, synchronous, per-buffer
  helper (spawning a subprocess per call would dwarf the actual conversion
  cost and add I/O to a function this module promises to keep pure/sync).
  ``audioop.ratecv`` was considered as a synchronous alternative but rejected
  for the same audioop-avoidance reason as the mu-law codec above.  Instead:
  - **Downsampling** (24kHz Gemini output -> 8kHz Twilio) reuses this
    repo's existing ``downsample_linear_samples`` (box averaging over
    integer factors) from ``audio_utils.py`` — already used elsewhere in
    this codebase for 16k->8k background-audio mixing, so its quality
    tradeoff (removes enough high-frequency energy to avoid harsh aliasing
    on phone-quality audio, per that function's own docstring) is
    already-accepted precedent, not a new one.
  - **Upsampling** (8kHz Twilio input -> 16kHz Gemini input) has no existing
    helper in this repo (``downsample_linear_samples`` only handles
    downsampling), so this module adds ``_upsample_linear_samples`` — simple
    linear interpolation between consecutive samples, doubling sample rate
    for the 8k->16k case. This is a well-known low-cost/reasonable-quality
    resampling method (linear interpolation), not brick-wall/sinc-filtered,
    so some high-frequency imaging artifacts are possible; but the target
    is *speech input to an ASR/LLM model*, not audio played back to a human,
    so any color introduced here has never been shown to affect this
    model's transcription/understanding quality in Google's own docs, and
    the intent is caller audio en route to Gemini, not Gemini's own more
    audio-quality-sensitive playback direction (see downsampling above,
    which handles that direction and reuses the already-vetted helper).
"""
from __future__ import annotations

from app.utils.audio_utils import (
    downsample_linear_samples,
    linear_samples_to_ulaw_bytes,
    pcm16le_bytes_to_linear_samples,
    ulaw_to_linear_sample,
)

# Sample rates involved (confirmed live against the real API, 2026-08-17 —
# see app/core/llm_models.py's GEMINI_LIVE_MODELS docstring for the source
# of truth on these numbers).
TWILIO_MULAW_RATE_HZ = 8000
GEMINI_INPUT_PCM_RATE_HZ = 16000
GEMINI_OUTPUT_PCM_RATE_HZ = 24000


def _upsample_linear_samples(samples: list[int], factor: int) -> list[int]:
    """
    Upsample linear PCM samples by an integer ``factor`` using simple linear
    interpolation between consecutive input samples. See module docstring
    for the quality tradeoff rationale (speech-input direction, not
    playback-to-human direction).

    ``factor`` must be a positive integer. ``factor == 1`` (or an empty
    input) returns a shallow copy unchanged.
    """
    if not samples or factor <= 1:
        return list(samples)

    n = len(samples)
    out: list[int] = []
    for i in range(n):
        a = samples[i]
        b = samples[i + 1] if i + 1 < n else a
        for step in range(factor):
            t = step / factor
            out.append(int(round(a + (b - a) * t)))
    return out


def _linear_samples_to_pcm16le_bytes(samples: list[int]) -> bytes:
    """Encode signed linear PCM samples to little-endian 16-bit PCM bytes,
    clamping to the valid int16 range (defensive — interpolation/averaging
    above should already stay in range, but never emit out-of-range bytes)."""
    out = bytearray(len(samples) * 2)
    for i, sample in enumerate(samples):
        clamped = max(-32768, min(32767, int(sample)))
        out[i * 2:i * 2 + 2] = int(clamped & 0xFFFF).to_bytes(2, "little", signed=False)
    return bytes(out)


def mulaw8k_to_pcm16_16k(mulaw_bytes: bytes) -> bytes:
    """
    Convert Twilio's inbound MULAW 8kHz mono caller audio to the raw PCM16
    16kHz little-endian mono format Gemini Live's ``send_realtime_input``
    expects (``audio/pcm;rate=16000``).

    Pure/synchronous — safe to call from any hot audio-receive path.
    Empty input returns empty output. Odd-length input is not meaningful for
    MULAW (1 byte == 1 sample) so every byte is processed; there is no
    "odd/even byte pairing" concern on the input side (unlike PCM16, see
    ``pcm16_24k_to_mulaw8k``).
    """
    if not mulaw_bytes:
        return b""
    linear_8k = [ulaw_to_linear_sample(b) for b in mulaw_bytes]
    linear_16k = _upsample_linear_samples(linear_8k, GEMINI_INPUT_PCM_RATE_HZ // TWILIO_MULAW_RATE_HZ)
    return _linear_samples_to_pcm16le_bytes(linear_16k)


def pcm16_24k_to_mulaw8k(pcm_bytes: bytes) -> bytes:
    """
    Convert Gemini Live's outbound raw PCM16 24kHz little-endian mono audio
    (``audio/pcm;rate=24000``) to Twilio's outbound MULAW 8kHz mono format.

    Pure/synchronous — safe to call from any hot audio-send path. Empty
    input returns empty output. A trailing odd byte (incomplete final
    sample) is silently dropped rather than raising — mirrors
    ``pcm16le_bytes_to_linear_samples``'s existing odd-length handling in
    ``audio_utils.py``.
    """
    if not pcm_bytes:
        return b""
    linear_24k = pcm16le_bytes_to_linear_samples(pcm_bytes)
    if not linear_24k:
        return b""
    linear_8k = downsample_linear_samples(
        linear_24k, GEMINI_OUTPUT_PCM_RATE_HZ, TWILIO_MULAW_RATE_HZ
    )
    return linear_samples_to_ulaw_bytes(linear_8k)
