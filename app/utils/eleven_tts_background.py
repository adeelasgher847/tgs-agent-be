"""
ElevenLabs-only: mix 8 kHz mu-law TTS with looping background beds (presets + optional files).

Agent settings (tts_settings_json), all optional for ElevenLabs:
- Omitted / null / empty dict → light stable ``office`` bed (safe default level).
- eleven_background: use ``none``/``off`` to disable; any other preset maps to ``office``.
- eleven_background_level: linear bed gain; ``<= 0`` with bed on → default level.

Mixing formula uses voice headroom to prevent clipping:
  mixed = int(voice * VOICE_HEADROOM) + int(background * level)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from app.utils.audio_utils import (
    MULAW_FRAME_BYTES,
    MULAW_SAMPLE_RATE_HZ,
    linear_to_ulaw_sample,
    linear_samples_to_ulaw_bytes,
    ulaw_to_linear_sample,
)

DEFAULT_ELEVEN_BACKGROUND_PRESET = "office"
DEFAULT_ELEVEN_BACKGROUND_LEVEL = 0.14
MAX_ELEVEN_BACKGROUND_LEVEL = 0.24

# Do NOT scale voice — keep it full-amplitude.  Background is already quiet
# enough (gain << 32767) that the sum never clips.  Attenuating the voice
# at 0.85 makes it sound thin/distorted, which is worse than rare clipping.
VOICE_HEADROOM = 0.87

# Display catalog (API). Runtime mixing is intentionally pinned to "office"
# for stable telephony quality.
ELEVEN_BACKGROUND_CATALOG: list[dict[str, str]] = [
    {
        "id": "soft_noise",
        "label": "Soft noise",
        "description": "Gentle bed; low distraction.",
    },
    {
        "id": "office",
        "label": "Office ambience",
        "description": "Subtle room tone / HVAC-style bed.",
    },
    {
        "id": "cafe",
        "label": "Cafe ambience",
        "description": "Busier mid-frequency bed.",
    },
    {
        "id": "outdoor",
        "label": "Outdoor / breeze",
        "description": "Air movement–style bed.",
    },
]

_VALID_IDS = {entry["id"] for entry in ELEVEN_BACKGROUND_CATALOG}

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources" / "eleven_tts_backgrounds"

_loop_cache: dict[str, bytes] = {}
_linear_loop_cache: dict[str, list[int]] = {}


def list_eleven_background_catalog() -> list[dict[str, str]]:
    """Static presets for UI (ElevenLabs TTS only)."""
    return list(ELEVEN_BACKGROUND_CATALOG)


def _coerce_positive_bed_level(level: float) -> float:
    """If the bed is enabled but level is zero/negative, use the safe default."""
    if level <= 0.0:
        return DEFAULT_ELEVEN_BACKGROUND_LEVEL
    return level


def parse_eleven_background_settings(
    settings_json: Optional[dict[str, Any]],
) -> tuple[Optional[str], float]:
    """
    Returns (background_id, clamped_level).

    When ``eleven_background`` is omitted (including ``None`` or empty ``{}`` JSON),
    ElevenLabs calls get a light stable ``office`` bed — no JSON required.

    ``eleven_background_level`` <= 0 with the bed enabled is treated as "use
    default" so mistaken zeros never mute the ambience.

    Returns (None, level) only when ``eleven_background`` explicitly disables
    via ``none`` / ``off`` / ``false`` / ``0``.
    """
    d = dict(settings_json or {})
    level_raw = d.get("eleven_background_level", DEFAULT_ELEVEN_BACKGROUND_LEVEL)
    try:
        level = float(level_raw)
    except (TypeError, ValueError):
        level = DEFAULT_ELEVEN_BACKGROUND_LEVEL
    level = max(0.0, min(MAX_ELEVEN_BACKGROUND_LEVEL, level))

    raw = d.get("eleven_background")
    if raw is None:
        return DEFAULT_ELEVEN_BACKGROUND_PRESET, _coerce_positive_bed_level(level)

    key = str(raw).strip().lower()
    if key in ("", "none", "off", "false", "0"):
        return None, level

    if key not in _VALID_IDS:
        return DEFAULT_ELEVEN_BACKGROUND_PRESET, _coerce_positive_bed_level(level)

    # Production hardening: use the validated office loop only.
    if key != "office":
        return DEFAULT_ELEVEN_BACKGROUND_PRESET, _coerce_positive_bed_level(level)
    return key, _coerce_positive_bed_level(level)


def cache_key_background_fragment(settings_json: Optional[dict[str, Any]]) -> str:
    """Suffix for TTS cache keys when ElevenLabs background may apply."""
    bg_id, level = parse_eleven_background_settings(settings_json)
    if not bg_id:
        return ""
    return f"_ebg:{bg_id}:{round(level, 3)}"


def _pinkish_sample(state: list[float], rng: random.Random) -> float:
    """Cheap pink-ish noise (cascaded leaky integrators on white noise)."""
    white = rng.uniform(-1.0, 1.0)
    state[0] = 0.997 * state[0] + 0.12 * white
    state[1] = 0.985 * state[1] + 0.10 * state[0]
    state[2] = 0.97 * state[2] + 0.08 * state[1]
    return state[2]


def _synthetic_loop_mulaw(preset_id: str) -> bytes:
    """
    8-second seamless looping 8 kHz mu-law background bed.

    Design choices that eliminate distortion:
    - 8 s duration (64 000 samples) → loop boundary click is very rare and quiet
    - Filter runs 2000 warm-up samples before recording → avoids the initial
      ramp-up discontinuity at the very start of the loop
    - Linear PCM is generated first; boundary crossfade is applied before
      mu-law encoding → no click at loop wrap-around
    - Low gains → background stays well below voice level and mu-law
      quantisation noise is inaudible at telephony SNR
    """
    LOOP_SECONDS = 8
    CROSSFADE_SAMPLES = MULAW_FRAME_BYTES  # 160 samples = 20 ms
    WARMUP = 2_000                         # discard warm-up samples

    n = MULAW_SAMPLE_RATE_HZ * LOOP_SECONDS  # 64 000 samples

    seed = (sum(ord(c) for c in preset_id) * 1103515245 + 12345) & 0x7FFFFFFF
    rng = random.Random(seed)
    state: list[float] = [0.0, 0.0, 0.0]
    slow = 0.0

    # Conservative gains: background is audible but stays well below voice
    # amplitude.  At level=0.15, peak background contribution ≈ 450 linear
    # (≈ 1.4 % of 32767 full scale = ~-37 dB relative to loud voice).
    gains = {
        "soft_noise": 1_200.0,
        "office": 2_000.0,
        "cafe": 2_800.0,
        "outdoor": 2_200.0,
    }
    gain = gains.get(preset_id, 1_500.0)

    # --- Warm-up: let filter settle without recording ---
    for _ in range(WARMUP):
        _pinkish_sample(state, rng)

    # --- Record linear PCM samples ---
    pcm: list[int] = []
    for i in range(n):
        p = _pinkish_sample(state, rng)
        if preset_id == "office":
            p += 0.06 * slow
            slow = 0.995 * slow + rng.uniform(-0.01, 0.01)
        elif preset_id == "cafe":
            # Softer high-freq flutter instead of raw white noise
            p += 0.10 * _pinkish_sample([state[0], state[1], state[2]], rng)
        elif preset_id == "outdoor":
            gust = 0.18 * rng.gauss(0.0, 0.5)
            p = 0.8 * p + gust
        pcm.append(int(max(-32768, min(32767, p * gain))))

    # --- Seamless loop: crossfade last CROSSFADE_SAMPLES into first ---
    for j in range(CROSSFADE_SAMPLES):
        t = j / CROSSFADE_SAMPLES         # 0 → 1
        head_j = pcm[j]
        tail_j = pcm[n - CROSSFADE_SAMPLES + j]
        # Blend tail out / head in so the loop wraps without a click
        pcm[j] = int(head_j * t + tail_j * (1.0 - t))

    # --- Encode to mu-law ---
    return bytes(linear_to_ulaw_sample(s) for s in pcm)


def _load_loop_from_file(preset_id: str) -> Optional[bytes]:
    path = _RESOURCES_DIR / f"{preset_id}.ulaw"
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) < MULAW_FRAME_BYTES:
        return None
    return data


def get_background_loop_bytes(preset_id: str) -> bytes:
    if preset_id not in _VALID_IDS:
        raise ValueError(f"Unknown ElevenLabs background preset: {preset_id}")
    loaded = _load_loop_from_file(preset_id)
    if loaded is not None:
        _loop_cache[preset_id] = loaded
        _linear_loop_cache.pop(preset_id, None)
        return loaded
    if preset_id in _loop_cache:
        return _loop_cache[preset_id]
    synthetic = _synthetic_loop_mulaw(preset_id)
    _loop_cache[preset_id] = synthetic
    return synthetic


def get_background_loop_linear_samples(preset_id: str) -> list[int]:
    """
    Return the background loop decoded to 8 kHz linear PCM samples.

    Cached separately so repeated mixing does not keep ulaw-decoding the same
    ambience loop over and over.
    """
    if preset_id not in _VALID_IDS:
        raise ValueError(f"Unknown ElevenLabs background preset: {preset_id}")
    loaded = _load_loop_from_file(preset_id)
    if loaded is not None:
        _loop_cache[preset_id] = loaded
        linear = [ulaw_to_linear_sample(b) for b in loaded]
        _linear_loop_cache[preset_id] = linear
        return linear
    if preset_id in _linear_loop_cache:
        return _linear_loop_cache[preset_id]
    loop = get_background_loop_bytes(preset_id)
    linear = [ulaw_to_linear_sample(b) for b in loop]
    _linear_loop_cache[preset_id] = linear
    return linear


class LinearBackgroundMixer:
    """Stateful linear-domain mixer that outputs final mu-law bytes."""

    __slots__ = ("_loop", "_level", "_pos")

    def __init__(self, background_id: str, level: float):
        self._loop = get_background_loop_linear_samples(background_id)
        self._level = max(0.0, min(MAX_ELEVEN_BACKGROUND_LEVEL, float(level)))
        self._pos = 0

    def mix_linear_samples_to_ulaw(self, voice_samples: list[int]) -> bytes:
        """
        Mix 8 kHz linear PCM voice samples with the background loop and encode
        the final result to mu-law exactly once.
        """
        if self._level <= 0.0 or not voice_samples:
            return linear_samples_to_ulaw_bytes(voice_samples)
        lim = len(self._loop)
        mixed: list[int] = []
        for voice_sample in voice_samples:
            bg = self._loop[self._pos]
            self._pos = (self._pos + 1) % lim
            sample = int(voice_sample * VOICE_HEADROOM) + int(bg * self._level)
            mixed.append(max(-32768, min(32767, sample)))
        return linear_samples_to_ulaw_bytes(mixed)


def mix_mulaw_bytes(voice: bytes, background_id: str, level: float) -> bytes:
    """Mix full mu-law buffer with looping background (byte-aligned 8 kHz).

    Voice is scaled by VOICE_HEADROOM (<1) to reduce clipping with the bed.
    Background gain
    and level are both small enough that the sum never clips in practice.
    """
    if not voice or level <= 0.0:
        return voice
    loop = get_background_loop_linear_samples(background_id)
    if not loop:
        return voice
    lim = len(loop)
    out = bytearray(len(voice))
    pos = 0
    for i, vb in enumerate(voice):
        v = ulaw_to_linear_sample(vb)
        b = loop[pos]
        pos = (pos + 1) % lim
        m = int(v * VOICE_HEADROOM) + int(b * level)
        m = max(-32768, min(32767, m))
        out[i] = linear_to_ulaw_sample(m)
    return bytes(out)


class BackgroundFrameMixer:
    """Stateful mixer for streaming: keeps phase across HTTP chunks."""

    __slots__ = ("_loop", "_level", "_pos")

    def __init__(self, background_id: str, level: float):
        self._loop = get_background_loop_linear_samples(background_id)
        self._level = max(0.0, min(MAX_ELEVEN_BACKGROUND_LEVEL, float(level)))
        self._pos = 0

    def mix_frame(self, frame: bytes) -> bytes:
        """Mix one 20ms mu-law frame with background.

        Voice is kept at full amplitude; background is added quietly on top.
        Safe to call on both TTS speech frames and mu-law silence (0xFF) frames.
        """
        if self._level <= 0.0 or not frame:
            return frame
        lim = len(self._loop)
        out = bytearray(len(frame))
        for i, vb in enumerate(frame):
            v = ulaw_to_linear_sample(vb)
            b = self._loop[self._pos]
            self._pos = (self._pos + 1) % lim
            m = int(v * VOICE_HEADROOM) + int(b * self._level)
            m = max(-32768, min(32767, m))
            out[i] = linear_to_ulaw_sample(m)
        return bytes(out)
