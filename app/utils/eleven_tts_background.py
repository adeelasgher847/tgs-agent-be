"""
ElevenLabs-only: mix 8 kHz mu-law TTS with looping background beds (presets + optional files).

Agent settings (tts_settings_json):
- eleven_background: preset id or "none"/"off" to disable
- eleven_background_level: linear gain on background before sum (0.0–0.35, default 0.2)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from app.utils.audio_utils import (
    MULAW_FRAME_BYTES,
    MULAW_SAMPLE_RATE_HZ,
    linear_to_ulaw_sample,
    ulaw_to_linear_sample,
)

DEFAULT_ELEVEN_BACKGROUND_LEVEL = 0.2
MAX_ELEVEN_BACKGROUND_LEVEL = 0.35

# Display catalog (API). File override: app/resources/eleven_tts_backgrounds/{id}.ulaw
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


def list_eleven_background_catalog() -> list[dict[str, str]]:
    """Static presets for UI (ElevenLabs TTS only)."""
    return list(ELEVEN_BACKGROUND_CATALOG)


def parse_eleven_background_settings(
    settings_json: Optional[dict[str, Any]],
) -> tuple[Optional[str], float]:
    """
    Returns (background_id or None if off/invalid, clamped level).
    """
    if not settings_json:
        return None, DEFAULT_ELEVEN_BACKGROUND_LEVEL

    raw = settings_json.get("eleven_background")
    if raw is None:
        return None, DEFAULT_ELEVEN_BACKGROUND_LEVEL
    key = str(raw).strip().lower()
    if key in ("", "none", "off", "false", "0"):
        return None, DEFAULT_ELEVEN_BACKGROUND_LEVEL

    level_raw = settings_json.get("eleven_background_level", DEFAULT_ELEVEN_BACKGROUND_LEVEL)
    try:
        level = float(level_raw)
    except (TypeError, ValueError):
        level = DEFAULT_ELEVEN_BACKGROUND_LEVEL
    level = max(0.0, min(MAX_ELEVEN_BACKGROUND_LEVEL, level))

    if key not in _VALID_IDS:
        return None, DEFAULT_ELEVEN_BACKGROUND_LEVEL
    return key, level


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
    """One second of looping 8 kHz mu-law (deterministic per preset)."""
    n = MULAW_SAMPLE_RATE_HZ
    seed = (sum(ord(c) for c in preset_id) * 1103515245 + 12345) & 0x7FFFFFFF
    rng = random.Random(seed)
    state: list[float] = [0.0, 0.0, 0.0]
    slow = 0.0
    out = bytearray()

    gains = {
        "soft_noise": 900.0,
        "office": 650.0,
        "cafe": 1100.0,
        "outdoor": 800.0,
    }
    gain = gains.get(preset_id, 800.0)

    for i in range(n):
        p = _pinkish_sample(state, rng)
        if preset_id == "office":
            p += 0.12 * slow
            slow = 0.995 * slow + rng.uniform(-0.02, 0.02)
        elif preset_id == "cafe":
            p += 0.25 * rng.uniform(-1.0, 1.0)
        elif preset_id == "outdoor":
            gust = 0.35 * (0.5 + 0.5 * (i / n)) * rng.uniform(-1.0, 1.0)
            p = 0.7 * p + gust
        lin = int(max(-32768, min(32767, p * gain)))
        out.append(linear_to_ulaw_sample(lin))

    return bytes(out)


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
    if preset_id in _loop_cache:
        return _loop_cache[preset_id]
    loaded = _load_loop_from_file(preset_id)
    if loaded is not None:
        _loop_cache[preset_id] = loaded
        return loaded
    synthetic = _synthetic_loop_mulaw(preset_id)
    _loop_cache[preset_id] = synthetic
    return synthetic


def mix_mulaw_bytes(voice: bytes, background_id: str, level: float) -> bytes:
    """Mix full mu-law buffer with looping background (byte-aligned 8 kHz)."""
    if not voice or level <= 0.0:
        return voice
    loop = get_background_loop_bytes(background_id)
    if not loop:
        return voice
    lim = len(loop)
    out = bytearray(len(voice))
    pos = 0
    for i, vb in enumerate(voice):
        v = ulaw_to_linear_sample(vb)
        b = ulaw_to_linear_sample(loop[pos])
        pos = (pos + 1) % lim
        m = v + int(b * level)
        m = max(-32768, min(32767, m))
        out[i] = linear_to_ulaw_sample(m)
    return bytes(out)


class BackgroundFrameMixer:
    """Stateful mixer for streaming: keeps phase across HTTP chunks."""

    __slots__ = ("_loop", "_level", "_pos")

    def __init__(self, background_id: str, level: float):
        self._loop = get_background_loop_bytes(background_id)
        self._level = max(0.0, min(MAX_ELEVEN_BACKGROUND_LEVEL, float(level)))
        self._pos = 0

    def mix_frame(self, frame: bytes) -> bytes:
        if self._level <= 0.0 or not frame:
            return frame
        lim = len(self._loop)
        out = bytearray(len(frame))
        for i, vb in enumerate(frame):
            v = ulaw_to_linear_sample(vb)
            b = ulaw_to_linear_sample(self._loop[self._pos])
            self._pos = (self._pos + 1) % lim
            m = v + int(b * self._level)
            m = max(-32768, min(32767, m))
            out[i] = linear_to_ulaw_sample(m)
        return bytes(out)
