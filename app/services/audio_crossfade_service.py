"""
Optimized Audio Crossfade Service
Eliminates clicks/pops in streaming TTS audio
Dynamic overlap + first-chunk micro-fade for low latency
"""

import numpy as np
from typing import Optional

class AudioCrossfadeService:
    def __init__(self, sample_rate: int = 8000, default_overlap_ms: int = 50):
        """
        Args:
            sample_rate: Audio sample rate in Hz (8000 for MULAW)
            default_overlap_ms: Default overlap in milliseconds
        """
        self.sample_rate = sample_rate
        self.default_overlap_samples = int((default_overlap_ms / 1000.0) * sample_rate)
        self._previous_tail: Optional[np.ndarray] = None

    def _mulaw_to_linear(self, mulaw_bytes: bytes) -> np.ndarray:
        """Convert mu-law bytes to linear PCM."""
        mulaw_data = np.frombuffer(mulaw_bytes, dtype=np.uint8).astype(np.int16)
        mulaw_data = mulaw_data ^ 0xFF
        sign = (mulaw_data >> 7) & 0x01
        exponent = (mulaw_data >> 4) & 0x07
        mantissa = mulaw_data & 0x0F
        linear = (mantissa * 2 + 33) * (2 ** exponent)
        linear = np.where(sign == 0, linear, -linear)
        return linear.astype(np.float32) / 32768.0

    def _linear_to_mulaw(self, linear_samples: np.ndarray) -> bytes:
        """Convert linear PCM to mu-law bytes."""
        linear_int = (linear_samples * 32768.0).astype(np.int16)
        sign = (linear_int < 0).astype(np.uint8)
        linear_abs = np.abs(linear_int).astype(np.int16) + 33
        linear_abs = np.clip(linear_abs, 0, 0x1FFF)
        exponent = np.zeros_like(linear_abs, dtype=np.uint8)
        for i in range(7, -1, -1):
            mask = 1 << (i + 5)
            exponent = np.where((linear_abs >= mask) & (exponent == 0), 7 - i, exponent)
        mantissa = (linear_abs >> (exponent + 4)) & 0x0F
        mulaw = (sign << 7) | (exponent << 4) | mantissa
        mulaw = mulaw ^ 0xFF
        return mulaw.astype(np.uint8).tobytes()

    def _create_fade_curve(self, length: int, fade_type: str = "in") -> np.ndarray:
        """Create smooth fade curve (cosine)."""
        x = np.linspace(0, 1, length)
        curve = 0.5 * (1 - np.cos(np.pi * x))
        if fade_type == "out":
            curve = 1.0 - curve
        return curve

    def process_chunk(self, mulaw_bytes: bytes, is_first: bool = False) -> bytes:
        """Process audio chunk with optimized crossfade."""
        if not mulaw_bytes:
            return b''

        current = self._mulaw_to_linear(mulaw_bytes)

        # First chunk: apply micro-fade-in to remove initial click
        if is_first or self._previous_tail is None:
            fade_len = min(10, len(current))  # 10ms fade-in
            fade_curve = np.linspace(0, 1, fade_len)
            current[:fade_len] *= fade_curve
            self._previous_tail = current[-self.default_overlap_samples:] if len(current) > self.default_overlap_samples else current
            return self._linear_to_mulaw(current)

        # Determine dynamic overlap (shorter if chunk small)
        overlap_len = min(len(self._previous_tail), len(current), self.default_overlap_samples)
        if overlap_len == 0:
            self._previous_tail = current[-self.default_overlap_samples:] if len(current) > self.default_overlap_samples else current
            return mulaw_bytes

        prev_tail = self._previous_tail[-overlap_len:]
        curr_head = current[:overlap_len]

        # Cosine crossfade
        fade_in = self._create_fade_curve(overlap_len, fade_type="in")
        fade_out = self._create_fade_curve(overlap_len, fade_type="out")
        crossfaded = (prev_tail * fade_out) + (curr_head * fade_in)

        output = np.concatenate([crossfaded, current[overlap_len:]])

        # Update tail for next chunk
        self._previous_tail = current[-self.default_overlap_samples:] if len(current) > self.default_overlap_samples else current

        return self._linear_to_mulaw(output)

    def reset(self):
        """Reset state for new audio stream."""
        self._previous_tail = None