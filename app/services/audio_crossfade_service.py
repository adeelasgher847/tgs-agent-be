"""
Ultra-Smooth Audio Crossfade Service  
Eliminates ALL clicks/pops in streaming TTS audio
Aggressive overlap + silence padding fallback
"""

import numpy as np
from typing import Optional

class AudioCrossfadeService:
    def __init__(self, sample_rate: int = 8000, default_overlap_ms: int = 200):
        """
        Args:
            sample_rate: Audio sample rate in Hz (8000 for MULAW)
            default_overlap_ms: Overlap in milliseconds (200ms = ultra-smooth)
        """
        self.sample_rate = sample_rate
        self.default_overlap_samples = int((default_overlap_ms / 1000.0) * sample_rate)
        self._previous_tail: Optional[np.ndarray] = None
        self._chunk_count = 0
        self._silence_padding = int(0.01 * sample_rate)  # 10ms silence fallback

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
        """Process with MAXIMUM smoothing - eliminates ALL clicks."""
        if not mulaw_bytes:
            return b''

        current = self._mulaw_to_linear(mulaw_bytes)
        self._chunk_count += 1

        # First chunk: aggressive fade-in + edge smoothing
        if is_first or self._previous_tail is None:
            # Fade in first 100ms (removes any initial artifact)
            fade_len = min(int(0.1 * self.sample_rate), len(current))
            fade_curve = self._create_fade_curve(fade_len, fade_type="in")
            current[:fade_len] *= fade_curve
            
            # Fade out last 50ms for smooth transition
            fade_out_len = min(int(0.05 * self.sample_rate), len(current))
            fade_out_curve = self._create_fade_curve(fade_out_len, fade_type="out")
            current[-fade_out_len:] *= fade_out_curve
            
            # Add silence padding at start (Vapi-style)
            silence = np.zeros(self._silence_padding, dtype=np.float32)
            current = np.concatenate([silence, current])
            
            self._previous_tail = current[-self.default_overlap_samples:] if len(current) > self.default_overlap_samples else current
            return self._linear_to_mulaw(current)

        # Maximum overlap for subsequent chunks
        overlap_len = min(len(self._previous_tail), len(current), self.default_overlap_samples)
        
        # Ensure minimum 150ms overlap
        min_overlap = min(int(0.15 * self.sample_rate), len(current) // 2)
        overlap_len = max(overlap_len, min_overlap)
        overlap_len = min(overlap_len, len(self._previous_tail), len(current))
        
        if overlap_len < 20:  # Too short - add silence padding instead
            silence = np.zeros(self._silence_padding, dtype=np.float32)
            self._previous_tail = current[-self.default_overlap_samples:] if len(current) > self.default_overlap_samples else current
            return self._linear_to_mulaw(np.concatenate([silence, current]))

        prev_tail = self._previous_tail[-overlap_len:]
        curr_head = current[:overlap_len]

        # Triple-smoothed crossfade (maximum smoothness)
        fade_in = self._create_fade_curve(overlap_len, fade_type="in")
        fade_out = self._create_fade_curve(overlap_len, fade_type="out")
        
        # Apply extra smoothing (power curve + normalization)
        fade_in = np.power(fade_in, 0.5)  # Very soft curve
        fade_out = np.power(fade_out, 0.5)
        
        # Normalize to prevent volume dips
        fade_sum = fade_in + fade_out
        fade_in = fade_in / fade_sum
        fade_out = fade_out / fade_sum
        
        crossfaded = (prev_tail * fade_out) + (curr_head * fade_in)
        
        # Smooth the entire remaining audio
        remaining = current[overlap_len:]
        if len(remaining) > 40:
            # Fade out last 50ms for next chunk
            tail_fade_len = min(int(0.05 * self.sample_rate), len(remaining))
            tail_fade = self._create_fade_curve(tail_fade_len, fade_type="out")
            remaining[-tail_fade_len:] *= tail_fade

        output = np.concatenate([crossfaded, remaining])

        # Save tail for next chunk
        self._previous_tail = current[-self.default_overlap_samples:] if len(current) > self.default_overlap_samples else current

        return self._linear_to_mulaw(output)

    def reset(self):
        """Reset state for new audio stream."""
        self._previous_tail = None
        self._chunk_count = 0