"""
Audio Crossfade Service - Eliminates clicks/pops in streaming audio
Simple and efficient crossfading for 8kHz MULAW audio chunks
"""

import numpy as np
from typing import Optional


class AudioCrossfadeService:
    """
    Applies smooth crossfade between consecutive audio chunks.
    Eliminates clicks/pops/taks in streaming TTS audio.
    """
    
    def __init__(self, overlap_ms: int = 75, sample_rate: int = 8000):
        """
        Args:
            overlap_ms: Overlap duration in milliseconds (75ms recommended)
            sample_rate: Audio sample rate in Hz (8000 for MULAW)
        """
        self.overlap_samples = int((overlap_ms / 1000.0) * sample_rate)
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
    
    def process_chunk(self, mulaw_bytes: bytes, is_first: bool = False) -> bytes:
        """
        Process audio chunk with crossfading.
        
        Args:
            mulaw_bytes: Raw mu-law audio bytes
            is_first: True for first chunk (no crossfade)
            
        Returns:
            Processed mu-law bytes with crossfade applied
        """
        if not mulaw_bytes:
            return b''
        
        current = self._mulaw_to_linear(mulaw_bytes)
        
        if is_first or self._previous_tail is None:
            self._previous_tail = current[-self.overlap_samples:] if len(current) > self.overlap_samples else current
            return mulaw_bytes
        
        overlap_len = min(len(self._previous_tail), len(current), self.overlap_samples)
        if overlap_len == 0:
            self._previous_tail = current[-self.overlap_samples:] if len(current) > self.overlap_samples else current
            return mulaw_bytes
        
        prev_tail = self._previous_tail[-overlap_len:]
        curr_head = current[:overlap_len]
        
        # Cosine crossfade (smoothest)
        x = np.linspace(0, 1, overlap_len)
        fade_in = 0.5 * (1 - np.cos(np.pi * x))
        fade_out = 1.0 - fade_in
        
        crossfaded = (prev_tail * fade_out) + (curr_head * fade_in)
        output = np.concatenate([crossfaded, current[overlap_len:]])
        
        self._previous_tail = current[-self.overlap_samples:] if len(current) > self.overlap_samples else current
        
        return self._linear_to_mulaw(output)
    
    def reset(self):
        """Reset state for new audio stream."""
        self._previous_tail = None

