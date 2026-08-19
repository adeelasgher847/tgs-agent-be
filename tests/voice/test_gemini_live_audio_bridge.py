"""
Unit tests for app/voice/gemini_live_audio_bridge.py — pure audio-conversion
functions bridging Twilio's MULAW/8kHz telephony audio to/from Gemini
Live's PCM16 16kHz-in/24kHz-out requirement.

All functions under test are synchronous and pure — no mocking, no I/O,
no google-genai SDK involvement at all.
"""
from __future__ import annotations

import math
import struct

import pytest

from app.utils.audio_utils import linear_to_ulaw_sample, ulaw_to_linear_sample
from app.voice.gemini_live_audio_bridge import (
    GEMINI_INPUT_PCM_RATE_HZ,
    GEMINI_OUTPUT_PCM_RATE_HZ,
    TWILIO_MULAW_RATE_HZ,
    _linear_samples_to_pcm16le_bytes,
    _upsample_linear_samples,
    mulaw8k_to_pcm16_16k,
    pcm16_24k_to_mulaw8k,
)


def _pcm16le_bytes(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _tone_samples(n: int, amplitude: int = 8000, period: int = 20) -> list[int]:
    """Deterministic non-silent waveform (not a pure DC signal) for
    round-trip sanity checks."""
    return [
        int(amplitude * math.sin(2 * math.pi * i / period))
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────
# mulaw8k_to_pcm16_16k
# ─────────────────────────────────────────────────────────────────────────


class TestMulaw8kToPcm16_16k:
    def test_empty_input_returns_empty_output(self):
        assert mulaw8k_to_pcm16_16k(b"") == b""

    def test_output_byte_length_doubles_sample_count(self):
        """N ms of 8kHz mono MULAW (1 byte/sample) -> ~2N ms worth of PCM16
        16kHz bytes: input_samples * 2 (rate factor) * 2 (bytes/PCM16 sample)."""
        # 20ms @ 8kHz = 160 samples/bytes (Twilio's own frame size).
        mulaw_bytes = bytes([0xFF] * 160)
        pcm16k = mulaw8k_to_pcm16_16k(mulaw_bytes)
        expected_len = 160 * (GEMINI_INPUT_PCM_RATE_HZ // TWILIO_MULAW_RATE_HZ) * 2
        assert len(pcm16k) == expected_len == 640

    def test_silence_handling(self):
        """MULAW 0xFF is the standard mu-law encoding of linear zero (silence)."""
        silence_mulaw = bytes([0xFF] * 80)
        pcm16k = mulaw8k_to_pcm16_16k(silence_mulaw)
        assert len(pcm16k) == 80 * 2 * 2
        samples = struct.unpack(f"<{len(pcm16k) // 2}h", pcm16k)
        # mu-law's zero code (0xFF) decodes to +/-ULAW_BIAS (132), not exact
        # zero — near-zero within the codec's own bias tolerance.
        assert all(abs(s) <= 132 for s in samples)

    def test_does_not_crash_on_single_byte(self):
        # Odd/degenerate input sizes are meaningful for MULAW (1 byte = 1
        # sample) — must not raise even for a single byte.
        out = mulaw8k_to_pcm16_16k(bytes([0x7F]))
        assert len(out) == 1 * 2 * 2

    def test_round_trip_approximately_recovers_tone(self):
        """mu-law decode -> upsample -> (downsample -> mu-law encode) should
        approximately recover the original mu-law-quantized tone, within
        mu-law's own quantization tolerance."""
        from app.utils.audio_utils import downsample_linear_samples, linear_samples_to_ulaw_bytes

        original_linear = _tone_samples(160, amplitude=8000)
        original_mulaw = bytes(linear_to_ulaw_sample(s) for s in original_linear)

        pcm16k = mulaw8k_to_pcm16_16k(original_mulaw)
        # Downsample back to 8k and re-encode to compare like-for-like.
        samples_16k = struct.unpack(f"<{len(pcm16k) // 2}h", pcm16k)
        samples_8k = downsample_linear_samples(list(samples_16k), 16000, 8000)
        round_tripped_mulaw = linear_samples_to_ulaw_bytes(samples_8k)

        recovered_linear = [ulaw_to_linear_sample(b) for b in round_tripped_mulaw]
        # mu-law quantization error can be a few percent of full scale at
        # this amplitude; allow generous tolerance since this is a lossy
        # codec round trip, not an exact-recovery check.
        max_diff = max(abs(a - b) for a, b in zip(original_linear, recovered_linear))
        assert max_diff < 2000


# ─────────────────────────────────────────────────────────────────────────
# pcm16_24k_to_mulaw8k
# ─────────────────────────────────────────────────────────────────────────


class TestPcm16_24kToMulaw8k:
    def test_empty_input_returns_empty_output(self):
        assert pcm16_24k_to_mulaw8k(b"") == b""

    def test_output_byte_length_matches_rate_ratio(self):
        """N ms of 24kHz PCM16 (2 bytes/sample) -> N/3 ms worth of MULAW
        8kHz bytes (1 byte/sample after /3 rate + /2 byte-width)."""
        samples = _tone_samples(240, amplitude=5000)  # 10ms @ 24kHz
        pcm_bytes = _pcm16le_bytes(samples)
        mulaw = pcm16_24k_to_mulaw8k(pcm_bytes)
        expected_len = 240 // (GEMINI_OUTPUT_PCM_RATE_HZ // TWILIO_MULAW_RATE_HZ)
        assert len(mulaw) == expected_len == 80

    def test_silence_handling(self):
        silence_pcm = _pcm16le_bytes([0] * 240)
        mulaw = pcm16_24k_to_mulaw8k(silence_pcm)
        assert len(mulaw) == 80
        # Silence should decode back to near-zero (within mu-law's own bias
        # tolerance — 0xFF, mu-law's zero code, decodes to +/-ULAW_BIAS=132).
        decoded = [ulaw_to_linear_sample(b) for b in mulaw]
        assert all(abs(s) <= 132 for s in decoded)

    def test_odd_length_buffer_does_not_crash(self):
        """A trailing incomplete PCM16 sample (odd byte count) must be
        dropped, not raise."""
        pcm_bytes = _pcm16le_bytes([100, 200, 300]) + b"\x01"  # 7 bytes total
        out = pcm16_24k_to_mulaw8k(pcm_bytes)
        # 3 usable samples -> usable=3, factor=3 -> 1 output sample.
        assert len(out) == 1

    def test_single_incomplete_byte_does_not_crash(self):
        assert pcm16_24k_to_mulaw8k(b"\x01") == b""

    def test_round_trip_approximately_recovers_tone(self):
        original = _tone_samples(240, amplitude=10000)
        pcm_bytes = _pcm16le_bytes(original)
        mulaw = pcm16_24k_to_mulaw8k(pcm_bytes)
        recovered = [ulaw_to_linear_sample(b) for b in mulaw]

        # Downsample the original 3x to compare like-for-like against the
        # 8kHz recovered signal (box-averaged, same method the function uses).
        from app.utils.audio_utils import downsample_linear_samples

        downsampled_original = downsample_linear_samples(original, 24000, 8000)
        assert len(recovered) == len(downsampled_original)
        max_diff = max(abs(a - b) for a, b in zip(downsampled_original, recovered))
        assert max_diff < 2500


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


class TestUpsampleLinearSamples:
    def test_empty_input(self):
        assert _upsample_linear_samples([], 2) == []

    def test_factor_one_returns_copy(self):
        samples = [1, 2, 3]
        out = _upsample_linear_samples(samples, 1)
        assert out == samples
        assert out is not samples

    def test_doubles_length(self):
        samples = [0, 100, 200, 100]
        out = _upsample_linear_samples(samples, 2)
        assert len(out) == 8

    def test_interpolates_between_samples(self):
        samples = [0, 100]
        out = _upsample_linear_samples(samples, 2)
        # out[0] = sample[0] exactly; out[1] halfway to sample[1].
        assert out[0] == 0
        assert out[1] == 50


class TestLinearSamplesToPcm16LeBytes:
    def test_empty(self):
        assert _linear_samples_to_pcm16le_bytes([]) == b""

    def test_round_trips_with_struct(self):
        samples = [0, 100, -100, 32767, -32768]
        encoded = _linear_samples_to_pcm16le_bytes(samples)
        decoded = list(struct.unpack(f"<{len(samples)}h", encoded))
        assert decoded == samples

    def test_clamps_out_of_range_values(self):
        encoded = _linear_samples_to_pcm16le_bytes([40000, -40000])
        decoded = list(struct.unpack("<2h", encoded))
        assert decoded == [32767, -32768]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
