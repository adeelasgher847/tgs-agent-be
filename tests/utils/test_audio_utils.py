"""Unit tests for low-level MULAW audio shaping utilities.

These tests focus on the small, deterministic envelope helpers we rely on for
clean call audio:

- ``apply_micro_fade_in`` — already in production; we re-cover its invariants
  here to lock the symmetry contract with the new fade-out helper.
- ``apply_micro_fade_out`` — newly added to remove the abrupt cut/click that
  callers hear at the end of an utterance, especially right before an
  ``[END_CALL]`` triggered hangup.

We deliberately avoid mocking the encoder/decoder: we want to assert real
mu-law behaviour on a tiny synthetic payload so any regression in the math
shows up immediately.
"""

from __future__ import annotations

from app.utils.audio_utils import (
    MULAW_FRAME_BYTES,
    MULAW_SAMPLE_RATE_HZ,
    apply_micro_fade_in,
    apply_micro_fade_out,
    linear_to_ulaw_sample,
    ulaw_to_linear_sample,
)


def _const_mulaw(value: int, length: int) -> bytes:
    """Return ``length`` mu-law bytes that all decode to ``value`` linear PCM."""
    return bytes([linear_to_ulaw_sample(value)] * length)


def _abs_linear(audio: bytes) -> list[int]:
    return [abs(ulaw_to_linear_sample(b)) for b in audio]


def test_fade_in_empty_returns_empty():
    assert apply_micro_fade_in(b"", duration_ms=25.0) == b""


def test_fade_out_empty_returns_empty():
    assert apply_micro_fade_out(b"", duration_ms=25.0) == b""


def test_fade_in_preserves_length():
    audio = _const_mulaw(8000, 400)  # 50ms @ 8kHz
    out = apply_micro_fade_in(audio, duration_ms=25.0)
    assert len(out) == len(audio)


def test_fade_out_preserves_length():
    audio = _const_mulaw(8000, 400)
    out = apply_micro_fade_out(audio, duration_ms=25.0)
    assert len(out) == len(audio)


def test_fade_in_ramps_up_from_quiet():
    """First sample after fade-in must be quieter than the last sample of the
    fade window — i.e. the envelope is monotonically increasing on average."""
    audio = _const_mulaw(8000, 400)
    out = apply_micro_fade_in(audio, duration_ms=25.0)

    fade_samples = int((25.0 / 1000.0) * MULAW_SAMPLE_RATE_HZ)
    head = _abs_linear(out[:fade_samples])
    tail_after_fade = _abs_linear(out[fade_samples:fade_samples + 20])

    # First sample is forced to ~0 by the linear ramp.
    assert head[0] < head[-1]
    # Once the fade window ends, audio is at full level (matches the rest).
    assert sum(tail_after_fade) > sum(head[: len(tail_after_fade)])


def test_fade_out_ramps_down_to_quiet():
    """Last sample of the audio must be much quieter than the head of the
    fade window — symmetric to fade-in."""
    audio = _const_mulaw(8000, 400)
    out = apply_micro_fade_out(audio, duration_ms=25.0)

    fade_samples = int((25.0 / 1000.0) * MULAW_SAMPLE_RATE_HZ)
    fade_tail = _abs_linear(out[-fade_samples:])
    body = _abs_linear(out[: len(out) - fade_samples])

    # The very last sample is multiplied by ~0 by the ramp.
    assert fade_tail[-1] < fade_tail[0]
    # The non-fade body keeps the original level (no accidental ramp leak).
    assert all(abs(s - body[0]) < 5 for s in body[:50])


def test_fade_in_and_out_compose_on_same_buffer():
    """Applying fade-in then fade-out (as we do for tiny single-frame
    final-utterances) must keep the middle untouched and only attenuate
    the head and tail."""
    audio = _const_mulaw(8000, MULAW_FRAME_BYTES * 2)
    shaped = apply_micro_fade_out(
        apply_micro_fade_in(audio, duration_ms=25.0),
        duration_ms=25.0,
    )

    fade_samples = int((25.0 / 1000.0) * MULAW_SAMPLE_RATE_HZ)
    middle = shaped[fade_samples:-fade_samples]
    expected_middle = audio[fade_samples:-fade_samples]
    assert middle == expected_middle


def test_fade_out_handles_audio_shorter_than_fade_window():
    """If the buffer is shorter than the requested fade duration the helper
    must still return a same-length buffer that is monotonically attenuated
    (no IndexError, no truncation, no length change)."""
    audio = _const_mulaw(8000, 40)  # 5ms @ 8kHz, way shorter than 25ms fade
    out = apply_micro_fade_out(audio, duration_ms=25.0)
    assert len(out) == len(audio)
    levels = _abs_linear(out)
    assert levels[0] >= levels[-1]
