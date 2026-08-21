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

import struct

import pytest

from app.utils.audio_utils import (
    MULAW_FRAME_BYTES,
    MULAW_SAMPLE_RATE_HZ,
    PCM16KStreamDownsampler,
    PCMStreamDownsampler,
    apply_micro_fade_in,
    apply_micro_fade_out,
    downsample_linear_samples,
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


# ─────────────────────────────────────────────────────────────────────────────
# PCMStreamDownsampler — generalized incremental PCM16LE -> mulaw downsampler
# (added for Hume AI TTS integration; also compared against the pre-existing
# 16k -> 8k only PCM16KStreamDownsampler used by the ElevenLabs background
# path.)
# ─────────────────────────────────────────────────────────────────────────────


def _pcm16le(samples: list[int]) -> bytes:
    return b"".join(struct.pack("<h", s) for s in samples)


def _reference_mulaw(samples: list[int], src_rate_hz: int, dst_rate_hz: int) -> bytes:
    """Independent reference: uses the pre-existing (non-incremental)
    ``downsample_linear_samples`` helper + ``linear_to_ulaw_sample`` directly,
    rather than re-deriving PCMStreamDownsampler's own box-average loop."""
    down = downsample_linear_samples(samples, src_rate_hz, dst_rate_hz)
    return bytes(linear_to_ulaw_sample(s) for s in down)


def test_pcm_stream_downsampler_48k_to_8k_matches_reference():
    """factor=6 (Hume's assumed 48kHz source -> Twilio's 8kHz mulaw)."""
    samples = [1000, 1200, 900, 1100, 1050, 950, -500, -600, -700, -800, -900, -1000]
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(48000, 8000)
    out = d.feed(pcm_bytes) + d.flush()

    assert out == _reference_mulaw(samples, 48000, 8000)
    assert len(out) == 2  # 12 samples / factor 6 = 2 output samples


def test_pcm_stream_downsampler_16k_to_8k_matches_reference():
    """factor=2 (arbitrary-ratio path exercised at the ratio the existing
    PCM16KStreamDownsampler was hard-coded for)."""
    samples = [100, 200, 300, 400, 1000, 2000, 5, 5]
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(16000, 8000)
    out = d.feed(pcm_bytes) + d.flush()

    assert out == _reference_mulaw(samples, 16000, 8000)


def test_pcm_stream_downsampler_16k_to_8k_is_NOT_always_byte_identical_to_pcm16k_downsampler():
    """
    POSSIBLE BUG (flagged, not fixed): PCMStreamDownsampler's box-average
    uses ``int(total / factor)`` (Python truncation toward zero), while the
    pre-existing PCM16KStreamDownsampler uses ``(s1 + s2) // 2`` (floor
    division). These are NOT equivalent for odd, negative sums — e.g.
    total=-3, factor=2: int(-3/2) == -1 but -3 // 2 == -2.

    The implementer's summary for this integration claimed the two
    downsamplers are "byte-identical" for the 16k->8k ratio. This test
    proves that claim is false in general (it only holds when every
    box-averaged sum happens to be non-negative or evenly divisible).
    Negative-sample audio (i.e. any real PCM waveform, since PCM alternates
    sign) hits this in practice.
    """
    samples = [-1, -2, 100, -101, 300, -301, 5, 5]
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(16000, 8000)
    pcm_stream_out = d.feed(pcm_bytes) + d.flush()

    d2 = PCM16KStreamDownsampler()
    linear_out = d2.feed(pcm_bytes) + d2.flush()
    pcm16k_out = bytes(linear_to_ulaw_sample(s) for s in linear_out)

    assert pcm_stream_out != pcm16k_out, (
        "Expected the two downsamplers to diverge on this negative-odd-sum "
        "input (truncation vs floor division) — if this now passes, the "
        "rounding behavior was changed and the 'byte-identical' claim should "
        "be re-verified end to end."
    )


def test_pcm_stream_downsampler_strips_leading_wav_header():
    samples = [1000, -1000, 2000, -2000, 500, -500]
    pcm_bytes = _pcm16le(samples)
    header = b"RIFF" + (36 + len(pcm_bytes)).to_bytes(4, "little") + b"WAVEfmt "
    # Minimal well-formed-enough header: only "RIFF"/"WAVE" markers and a
    # "data" chunk id + length are actually inspected by strip logic.
    wav_bytes = header + b"\x10\x00\x00\x00" + b"\x00" * 16 + b"data" + len(pcm_bytes).to_bytes(4, "little") + pcm_bytes

    d = PCMStreamDownsampler(16000, 8000)
    out = d.feed(wav_bytes) + d.flush()

    assert out == _reference_mulaw(samples, 16000, 8000)


def test_pcm_stream_downsampler_headerless_input_is_unaffected():
    # >=12 bytes so the header-vs-not decision resolves within this single
    # feed() call (see the POSSIBLE BUG test below for what happens when it
    # can't).
    samples = [1000, -1000, 2000, -2000, 500, -500]
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(16000, 8000)
    out = d.feed(pcm_bytes) + d.flush()

    assert out == _reference_mulaw(samples, 16000, 8000)


def test_pcm_stream_downsampler_buffers_partial_bytes_across_feed_calls():
    """Feeding a chunk that ends mid-sample-group (and even mid-int16-sample)
    must not lose or corrupt data — the remainder is buffered until enough
    bytes arrive to complete a full output sample."""
    samples = [1000, 1200, 900, 1100, 1050, 950]  # factor=6 -> 1 output sample
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(48000, 8000)
    out = b""
    # Feed one byte at a time, including a split mid-int16-sample.
    for i in range(0, len(pcm_bytes), 3):
        out += d.feed(pcm_bytes[i:i + 3])
    out += d.flush()

    assert out == _reference_mulaw(samples, 48000, 8000)


def test_pcm_stream_downsampler_feed_returns_empty_until_full_group_buffered():
    """No output should be emitted from feed() until a whole factor-sized
    group of samples has arrived. Uses factor=8 (64000->8000, 16 bytes/group)
    fed with >=12 bytes so the header-detection ambiguity below doesn't
    confound this assertion."""
    samples = [1000, 1200, 900, 1100, 1050, 950]  # 6 of 8 needed for a full group
    pcm_bytes = _pcm16le(samples)
    assert len(pcm_bytes) >= 12

    d = PCMStreamDownsampler(64000, 8000)
    out = d.feed(pcm_bytes)

    assert out == b""


def test_pcm_stream_downsampler_flush_encodes_partial_tail():
    """flush() must encode a short trailing group (fewer than `factor`
    samples) rather than discarding it. Feeds a full group (48k->8k,
    factor=6) plus a partial group so >=12 bytes pass through feed() first
    and the header decision is already resolved before flush() runs (see
    the POSSIBLE BUG test below for the case where it isn't)."""
    full_group = [1000, 1200, 900, 1100, 1050, 950]
    partial_tail = [300, -200, 100]
    samples = full_group + partial_tail
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(48000, 8000)
    head = d.feed(pcm_bytes)
    tail = d.flush()

    expected_head_avg = int(sum(full_group) / 6)
    expected_tail_avg = int(sum(partial_tail) / len(partial_tail))
    assert head == bytes([linear_to_ulaw_sample(expected_head_avg)])
    assert tail == bytes([linear_to_ulaw_sample(expected_tail_avg)])


def test_pcm_stream_downsampler_flush_silently_drops_short_tail_before_header_decided():
    """
    POSSIBLE BUG (flagged, not fixed): if the *total* audio fed across the
    stream's lifetime (feed() calls + the final flush()) never reaches 12
    bytes, ``_strip_header_if_ready()`` can never determine whether the
    stream started with a RIFF/WAVE header (it needs >=12 bytes to check the
    magic markers) — and ``flush()`` responds to that "still unknown"
    state by clearing the buffer and returning empty audio instead of
    treating the data as headerless PCM (which callers would reasonably
    expect for a short trailing chunk). Concretely: a 3-sample (6-byte)
    partial group at the very end of a short utterance is silently dropped
    with **no error, warning, or partial audio returned** if it's the only
    data the stream ever saw.
    """
    samples = [300, -200, 100]  # 6 bytes total — under the 12-byte header threshold
    pcm_bytes = _pcm16le(samples)
    assert len(pcm_bytes) < 12

    d = PCMStreamDownsampler(48000, 8000)
    fed = d.feed(pcm_bytes)
    tail = d.flush()

    assert fed == b""
    assert tail == b"", (
        "If this assertion starts failing, the header-ambiguity-drops-short-"
        "tail behavior has been fixed upstream — update/remove this "
        "regression test and consider removing the flagged-bug note in the "
        "test-writer report."
    )


def test_pcm_stream_downsampler_flush_with_nothing_buffered_returns_empty():
    d = PCMStreamDownsampler(48000, 8000)
    assert d.flush() == b""


def test_pcm_stream_downsampler_raises_on_non_integer_ratio():
    with pytest.raises(ValueError):
        PCMStreamDownsampler(44100, 8000)


def test_pcm_stream_downsampler_raises_on_zero_or_negative_rates():
    with pytest.raises(ValueError):
        PCMStreamDownsampler(0, 8000)
    with pytest.raises(ValueError):
        PCMStreamDownsampler(48000, 0)
    with pytest.raises(ValueError):
        PCMStreamDownsampler(-48000, 8000)


def test_pcm_stream_downsampler_same_rate_is_a_passthrough_encode():
    # >=12 bytes so the header decision resolves within this feed() call.
    samples = [1000, -1000, 2000, -2000, 500, -500]
    pcm_bytes = _pcm16le(samples)

    d = PCMStreamDownsampler(8000, 8000)
    out = d.feed(pcm_bytes) + d.flush()

    assert out == bytes(linear_to_ulaw_sample(s) for s in samples)
