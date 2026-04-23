"""Tests for ElevenLabs TTS background mixing utilities."""

from app.utils.audio_utils import MULAW_FRAME_BYTES, linear_to_ulaw_sample
from app.utils.eleven_tts_background import (
    BackgroundFrameMixer,
    cache_key_background_fragment,
    get_background_loop_bytes,
    mix_mulaw_bytes,
    parse_eleven_background_settings,
)


def test_parse_eleven_background_none_and_off():
    # No settings → default to "office" at 0.4
    bid, lvl = parse_eleven_background_settings(None)
    assert bid == "office"
    assert abs(lvl - 0.4) < 1e-6

    # Empty dict → default to "office"
    assert parse_eleven_background_settings({}) == ("office", 0.4)

    # Explicitly disabled → None
    assert parse_eleven_background_settings({"eleven_background": "none"})[0] is None
    assert parse_eleven_background_settings({"eleven_background": "OFF"})[0] is None


def test_parse_eleven_background_valid():
    bid, lvl = parse_eleven_background_settings(
        {"eleven_background": "soft_noise", "eleven_background_level": 0.15}
    )
    assert bid == "soft_noise"
    assert abs(lvl - 0.15) < 1e-6


def test_parse_eleven_background_clamps_level():
    _, lvl = parse_eleven_background_settings(
        {"eleven_background": "office", "eleven_background_level": 99.0}
    )
    assert lvl == 0.55


def test_parse_unknown_preset_falls_back_to_default():
    # Unknown preset → fall back to default "office" (not None)
    assert parse_eleven_background_settings({"eleven_background": "not_a_real_preset"})[0] == "office"


def test_cache_key_fragment_empty_without_background():
    # Explicitly disabled → empty suffix
    assert cache_key_background_fragment({"eleven_background": "none"}) == ""
    assert cache_key_background_fragment({"eleven_background": "off"}) == ""
    # Default (no key set) → office is used → non-empty suffix
    assert cache_key_background_fragment({}) != ""
    assert cache_key_background_fragment(None) != ""


def test_cache_key_fragment_with_background():
    s = cache_key_background_fragment({"eleven_background": "cafe", "eleven_background_level": 0.1})
    assert s.startswith("_ebg:cafe:0.1")


def test_mix_mulaw_bytes_identity_at_zero_level():
    voice = bytes([linear_to_ulaw_sample(1000)] * 320)
    out = mix_mulaw_bytes(voice, "soft_noise", 0.0)
    assert out == voice


def test_mix_mulaw_bytes_changes_samples():
    # Near-silence mu-law; mixing soft_noise should perturb samples
    voice = bytes([0xFF] * MULAW_FRAME_BYTES * 2)
    out = mix_mulaw_bytes(voice, "soft_noise", 0.25)
    assert len(out) == len(voice)
    assert out != voice


def test_background_frame_mixer_maintains_phase_across_frames():
    m = BackgroundFrameMixer("office", 0.2)
    f1 = bytes([0xFF]) * MULAW_FRAME_BYTES
    f2 = bytes([0xFF]) * MULAW_FRAME_BYTES
    o1 = m.mix_frame(f1)
    o2 = m.mix_frame(f2)
    # Same silent input should yield different outputs as the bed advances
    assert o1 != o2
