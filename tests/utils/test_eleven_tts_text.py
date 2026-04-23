"""Tests for Eleven-style audio tag stripping on non-ElevenLabs TTS."""

from app.utils.eleven_tts_text import (
    prepare_tts_text_for_provider,
    strip_eleven_v3_style_tags_for_non_eleven_tts,
)


def test_elevenlabs_pass_through():
    raw = "[breathes] Hello there [sigh]"
    assert prepare_tts_text_for_provider(raw, "elevenlabs") == raw
    assert prepare_tts_text_for_provider(raw, "ElevenLabs") == raw


def test_google_strips_known_tags():
    assert (
        prepare_tts_text_for_provider("[breathes] Hello there.", "google")
        == "Hello there."
    )
    assert (
        strip_eleven_v3_style_tags_for_non_eleven_tts(
            "Start [pause] middle [whispers] end"
        )
        == "Start middle end"
    )


def test_unknown_brackets_preserved():
    s = "Price is [500] and code [SKU-12]."
    assert strip_eleven_v3_style_tags_for_non_eleven_tts(s) == s


def test_no_brackets_fast_path():
    s = "Plain text without tags."
    assert strip_eleven_v3_style_tags_for_non_eleven_tts(s) is s


def test_empty_after_strip():
    assert prepare_tts_text_for_provider("[breathes]", "google") == ""


def test_default_non_eleven_strips():
    assert "[breathes]" not in prepare_tts_text_for_provider("[breathes] Hi", None)
    assert "[breathes]" not in prepare_tts_text_for_provider("[breathes] Hi", "openai-tts")
