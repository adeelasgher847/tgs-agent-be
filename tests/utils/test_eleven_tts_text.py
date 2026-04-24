"""Tests for Eleven-style audio tag stripping on non-ElevenLabs TTS."""

from app.utils.eleven_tts_text import (
    apply_elevenlabs_breathing_fallback,
    build_elevenlabs_audio_tag_prompt_block,
    contains_elevenlabs_audio_tag,
    prepare_tts_text_for_provider,
    strip_eleven_v3_style_tags_for_non_eleven_tts,
    supports_elevenlabs_audio_tags,
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
    assert prepare_tts_text_for_provider("[sad] I'm sorry to hear that.", "google") == "I'm sorry to hear that."
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


def test_only_elevenlabs_supports_audio_tags():
    assert supports_elevenlabs_audio_tags("elevenlabs") is True
    assert supports_elevenlabs_audio_tags("ElevenLabs") is True
    assert supports_elevenlabs_audio_tags("google") is False
    assert supports_elevenlabs_audio_tags(None) is False


def test_supports_respects_config_disable(monkeypatch):
    from app.core import config
    from app.utils import eleven_tts_text

    monkeypatch.setattr(config.settings, "ENABLE_ELEVENLABS_AUDIO_TAGS", False)
    # settings is the same object used by the module
    assert eleven_tts_text.supports_elevenlabs_audio_tags("elevenlabs") is False
    assert build_elevenlabs_audio_tag_prompt_block("elevenlabs") == ""


def test_prompt_block_only_emitted_for_elevenlabs():
    from app.utils.eleven_tts_text import get_elevenlabs_voice_prompt_rule_lines

    block = build_elevenlabs_audio_tag_prompt_block("elevenlabs")
    assert "ELEVENLABS" in block
    assert "[pause]" in block or "[pauses]" in block
    assert "[excited]" in block
    assert "[sad]" in block or "sorrowful" in block
    assert build_elevenlabs_audio_tag_prompt_block("google") == ""
    a, b, c = get_elevenlabs_voice_prompt_rule_lines()
    assert a.startswith("- OUTPUT")
    assert "3. NO SSML" in c


def test_contains_elevenlabs_audio_tag_detects_known_tags():
    assert contains_elevenlabs_audio_tag("[breathes] Hello") is True
    assert contains_elevenlabs_audio_tag("Price is [500]") is False


def test_breathing_fallback_added_for_long_eleven_text():
    raw = "Hello there, thank you for calling today."
    assert apply_elevenlabs_breathing_fallback(raw) == "[breathes] Hello there, thank you for calling today."
    assert prepare_tts_text_for_provider(raw, "elevenlabs") == raw


def test_breathing_fallback_skips_short_or_control_token_text():
    assert apply_elevenlabs_breathing_fallback("Thanks") == "Thanks"
    token_text = "Sure [CHECK_SLOTS:date=2026-05-01]"
    assert apply_elevenlabs_breathing_fallback(token_text) == token_text
