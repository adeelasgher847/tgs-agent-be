"""Tests for Eleven-style audio tag stripping on non-ElevenLabs TTS."""

from app.utils.eleven_tts_text import (
    prepare_tts_text_for_provider,
    strip_eleven_v3_style_tags_for_non_eleven_tts,
)


def test_elevenlabs_strips_audio_and_ssml_tags():
    """Verify ElevenLabs strips SSML tags and audio bracket tags to prevent literal reading."""
    assert prepare_tts_text_for_provider("[breathes] Hello there [sigh]", "elevenlabs") == "Hello there"
    assert prepare_tts_text_for_provider("[excited] That's wonderful!", "elevenlabs") == "That's wonderful!"
    assert prepare_tts_text_for_provider("[sad] I'm sorry to hear that.", "elevenlabs") == "I'm sorry to hear that."
    assert (
        prepare_tts_text_for_provider(
            '<speak><prosody rate="0.93" pitch="0st" volume="medium"><break time="400ms"/> Hello world.</prosody></speak>',
            "elevenlabs",
        )
        == "Hello world."
    )
    assert prepare_tts_text_for_provider('<break time="400ms"/> Let me see.', "elevenlabs") == "Let me see."


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


def test_tags_split_across_streaming_chunks():
    """Verify split SSML/XML tags across streaming chunks do not leak tag words."""
    # Chunk 1 ends with unclosed tag
    assert prepare_tts_text_for_provider('Hello <prosody rate="95%"', "elevenlabs") == "Hello"
    # Chunk 2 starts with trailing tag closure
    assert prepare_tts_text_for_provider(' pitch="+1st">how are you?', "elevenlabs") == "how are you?"
    # Unclosed break tag fragment
    assert prepare_tts_text_for_provider('time="400ms"/> Welcome!', "elevenlabs") == "Welcome!"


def test_unknown_brackets_preserved():
    s = "Price is [500] and code [SKU-12]."
    assert strip_eleven_v3_style_tags_for_non_eleven_tts(s) == s
    assert prepare_tts_text_for_provider(s, "elevenlabs") == s


def test_normal_business_and_punctuation_intact():
    s = "Your 10% discount on order #1234 is confirmed for 2:30 PM."
    assert prepare_tts_text_for_provider(s, "elevenlabs") == s


def test_no_brackets_fast_path():
    s = "Plain text without tags."
    assert strip_eleven_v3_style_tags_for_non_eleven_tts(s) == s


def test_empty_after_strip():
    assert prepare_tts_text_for_provider("[breathes]", "google") == ""
    assert prepare_tts_text_for_provider("[breathes]", "elevenlabs") == ""
    assert prepare_tts_text_for_provider('<break time="200ms"/>', "elevenlabs") == ""


def test_default_non_eleven_strips():
    assert "[breathes]" not in prepare_tts_text_for_provider("[breathes] Hi", None)
    assert "[breathes]" not in prepare_tts_text_for_provider("[breathes] Hi", "openai-tts")



