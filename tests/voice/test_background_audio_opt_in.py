"""
Tests for TTS background audio default behavior and explicit opt-in.

Verifies:
1. Background audio is DISABLED by default when background_enabled is omitted.
2. Background audio is ENABLED when background_enabled is explicitly true / "true" / "1".
3. Background audio is DISABLED when background_enabled is false / "false" / "0".
4. Background audio is never enabled for non-ElevenLabs providers (Google, Deepgram, etc.).
"""
from unittest.mock import MagicMock
import pytest

from app.voice.tts_stream_mixin import TtsStreamMixin


class _FakeHandler(TtsStreamMixin):
    def __init__(self, agent):
        self.agent = agent
        self.db = None


def _make_mock_agent(provider_slug="elevenlabs", settings_json=None):
    agent = MagicMock()
    agent.tts_provider_slug = provider_slug
    agent.tts_voice_external_id = "test-voice-id"
    agent.tts_voice_settings_json = None
    agent.tts_settings_json = settings_json if settings_json is not None else {}
    agent.tts_provider_id = None
    agent.tts_voice_id = None
    agent.tts_provider = MagicMock(slug=provider_slug)
    agent.tts_voice = MagicMock(external_voice_id="test-voice-id")
    agent.language = "en"
    agent.voice_type = "female"
    return agent


def test_background_audio_disabled_by_default():
    agent = _make_mock_agent(provider_slug="elevenlabs", settings_json={})
    handler = _FakeHandler(agent)
    assert handler._is_background_audio_enabled() is False


def test_background_audio_explicit_opt_in_boolean():
    agent = _make_mock_agent(provider_slug="elevenlabs", settings_json={"background_enabled": True})
    handler = _FakeHandler(agent)
    assert handler._is_background_audio_enabled() is True


def test_background_audio_explicit_opt_in_string_variants():
    for val in ["true", "True", "1", "on", "yes"]:
        agent = _make_mock_agent(provider_slug="elevenlabs", settings_json={"background_enabled": val})
        handler = _FakeHandler(agent)
        assert handler._is_background_audio_enabled() is True, f"Failed for {val}"


def test_background_audio_explicit_opt_out():
    for val in [False, "false", "0", "off", "no"]:
        agent = _make_mock_agent(provider_slug="elevenlabs", settings_json={"background_enabled": val})
        handler = _FakeHandler(agent)
        assert handler._is_background_audio_enabled() is False, f"Failed for {val}"


def test_background_audio_non_elevenlabs_never_enabled():
    agent = _make_mock_agent(provider_slug="google", settings_json={"background_enabled": True})
    handler = _FakeHandler(agent)
    assert handler._is_background_audio_enabled() is False
