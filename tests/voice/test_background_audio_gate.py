"""Ambient office bed must stay off for all TTS providers by default."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.config import settings
from app.voice.tts_stream_mixin import TtsStreamMixin


class _GateProbe(TtsStreamMixin):
    """Minimal stand-in that only exercises _is_background_audio_enabled."""

    def __init__(self, agent):
        self.agent = agent
        self.db = None


def test_background_off_by_default_even_when_agent_opts_in(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_BACKGROUND_AUDIO_ENABLED", False)
    agent = SimpleNamespace(
        tts_settings_json={
            "background_enabled": True,
            "background_profile": "office",
        }
    )
    assert _GateProbe(agent)._is_background_audio_enabled() is False


def test_background_requires_env_and_agent_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_BACKGROUND_AUDIO_ENABLED", True)
    agent = SimpleNamespace(
        tts_settings_json={
            "background_enabled": True,
            "background_profile": "office",
        }
    )
    assert _GateProbe(agent)._is_background_audio_enabled() is True
    agent.tts_settings_json["background_enabled"] = False
    assert _GateProbe(agent)._is_background_audio_enabled() is False


def test_background_gate_is_provider_agnostic(monkeypatch):
    """Google/Rime/ElevenLabs all share the same gate — no provider slug check."""
    monkeypatch.setattr(settings, "VOICE_BACKGROUND_AUDIO_ENABLED", True)
    agent = SimpleNamespace(
        tts_settings_json={
            "background_enabled": True,
            "background_profile": "office",
        }
    )
    assert _GateProbe(agent)._is_background_audio_enabled() is True
