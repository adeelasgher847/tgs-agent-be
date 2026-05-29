"""
Uniform TTS speed + volume across all providers.

Covers:
  - apply_volume_fade supports gain > 1.0 (clamped to safe max)
  - resolve_tts_runtime merges nested {"settings": {...}} into top level
  - resolve_tts_runtime clamps out-of-range speed/volume to safe bounds
  - GoogleTTSAdapter maps user `speed` to `speaking_rate`
  - ElevenLabsAdapter forwards `speed` via voice_settings and strips `volume`
  - RimeTTSAdapter inverts user speed for mistv2 (slower user → higher alpha)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. apply_volume_fade — extended for gain > 1.0
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyVolumeFade:
    # mulaw 0x00 inverts to 0xFF → max-amplitude sample. Mid-bucket bytes like
    # 0x7F decode to such a tiny linear value that *2 and /2 both round-trip
    # to the same encoded byte (quantization). Use a high-amplitude byte so
    # gain effects are observable after re-encoding.
    HIGH_AMP_BYTE = 0x10

    def test_volume_one_returns_input_unchanged(self):
        from app.utils.audio_utils import apply_volume_fade

        audio = bytes([self.HIGH_AMP_BYTE] * 160)
        out = apply_volume_fade(audio, 1.0)
        assert out is audio  # identity short-circuit

    def test_volume_zero_returns_silence(self):
        from app.utils.audio_utils import apply_volume_fade

        audio = bytes([self.HIGH_AMP_BYTE] * 160)
        out = apply_volume_fade(audio, 0.0)
        assert out == bytes([0xFF] * 160)

    def test_volume_half_reduces_amplitude(self):
        from app.utils.audio_utils import (
            apply_volume_fade,
            ulaw_to_linear_sample,
        )

        audio = bytes([self.HIGH_AMP_BYTE] * 160)
        quieter = apply_volume_fade(audio, 0.5)
        orig_amp = abs(ulaw_to_linear_sample(audio[0]))
        new_amp = abs(ulaw_to_linear_sample(quieter[0]))
        assert new_amp < orig_amp

    def test_volume_above_one_increases_amplitude(self):
        """Regression: previously >=1.0 short-circuited to original audio."""
        from app.utils.audio_utils import (
            apply_volume_fade,
            ulaw_to_linear_sample,
        )

        audio = bytes([self.HIGH_AMP_BYTE] * 160)
        louder = apply_volume_fade(audio, 1.5)
        orig_amp = abs(ulaw_to_linear_sample(audio[0]))
        new_amp = abs(ulaw_to_linear_sample(louder[0]))
        assert new_amp > orig_amp

    def test_extreme_volume_is_clamped_no_overflow(self):
        """Very large gain must not raise nor produce non-bytes output."""
        from app.utils.audio_utils import apply_volume_fade

        audio = bytes([self.HIGH_AMP_BYTE] * 160)
        out = apply_volume_fade(audio, 99.0)
        assert isinstance(out, (bytes, bytearray))
        assert len(out) == len(audio)


# ─────────────────────────────────────────────────────────────────────────────
# 2. resolve_tts_runtime — nested settings merge + clamping
# ─────────────────────────────────────────────────────────────────────────────


def _rime_agent(tts_settings_json: Dict[str, Any]) -> MagicMock:
    agent = MagicMock()
    agent.language = "en"
    agent.tts_settings_json = tts_settings_json
    agent.tts_provider_slug = "rime"
    agent.tts_voice_external_id = "mistv2_Wildflower"
    agent.tts_language = "en"
    agent.encrypted_elevenlabs_api_key = None
    return agent


class TestResolveTtsRuntimeMerge:
    def test_nested_settings_promoted_to_top_level(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = _rime_agent({"settings": {"speed": 0.8, "volume": 0.5}})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["speed"] == pytest.approx(0.8)
        assert runtime.settings_json["volume"] == pytest.approx(0.5)

    def test_flat_top_level_still_works(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = _rime_agent({"speed": 1.2, "volume": 1.0})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["speed"] == pytest.approx(1.2)
        assert runtime.settings_json["volume"] == pytest.approx(1.0)

    def test_top_level_wins_on_conflict_with_nested(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = _rime_agent({"speed": 1.5, "settings": {"speed": 0.5}})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["speed"] == pytest.approx(1.5)

    def test_defaults_to_one_when_absent(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = _rime_agent({})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["speed"] == 1.0
        assert runtime.settings_json["volume"] == 1.0

    def test_speed_above_max_is_clamped(self):
        from app.core.agent_runtime import TTS_SPEED_MAX, resolve_tts_runtime

        agent = _rime_agent({"speed": 50.0})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["speed"] == TTS_SPEED_MAX

    def test_volume_above_max_is_clamped(self):
        from app.core.agent_runtime import TTS_VOLUME_MAX, resolve_tts_runtime

        agent = _rime_agent({"volume": 99.0})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["volume"] == TTS_VOLUME_MAX

    def test_negative_volume_is_clamped_to_zero(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = _rime_agent({"volume": -2.0})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["volume"] == 0.0

    def test_invalid_value_falls_back_to_default(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = _rime_agent({"speed": "fast", "volume": None})
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json["speed"] == 1.0
        assert runtime.settings_json["volume"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. GoogleTTSAdapter — speed → speaking_rate
# ─────────────────────────────────────────────────────────────────────────────


class TestGoogleSpeed:
    def test_user_speed_maps_to_speaking_rate(self):
        from app.utils.tts_adapter import GoogleTTSAdapter

        captured: Dict[str, Any] = {}

        def _fake_tts(**kwargs):
            captured.update(kwargs)
            return b"\xff" * 160

        with patch(
            "app.services.google_tts_service.google_tts_service.text_to_speech",
            side_effect=_fake_tts,
        ):
            adapter = GoogleTTSAdapter()
            adapter.synthesize(
                text="hello",
                voice_external_id="en-US-Standard-A",
                settings_json={"speed": 1.2, "volume": 0.7},
            )

        assert captured.get("speaking_rate") == pytest.approx(1.2)

    def test_explicit_speaking_rate_wins_over_speed(self):
        from app.utils.tts_adapter import GoogleTTSAdapter

        captured: Dict[str, Any] = {}

        def _fake_tts(**kwargs):
            captured.update(kwargs)
            return b"\xff" * 160

        with patch(
            "app.services.google_tts_service.google_tts_service.text_to_speech",
            side_effect=_fake_tts,
        ):
            adapter = GoogleTTSAdapter()
            adapter.synthesize(
                text="hello",
                voice_external_id="en-US-Standard-A",
                settings_json={"speed": 1.2, "speaking_rate": 0.9},
            )

        assert captured.get("speaking_rate") == pytest.approx(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ElevenLabsAdapter — speed in voice_settings, volume stripped
# ─────────────────────────────────────────────────────────────────────────────


class TestElevenLabsSpeedVolume:
    def test_speed_lands_in_voice_settings(self):
        from app.utils.tts_adapter import ElevenLabsAdapter

        captured: Dict[str, Any] = {}

        def _fake_tts(**kwargs):
            captured.update(kwargs)
            return b"\xff" * 160

        with patch(
            "app.services.elevenlabs_service.elevenlabs_service.text_to_speech",
            side_effect=_fake_tts,
        ):
            adapter = ElevenLabsAdapter()
            adapter.synthesize(
                text="hi",
                voice_external_id="voice-1",
                settings_json={"speed": 1.1, "volume": 0.6},
            )

        voice_settings = captured.get("voice_settings") or {}
        assert voice_settings.get("speed") == pytest.approx(1.1)
        # volume must be stripped — not part of ElevenLabs API
        assert "volume" not in voice_settings


# ─────────────────────────────────────────────────────────────────────────────
# 5. RimeTTSAdapter — inversion helper
# ─────────────────────────────────────────────────────────────────────────────


class TestRimeSpeedInversion:
    def test_mistv2_inverts_user_speed(self):
        from app.utils.tts_adapter import RimeTTSAdapter

        # user speed > 1 (faster) → speedAlpha < 1 (faster on mistv2)
        alpha_fast = RimeTTSAdapter._user_speed_to_speed_alpha(1.5, "mistv2")
        assert alpha_fast < 1.0
        # user speed < 1 (slower) → speedAlpha > 1 (slower on mistv2)
        alpha_slow = RimeTTSAdapter._user_speed_to_speed_alpha(0.8, "mistv2")
        assert alpha_slow > 1.0

    def test_mistv3_does_not_invert(self):
        from app.utils.tts_adapter import RimeTTSAdapter

        alpha = RimeTTSAdapter._user_speed_to_speed_alpha(1.5, "mistv3")
        assert alpha == pytest.approx(1.5)

    def test_invalid_speed_defaults_to_one(self):
        from app.utils.tts_adapter import RimeTTSAdapter

        assert RimeTTSAdapter._user_speed_to_speed_alpha("nope", "mistv2") == 1.0
        assert RimeTTSAdapter._user_speed_to_speed_alpha(0.0, "mistv2") == 1.0
