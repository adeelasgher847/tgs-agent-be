"""Unit tests for ticket-field runtime resolution."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from app.core.agent_runtime import resolve_llm_runtime, resolve_tts_runtime
from app.core.llm_models import infer_llm_provider


def test_infer_llm_provider_openai():
    assert infer_llm_provider("gpt-4o-mini") == "openai"


def test_infer_llm_provider_groq():
    assert infer_llm_provider("llama-3.1-70b-versatile") == "groq"


def test_resolve_llm_runtime_prefers_ticket_llm_model():
    agent = MagicMock()
    agent.llm_model = "gpt-4o-mini"
    agent.model = None
    agent.provider = None
    agent.agent_temperature = None
    agent.agent_max_tokens = None

    runtime = resolve_llm_runtime(agent)
    assert runtime.model_name == "gpt-4o-mini"
    assert runtime.provider_slug == "openai"
    assert runtime.used_ticket_llm is True


def test_resolve_tts_runtime_ticket_elevenlabs():
    agent = MagicMock()
    agent.tts_provider_slug = "11labs"
    agent.tts_voice_external_id = "voice-abc"
    agent.tts_language = "en"
    agent.encrypted_elevenlabs_api_key = None
    agent.tts_settings_json = {}
    agent.language = "en"
    agent.tts_provider = None

    runtime = resolve_tts_runtime(agent)
    assert runtime.adapter_slug == "elevenlabs"
    assert runtime.voice_external_id == "voice-abc"
    assert runtime.used_ticket_tts is True


def test_resolve_tts_runtime_byo_injects_api_key(monkeypatch):
    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.tts_provider_slug = "11labs_byo"
    agent.tts_voice_external_id = "v1"
    agent.tts_language = "en"
    agent.encrypted_elevenlabs_api_key = "enc"
    agent.tts_settings_json = {}
    agent.language = "en"
    agent.tts_provider = None

    monkeypatch.setattr(
        "app.core.agent_runtime.decrypt_api_key",
        lambda _enc: "xi-test-key",
    )
    runtime = resolve_tts_runtime(agent)
    assert runtime.settings_json.get("elevenlabs_api_key") == "xi-test-key"
