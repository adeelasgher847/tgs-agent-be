"""Unit tests for ticket-field runtime resolution."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core.agent_runtime import (
    llm_service_for_provider,
    resolve_llm_runtime,
    resolve_tts_runtime,
)
from app.core.llm_models import ALLOWED_LLM_MODELS, infer_llm_provider


def test_infer_llm_provider_openai():
    assert infer_llm_provider("gpt-4o-mini") == "openai"


def test_infer_llm_provider_groq():
    assert infer_llm_provider("llama-3.1-70b-versatile") == "groq"


# ─────────────────────────────── gpt-5 reasoning family (new models) ──────────

_NEW_GPT5_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4"]


@pytest.mark.parametrize("model_name", _NEW_GPT5_MODELS)
def test_gpt5_models_in_allow_list(model_name):
    assert model_name in ALLOWED_LLM_MODELS


@pytest.mark.parametrize("model_name", _NEW_GPT5_MODELS)
def test_infer_llm_provider_resolves_gpt5_models_to_openai(model_name):
    assert infer_llm_provider(model_name) == "openai"


@pytest.mark.parametrize("model_name", _NEW_GPT5_MODELS)
def test_resolve_llm_runtime_routes_gpt5_models_to_openai(model_name):
    agent = MagicMock()
    agent.llm_model = model_name
    agent.model = None
    agent.provider = None
    agent.agent_temperature = None
    agent.agent_max_tokens = None

    runtime = resolve_llm_runtime(agent)
    assert runtime.model_name == model_name
    assert runtime.provider_slug == "openai"
    assert runtime.used_ticket_llm is True


@pytest.mark.parametrize("model_name", _NEW_GPT5_MODELS + ["gpt-4.1"])
def test_llm_service_for_provider_routes_gpt_family_to_openai_singleton(model_name):
    from app.services.openai_service import openai_service

    provider_slug = infer_llm_provider(model_name)
    assert llm_service_for_provider(provider_slug) is openai_service


@pytest.mark.parametrize(
    "model_name, expected_provider, expected_service_attr",
    [
        ("gemini-2.5-flash", "gemini", "vertex_gemini_service"),
        ("gemini-2.0-flash-001", "gemini", "vertex_gemini_service"),
        ("gemini-1.5-pro", "gemini", "vertex_gemini_service"),
        ("claude-3-5-sonnet", "gemini", "vertex_gemini_service"),  # no Anthropic service yet
        ("llama-3.1-70b-versatile", "groq", "groq_service"),
        ("llama-3.1-8b-instant", "groq", "groq_service"),
        ("gpt-4o", "openai", "openai_service"),
        ("gpt-4o-mini", "openai", "openai_service"),
        ("gpt-4.1", "openai", "openai_service"),
        ("gpt-4.1-mini", "openai", "openai_service"),
        ("gpt-4-turbo", "openai", "openai_service"),
    ],
)
def test_existing_models_still_route_to_prior_providers(
    model_name, expected_provider, expected_service_attr
):
    """Regression: adding gpt-5 models must not change routing for any
    pre-existing allow-listed model."""
    assert infer_llm_provider(model_name) == expected_provider

    if expected_service_attr == "openai_service":
        from app.services.openai_service import openai_service as expected_service
    elif expected_service_attr == "groq_service":
        from app.services.groq_service import groq_service as expected_service
    else:
        from app.services.vertex_gemini_service import vertex_gemini_service as expected_service

    assert llm_service_for_provider(expected_provider) is expected_service


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
    # Platform ElevenLabs (raw slug "11labs", not "11labs_byo") must NOT be
    # flagged as BYO — the ElevenLabs surcharge applies to this call.
    assert runtime.is_byo_elevenlabs is False


def _byo_agent(ciphertext: str = "enc") -> MagicMock:
    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.tts_provider_slug = "11labs_byo"
    agent.tts_voice_external_id = "v1"
    agent.tts_language = "en"
    agent.encrypted_elevenlabs_api_key = ciphertext
    agent.tts_settings_json = {}
    agent.language = "en"
    agent.tts_provider = None
    return agent


def test_resolve_tts_runtime_byo_injects_api_key_pgcrypto():
    """pgcrypto ciphertext → decrypt_stored_elevenlabs_key → key injected."""
    agent = _byo_agent("jA0ECQMCpgcrypto_base64==")  # non-eyJ prefix = pgcrypto
    mock_db = MagicMock()

    with patch(
        "app.core.db_encryption.decrypt_stored_elevenlabs_key",
        return_value="xi-pgcrypto-key",
    ) as mock_dec:
        runtime = resolve_tts_runtime(agent, db=mock_db)

    assert runtime.settings_json.get("elevenlabs_api_key") == "xi-pgcrypto-key"
    mock_dec.assert_called_once_with("jA0ECQMCpgcrypto_base64==", db=mock_db)
    # BYO ElevenLabs with a usable stored key: adapter_slug still collapses
    # to "elevenlabs", but is_byo_elevenlabs must be True so the platform
    # surcharge (app.services.credit_service) is exempted.
    assert runtime.adapter_slug == "elevenlabs"
    assert runtime.is_byo_elevenlabs is True


def test_resolve_tts_runtime_byo_slug_without_stored_key_not_flagged_byo():
    """BYO slug ("11labs_byo") with NO stored encrypted key is not a usable
    BYO configuration — must not be exempted from the surcharge."""
    agent = _byo_agent(ciphertext=None)
    agent.encrypted_elevenlabs_api_key = None

    runtime = resolve_tts_runtime(agent, db=MagicMock())

    assert runtime.adapter_slug == "elevenlabs"
    assert runtime.is_byo_elevenlabs is False


def test_resolve_tts_runtime_byo_injects_api_key_jwt_legacy():
    """Legacy JWT ciphertext (eyJ…) decrypts via JWT fallback inside unified helper."""
    jwt_ct = "eyJhbGciOiJIUzI1NiJ9.payload.sig"
    agent = _byo_agent(jwt_ct)

    with patch(
        "app.core.db_encryption.decrypt_stored_elevenlabs_key",
        return_value="xi-jwt-legacy-key",
    ) as mock_dec:
        runtime = resolve_tts_runtime(agent, db=None)

    assert runtime.settings_json.get("elevenlabs_api_key") == "xi-jwt-legacy-key"
    mock_dec.assert_called_once_with(jwt_ct, db=None)


def test_resolve_tts_runtime_byo_no_db_opens_session():
    """When db=None and ciphertext is pgcrypto, decrypt_stored_elevenlabs_key still called."""
    agent = _byo_agent("jA0ECQMCnodb==")
    with patch(
        "app.core.db_encryption.decrypt_stored_elevenlabs_key",
        return_value="xi-nodb-key",
    ) as mock_dec:
        runtime = resolve_tts_runtime(agent, db=None)

    assert runtime.settings_json.get("elevenlabs_api_key") == "xi-nodb-key"
    mock_dec.assert_called_once_with("jA0ECQMCnodb==", db=None)


def test_resolve_tts_runtime_byo_legacy_monkeypatch(monkeypatch):
    """Backward-compat test: the old monkeypatch style still works via the helper."""
    agent = _byo_agent("eyJhbGciOiJIUzI1NiJ9.x.y")

    monkeypatch.setattr(
        "app.core.db_encryption.decrypt_stored_elevenlabs_key",
        lambda ct, *, db=None: "xi-monkeypatched",
    )
    runtime = resolve_tts_runtime(agent)
    assert runtime.settings_json.get("elevenlabs_api_key") == "xi-monkeypatched"
