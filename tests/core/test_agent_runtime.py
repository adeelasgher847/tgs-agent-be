"""Unit tests for ticket-field runtime resolution."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

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
