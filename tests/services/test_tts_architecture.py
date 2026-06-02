import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db, require_admin_or_owner, require_member_or_admin, require_tenant
from app.db.base import Base
from app.main import app
from app.models.agent import Agent
from app.models.tenant import Tenant
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.models.user import User
from app.routers.tts_audio import audio_cache
from app.schemas.agent import AgentCreate
from app.services.agent_service import agent_service


def _agent_create(**overrides) -> AgentCreate:
    """Minimal valid ticket-shaped create payload for service tests."""
    data = {
        "name": overrides.pop("name", "Test Agent"),
        "llmModel": overrides.pop("llm_model", "gpt-4o-mini"),
        "ttsModel": overrides.pop(
            "tts_model",
            {"provider": "11labs", "voiceId": "voice-id", "language": "en"},
        ),
    }
    data.update(overrides)
    return AgentCreate.model_validate(data)
from app.services.bidirectional_stream_service import generate_mulaw_tts
from app.services.tts_catalog_service import tts_catalog_service


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def tts_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    tenant = Tenant(name="TTS Tenant", schema_name="tts_tenant")
    db.add(tenant)
    db.flush()

    user = User(
        email=f"tts-{uuid.uuid4()}@example.com",
        hashed_password="hash",
        first_name="TTS",
        last_name="User",
        current_tenant_id=tenant.id,
    )
    db.add(user)
    db.commit()
    db.refresh(tenant)
    db.refresh(user)

    try:
        yield db, tenant, user
    finally:
        db.close()


def test_agent_create_rejects_tts_provider_voice_mismatch(tts_db):
    db, tenant, user = tts_db
    provider_a = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    provider_b = TTSProvider(slug="openai-tts", display_name="OpenAI TTS")
    db.add_all([provider_a, provider_b])
    db.flush()

    voice_b = TTSVoice(
        provider_id=provider_b.id,
        external_voice_id="voice-openai",
        display_name="Voice B",
        is_active=True,
    )
    db.add(voice_b)
    db.commit()

    payload = _agent_create(
        name="Mismatch Agent",
        tts_provider_id=provider_a.id,
        tts_voice_id=voice_b.id,
    )

    with pytest.raises(HTTPException) as excinfo:
        agent_service.create_agent(db, payload, tenant.id, user.id)
    assert excinfo.value.status_code == 422
    assert "does not belong" in str(excinfo.value.detail)


def test_agent_create_rejects_tts_payload_api_key(tts_db):
    db, tenant, user = tts_db
    provider = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    db.add(provider)
    db.flush()
    voice = TTSVoice(
        provider_id=provider.id,
        external_voice_id="voice-eleven",
        display_name="Voice A",
        is_active=True,
    )
    db.add(voice)
    db.commit()

    payload = _agent_create(
        name="Unsafe Agent",
        tts_provider_id=provider.id,
        tts_voice_id=voice.id,
        tts_settings_json={"api_key": "should-not-pass"},
    )

    with pytest.raises(HTTPException) as excinfo:
        agent_service.create_agent(db, payload, tenant.id, user.id)
    assert excinfo.value.status_code == 422
    assert "credentials must not be passed" in str(excinfo.value.detail)


def test_agent_create_rejects_invalid_background_enabled(tts_db):
    db, tenant, user = tts_db
    provider = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    db.add(provider)
    db.flush()
    voice = TTSVoice(
        provider_id=provider.id,
        external_voice_id="voice-eleven",
        display_name="Voice A",
        is_active=True,
    )
    db.add(voice)
    db.commit()

    payload = _agent_create(
        name="Invalid BG Enabled Agent",
        tts_provider_id=provider.id,
        tts_voice_id=voice.id,
        tts_settings_json={"background_enabled": "maybe"},
    )

    with pytest.raises(HTTPException) as excinfo:
        agent_service.create_agent(db, payload, tenant.id, user.id)
    assert excinfo.value.status_code == 422
    assert "background_enabled" in str(excinfo.value.detail)


def test_agent_create_rejects_invalid_background_profile(tts_db):
    db, tenant, user = tts_db
    provider = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    db.add(provider)
    db.flush()
    voice = TTSVoice(
        provider_id=provider.id,
        external_voice_id="voice-eleven",
        display_name="Voice A",
        is_active=True,
    )
    db.add(voice)
    db.commit()

    payload = _agent_create(
        name="Invalid BG Profile Agent",
        tts_provider_id=provider.id,
        tts_voice_id=voice.id,
        tts_settings_json={"background_profile": "airport"},
    )

    with pytest.raises(HTTPException) as excinfo:
        agent_service.create_agent(db, payload, tenant.id, user.id)
    assert excinfo.value.status_code == 422
    assert "background_profile" in str(excinfo.value.detail)


def test_agent_create_rejects_invalid_background_volume(tts_db):
    db, tenant, user = tts_db
    provider = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    db.add(provider)
    db.flush()
    voice = TTSVoice(
        provider_id=provider.id,
        external_voice_id="voice-eleven",
        display_name="Voice A",
        is_active=True,
    )
    db.add(voice)
    db.commit()

    payload = _agent_create(
        name="Invalid BG Volume Agent",
        tts_provider_id=provider.id,
        tts_voice_id=voice.id,
        tts_settings_json={"background_volume": 120},
    )

    with pytest.raises(HTTPException) as excinfo:
        agent_service.create_agent(db, payload, tenant.id, user.id)
    assert excinfo.value.status_code == 422
    assert "background_volume" in str(excinfo.value.detail)


def test_tts_router_lists_providers_and_voices(client, db):
    provider = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    db.add(provider)
    db.flush()
    voice = TTSVoice(
        provider_id=provider.id,
        external_voice_id="voice_1",
        display_name="Voice 1",
        preview_audio_url="https://cdn.example.com/voice_1.mp3",
        is_active=True,
    )
    db.add(voice)
    db.commit()

    test_user = db.query(User).first()
    test_user.current_tenant_id = db.query(Tenant).first().id

    def _user_override():
        return test_user

    app.dependency_overrides[require_tenant] = _user_override
    app.dependency_overrides[require_member_or_admin] = _user_override
    app.dependency_overrides[require_admin_or_owner] = _user_override

    def _get_db():
        yield db

    app.dependency_overrides[get_db] = _get_db

    try:
        providers_res = client.get("/api/v1/tts/providers")
        assert providers_res.status_code == 200
        providers_data = providers_res.json()["data"]["providers"]
        assert len(providers_data) >= 1

        voices_res = client.get(f"/api/v1/tts/voices?provider_id={provider.id}")
        assert voices_res.status_code == 200
        voices_data = voices_res.json()["data"]["voices"]
        assert len(voices_data) == 1
        assert voices_data[0]["preview_audio_url"] == "https://cdn.example.com/voice_1.mp3"

        # eleven-backgrounds endpoint removed; background defaults to office@0.4 automatically.
        bg_res = client.get("/api/v1/tts/eleven-backgrounds")
        assert bg_res.status_code == 404
    finally:
        app.dependency_overrides.pop(require_tenant, None)
        app.dependency_overrides.pop(require_member_or_admin, None)
        app.dependency_overrides.pop(require_admin_or_owner, None)
        app.dependency_overrides.pop(get_db, None)


def test_tts_voice_sync_upserts_rows(tts_db):
    db, _, _ = tts_db
    provider = TTSProvider(slug="elevenlabs", display_name="ElevenLabs")
    db.add(provider)
    db.commit()

    class _FakeAdapter:
        def list_voices(self):
            return [{"voice_id": "voice_a", "name": "Voice A", "preview_url": "https://a.mp3"}]

        def normalize_voice_payload(self, payload):
            return {
                "external_voice_id": payload["voice_id"],
                "display_name": payload["name"],
                "language_code": "en",
                "gender": "female",
                "accent": None,
                "description": None,
                "preview_audio_url": payload.get("preview_url"),
                "sample_rate_hz": None,
                "metadata_json": payload,
            }

    with patch("app.services.tts_catalog_service.get_tts_adapter_for_provider", return_value=_FakeAdapter()):
        result = tts_catalog_service.sync_provider_voices(db, "elevenlabs")
        assert result["created"] == 1
        assert result["fetched"] == 1

        # Upsert same voice with a changed display name
        class _FakeAdapterUpdated(_FakeAdapter):
            def list_voices(self):
                return [{"voice_id": "voice_a", "name": "Voice A Updated", "preview_url": "https://a2.mp3"}]

        with patch("app.services.tts_catalog_service.get_tts_adapter_for_provider", return_value=_FakeAdapterUpdated()):
            result2 = tts_catalog_service.sync_provider_voices(db, "elevenlabs")
            assert result2["updated"] == 1

    updated_voice = db.query(TTSVoice).filter(TTSVoice.external_voice_id == "voice_a").first()
    assert updated_voice is not None
    assert updated_voice.display_name == "Voice A Updated"
    assert updated_voice.preview_audio_url == "https://a2.mp3"


def test_generate_mulaw_tts_uses_provider_adapter_for_agent():
    audio_cache.clear()

    class _FakeAdapter:
        def __init__(self):
            self.calls = []

        def synthesize(self, text, voice_external_id, settings_json=None):
            self.calls.append((text, voice_external_id, settings_json))
            return b"\xff" * 320

    fake_adapter = _FakeAdapter()
    fake_agent = SimpleNamespace(
        tts_provider=SimpleNamespace(slug="elevenlabs"),
        tts_voice=SimpleNamespace(external_voice_id="voice-eleven"),
        tts_settings_json={"stability": 0.5, "eleven_background": "off"},
    )

    with patch("app.services.bidirectional_stream_service.get_tts_adapter", return_value=fake_adapter):
        audio = asyncio.run(
            generate_mulaw_tts(
                text="Hello from provider adapter",
                lang="en",
                voice="female",
                agent=fake_agent,
            )
        )

    assert audio == b"\xff" * 320
    assert len(fake_adapter.calls) == 1
    text, voice_external_id, settings_json = fake_adapter.calls[0]
    assert text == "Hello from provider adapter"
    assert voice_external_id == "voice-eleven"
    assert settings_json["output_format"] == "ulaw_8000"


def test_generate_mulaw_tts_strips_tags_for_google_agent():
    """Google path must not send Eleven v3 [tags] to the TTS API."""
    audio_cache.clear()

    class _FakeG:
        def text_to_speech(self, **kwargs):
            t = kwargs.get("text", "")
            assert "[breathes]" not in t
            return b"ok"

    fake_agent = SimpleNamespace(
        tts_provider=SimpleNamespace(slug="google"),
        tts_voice=SimpleNamespace(external_voice_id="en-US-Studio-O"),
    )

    with patch("app.services.bidirectional_stream_service.google_tts_service", _FakeG()):
        audio = asyncio.run(
            generate_mulaw_tts(
                text="[breathes] Hello world",
                lang="en",
                voice="female",
                agent=fake_agent,
            )
        )
    assert audio == b"ok"


def test_generate_mulaw_tts_mixes_eleven_background_when_configured():
    audio_cache.clear()

    class _FakeAdapter:
        def synthesize(self, text, voice_external_id, settings_json=None):
            assert settings_json["output_format"] == "pcm_16000"
            # 40 ms of near-silence PCM16 @ 16 kHz -> 640 samples -> 1280 bytes
            return b"\x00\x00" * 640

    fake_agent = SimpleNamespace(
        tts_provider=SimpleNamespace(slug="elevenlabs"),
        tts_voice=SimpleNamespace(external_voice_id="voice-eleven"),
        tts_settings_json={
            "eleven_background": "soft_noise",
            "eleven_background_level": 0.25,
        },
    )

    fake_adapter = _FakeAdapter()
    with patch("app.services.bidirectional_stream_service.get_tts_adapter", return_value=fake_adapter):
        audio = asyncio.run(
            generate_mulaw_tts(
                text="Hello with bed",
                lang="en",
                voice="female",
                agent=fake_agent,
            )
        )

    assert len(audio) > 0
    assert audio != b"\xff" * len(audio)


def test_generate_mulaw_tts_separate_cache_entries_per_background():
    audio_cache.clear()
    calls = {"n": 0}

    class _FakeAdapter:
        def synthesize(self, text, voice_external_id, settings_json=None):
            calls["n"] += 1
            assert settings_json["output_format"] == "pcm_16000"
            return b"\x00\x00" * 640

    adapter = _FakeAdapter()
    agent_a = SimpleNamespace(
        tts_provider=SimpleNamespace(slug="elevenlabs"),
        tts_voice=SimpleNamespace(external_voice_id="v1"),
        tts_settings_json={"eleven_background": "soft_noise", "eleven_background_level": 0.12},
    )
    agent_b = SimpleNamespace(
        tts_provider=SimpleNamespace(slug="elevenlabs"),
        tts_voice=SimpleNamespace(external_voice_id="v1"),
        tts_settings_json={"eleven_background": "cafe", "eleven_background_level": 0.18},
    )

    with patch("app.services.bidirectional_stream_service.get_tts_adapter", return_value=adapter):
        asyncio.run(generate_mulaw_tts(text="Same text", lang="en", voice="female", agent=agent_a))
        asyncio.run(generate_mulaw_tts(text="Same text", lang="en", voice="female", agent=agent_b))

    assert calls["n"] == 2


def test_ensure_default_provider_seeds_rime(tts_db):
    """ensure_default_provider must create a 'rime' provider row idempotently."""
    db, _, _ = tts_db

    # First call: seeds all three default providers including rime.
    tts_catalog_service.ensure_default_provider(db)
    provider = tts_catalog_service.get_provider_by_slug(db, "rime")
    assert provider is not None, "rime provider must be present after default seeding"
    assert provider.is_active is True
    assert provider.supports_streaming is True

    # Second call: must not raise or create a duplicate.
    tts_catalog_service.ensure_default_provider(db)
    count = db.query(TTSProvider).filter(TTSProvider.slug == "rime").count()
    assert count == 1, "ensure_default_provider must be idempotent (no duplicate rime rows)"
