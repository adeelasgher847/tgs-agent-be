import contextvars
import os
import sys
import sqlite3
from unittest.mock import MagicMock

# Rime TTS is validated at app startup; tests must not depend on a developer .env file.
os.environ["RATE_LIMIT_ENABLED"] = "False"
os.environ.setdefault("RIME_API_KEY", "test-rime-key-for-pytest")
os.environ.setdefault(
    "ELEVENLABS_ENCRYPTION_KEY",
    "test-elevenlabs-encryption-key-for-pytest-only",
)
os.environ.setdefault(
    "WEBHOOK_SECRET_ENCRYPTION_KEY",
    "test-webhook-encryption-key-for-pytest-only",
)

# Mock google submodules recursively to avoid ImportError in unit tests.
# Live Google STT integration tests set RUN_GOOGLE_STT_INTEGRATION=1 to skip mocks.
_RUN_GOOGLE_STT_INTEGRATION = os.environ.get(
    "RUN_GOOGLE_STT_INTEGRATION", ""
).lower() in ("1", "true", "yes")

if not _RUN_GOOGLE_STT_INTEGRATION:
    sys.modules["google"] = MagicMock()
    sys.modules["google.genai"] = MagicMock()
    sys.modules["google.oauth2"] = MagicMock()
    sys.modules["google.oauth2"].id_token = MagicMock()
    sys.modules["google.auth"] = MagicMock()
    sys.modules["google.auth"].transport = MagicMock()
    sys.modules["google.auth.transport"] = MagicMock()
    sys.modules["google.auth.transport"].requests = MagicMock()
    sys.modules["google.auth.transport.requests"] = MagicMock()
    sys.modules["google.cloud"] = MagicMock()
    sys.modules["google.cloud"].speech = MagicMock()
    sys.modules["google.cloud.speech_v1p1beta1"] = MagicMock()
    sys.modules["google.cloud.speech_v1p1beta1"].types = MagicMock()
    sys.modules["google.api_core"] = MagicMock()
    sys.modules["google.api_core"].exceptions = MagicMock()
    sys.modules["google.api_core.client_options"] = MagicMock()
    sys.modules["google.api_core.client_options"].ClientOptions = MagicMock()
    sys.modules["google.api_core.exceptions"] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.db.session import SessionLocal
from app.models.user import User
from app.models.role import Role
from app.models.tenant import Tenant


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that require a live PostgreSQL database",
    )
    config.addinivalue_line(
        "markers",
        "google_stt_live: live Google Cloud STT tests (RUN_GOOGLE_STT_INTEGRATION=1)",
    )


# ---------------------------------------------------------------------------
# SQLite engine — used by all unit/functional tests (no Postgres required)
#
# A single sqlite3 connection is shared across ALL threads (including the
# AnyIO worker threads that FastAPI uses to run sync dependencies).  This
# ensures every session — whether created in the db fixture (MainThread) or
# in override_get_db (AnyIO worker thread) — operates on the exact same
# in-memory database, so DDL (drop_all / create_all) and DML written in one
# thread are immediately visible to the other.
# ---------------------------------------------------------------------------

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


# Single shared raw connection — thread-safe because check_same_thread=False.
_shared_sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)

# Per-context test session for override_get_db (safe with pytest-xdist / AnyIO threads).
_active_test_session_var: contextvars.ContextVar = contextvars.ContextVar(
    "_active_test_session", default=None
)

engine = create_engine(
    "sqlite://",
    creator=lambda: _shared_sqlite_conn,
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    session = _active_test_session_var.get()
    if session is not None:
        yield session
        return
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


from app.api.deps import get_db
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def db():
    """Fresh SQLite database for the entire test module."""
    _shared_sqlite_conn.rollback()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    session_token = _active_test_session_var.set(db)
    try:
        admin_role = Role(name="admin", description="Administrator role")
        db.add(admin_role)
        db.commit()
        db.refresh(admin_role)

        user_role = Role(name="user", description="Regular user role")
        db.add(user_role)
        db.commit()
        db.refresh(user_role)

        manager_role = Role(name="manager", description="Manager role")
        config_role = Role(name="config_only", description="Config only role")
        readonly_role = Role(name="read_only", description="Read only role")
        billing_role = Role(name="billing_only", description="Billing only role")
        db.add_all([manager_role, config_role, readonly_role, billing_role])
        db.commit()

        test_tenant = Tenant(name="Test Tenant", schema_name="test_tenant_schema")
        db.add(test_tenant)
        db.commit()
        db.refresh(test_tenant)

        test_user = User(
            email="test@example.com",
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4tbQJbqK8O",
            first_name="Test",
            last_name="User",
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        test_user.tenants.append(test_tenant)
        test_user.current_tenant_id = test_tenant.id
        db.commit()

        # Seed catalog data for Schema v2 validations in unit tests
        from app.models.provider import Provider
        from app.models.model import Model
        from app.models.stt_provider import STTProvider
        from app.models.stt_model import STTModel
        from app.models.tts_provider import TTSProvider
        from app.models.tts_voice import TTSVoice

        # Seed LLM Providers and Models
        openai_p = Provider(name="openai", is_active=True)
        gemini_p = Provider(name="gemini", is_active=True)
        groq_p = Provider(name="groq", is_active=True)
        db.add_all([openai_p, gemini_p, groq_p])
        db.commit()

        db.add_all([
            Model(provider_id=openai_p.id, model_name="gpt-4o-mini", archive=False),
            Model(provider_id=openai_p.id, model_name="gpt-4o", archive=False),
            Model(provider_id=openai_p.id, model_name="gpt-4.1", archive=False),
            Model(provider_id=openai_p.id, model_name="gpt-4.1-mini", archive=False),
            Model(provider_id=openai_p.id, model_name="gpt-4-turbo", archive=False),
            Model(provider_id=gemini_p.id, model_name="gemini-2.5-flash", archive=False),
            Model(provider_id=gemini_p.id, model_name="gemini-2.0-flash-001", archive=False),
            Model(provider_id=gemini_p.id, model_name="gemini-2.0-flash", archive=False),
            Model(provider_id=gemini_p.id, model_name="gemini-1.5-pro", archive=False),
            Model(provider_id=gemini_p.id, model_name="gemini-1.5-flash", archive=False),
            Model(provider_id=gemini_p.id, model_name="claude-3-5-sonnet", archive=False),
            Model(provider_id=gemini_p.id, model_name="claude-3-haiku", archive=False),
            Model(provider_id=groq_p.id, model_name="llama-3.1-70b-versatile", archive=False),
            Model(provider_id=groq_p.id, model_name="llama-3.1-8b-instant", archive=False),
        ])
        db.commit()

        # Seed STT Providers and Models
        stt_deepgram = STTProvider(slug="deepgram", display_name="Deepgram", is_active=True)
        stt_google = STTProvider(slug="google", display_name="Google Cloud STT", is_active=True)
        db.add_all([stt_deepgram, stt_google])
        db.commit()

        db.add_all([
            STTModel(provider_id=stt_deepgram.id, external_model_id="nova-3", display_name="Nova-3", language_code="en", is_active=True),
            STTModel(provider_id=stt_google.id, external_model_id="chirp-3", display_name="Chirp-3", language_code="en", is_active=True),
        ])
        db.commit()

        # Seed TTS Providers and Voices
        tts_eleven = TTSProvider(slug="elevenlabs", display_name="ElevenLabs", is_active=True)
        tts_rime = TTSProvider(slug="rime", display_name="Rime", is_active=True)
        tts_cartesia = TTSProvider(slug="cartesia", display_name="Cartesia", is_active=True)
        db.add_all([tts_eleven, tts_rime, tts_cartesia])
        db.commit()

        db.add_all([
            TTSVoice(provider_id=tts_eleven.id, external_voice_id="21m00Tcm4TlvDq8ikWAM", display_name="Rachel", language_code="en", is_active=True),
            TTSVoice(provider_id=tts_eleven.id, external_voice_id="voice-1", display_name="Voice 1", language_code="en", is_active=True),
            TTSVoice(provider_id=tts_eleven.id, external_voice_id="EXAVITQu4vr4xnSDxMaL", display_name="Rachel 2", language_code="en", is_active=True),
            TTSVoice(provider_id=tts_eleven.id, external_voice_id="vY", display_name="Voice Y", language_code="en", is_active=True),
            TTSVoice(provider_id=tts_eleven.id, external_voice_id="vZ", display_name="Voice Z", language_code="en", is_active=True),
            TTSVoice(provider_id=tts_rime.id, external_voice_id="rime-voice-1", display_name="Rime Voice 1", language_code="en", is_active=True),
            TTSVoice(provider_id=tts_cartesia.id, external_voice_id="cartesia-voice-1", display_name="Cartesia Voice 1", language_code="en", is_active=True),
        ])
        db.commit()

        yield db
    finally:
        _active_test_session_var.reset(session_token)
        db.close()


@pytest.fixture(scope="module")
def client(db):
    """TestClient sharing the same SQLite database session for the module."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# PostgreSQL integration fixtures — only active when TEST_DATABASE_URL is set.
#
# Uses an isolated schema (test_sprint1_<pid>) so parallel runs don't collide.
# The schema is dropped on teardown regardless of test outcome.
# ---------------------------------------------------------------------------

_TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
_PG_AVAILABLE = bool(_TEST_DATABASE_URL)
_pg_test_schema: str | None = None

_INTEGRATION_SKIP = pytest.mark.skipif(
    not _PG_AVAILABLE,
    reason="TEST_DATABASE_URL not set — skipping integration tests",
)


@pytest.fixture(scope="session")
def pg_engine():
    """Session-scoped Postgres engine pointed at the isolated test schema."""
    global _pg_test_schema
    if not _PG_AVAILABLE:
        pytest.skip("TEST_DATABASE_URL not set")

    import uuid as _uuid
    schema = f"test_sprint1_{os.getpid()}_{_uuid.uuid4().hex[:6]}"
    _pg_test_schema = schema

    engine = create_engine(_TEST_DATABASE_URL, echo=False)

    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.execute(text(f'SET search_path TO "{schema}", public'))
        conn.commit()

    schema_engine = create_engine(
        _TEST_DATABASE_URL,
        echo=False,
        connect_args={"options": f"-csearch_path={schema},public"},
    )

    Base.metadata.create_all(bind=schema_engine)

    yield schema_engine

    Base.metadata.drop_all(bind=schema_engine)
    schema_engine.dispose()

    with engine.connect() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        conn.commit()
    engine.dispose()
    _pg_test_schema = None


@pytest.fixture(scope="module")
def pg_auth_middleware(pg_engine):
    """
    Point ApiKeyMiddleware async lookups at the same isolated Postgres schema
    as pg_engine so real x-api-key auth works without patching _resolve_api_key.
    """
    import asyncio

    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    import app.middleware.api_key_middleware as auth_mw
    from app.db.async_url import database_url_to_async

    if not _pg_test_schema:
        pytest.skip("Postgres test schema not initialized")

    schema = _pg_test_schema
    async_url = database_url_to_async(_TEST_DATABASE_URL)
    async_engine = create_async_engine(async_url, pool_pre_ping=True)

    @event.listens_for(async_engine.sync_engine, "connect")
    def _set_search_path(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute(f'SET search_path TO "{schema}", public')
        cursor.close()

    session_factory = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    prev_engine = auth_mw._async_engine
    prev_factory = auth_mw._AsyncSessionLocal
    prev_redis = auth_mw._redis
    auth_mw._async_engine = async_engine
    auth_mw._AsyncSessionLocal = session_factory
    auth_mw._redis = None

    yield

    auth_mw._async_engine = prev_engine
    auth_mw._AsyncSessionLocal = prev_factory
    auth_mw._redis = prev_redis
    try:
        asyncio.run(async_engine.dispose())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_engine.dispose())
        loop.close()


@pytest.fixture(scope="session")
def pg_session_factory(pg_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)


@pytest.fixture()
def pg_session(pg_session_factory):
    """Per-test Postgres session; rolls back after each test for isolation."""
    session = pg_session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="module")
def pg_client(pg_engine, pg_auth_middleware, pg_session_factory):
    """
    Session-scoped TestClient wired to the Postgres test schema.

    The ``get_db`` override is applied only within this fixture's scope
    and does not affect the SQLite unit-test override.
    """
    def _pg_get_db():
        db = pg_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _pg_get_db
    with TestClient(app) as c:
        yield c
    # Restore the default SQLite override for subsequent unit tests.
    app.dependency_overrides[get_db] = override_get_db
