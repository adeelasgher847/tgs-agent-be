"""
Shared fixtures for API key middleware tests.

Uses the same SQLite in-memory DB setup as the root conftest but creates
isolated tables per test function so tests don't share state.
"""
from __future__ import annotations

import hashlib
import sys
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

# ── Google stubs (mirrors root conftest) ──────────────────────────────────────
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", MagicMock())
sys.modules.setdefault("google.oauth2", MagicMock())
sys.modules.setdefault("google.auth", MagicMock())
sys.modules.setdefault("google.auth.transport", MagicMock())
sys.modules.setdefault("google.auth.transport.requests", MagicMock())
sys.modules.setdefault("google.cloud", MagicMock())
sys.modules.setdefault("google.cloud.speech_v1p1beta1", MagicMock())
sys.modules.setdefault("google.api_core", MagicMock())
sys.modules.setdefault("google.api_core.exceptions", MagicMock())
sys.modules.setdefault("google.api_core.client_options", MagicMock())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base_class import Base
from app.models.tenant import Tenant
from app.models.api_key import Apikey
from app.middleware.api_key_middleware import ApiKeyMiddleware

# ── SQLite in-memory setup ────────────────────────────────────────────────────
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import JSON


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Minimal FastAPI app for middleware tests ──────────────────────────────────

def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with ApiKeyMiddleware wired in."""
    mini = FastAPI()
    mini.add_middleware(ApiKeyMiddleware)

    @mini.get("/api/v1/protected")
    def protected():
        return {"ok": True}

    @mini.get("/api/v1/auth/login")
    def public_auth():
        return {"ok": True}

    @mini.get("/health")
    def health():
        return {"ok": True}

    return mini


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function", autouse=True)
def _reset_db():
    """Drop and recreate tables before each test for isolation."""
    # Only import models needed; Base covers all registered models
    from app.db import base as _  # ensure all models are registered  # noqa: F401
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)


@pytest.fixture
def db_session():
    session = _Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def tenant(db_session):
    t = Tenant(name="ACME Corp", schema_name="acme", status="active")
    db_session.add(t)
    db_session.commit()
    db_session.refresh(t)
    return t


@pytest.fixture
def inactive_tenant(db_session):
    t = Tenant(name="Inactive Corp", schema_name="inactive", status="pending_payment")
    db_session.add(t)
    db_session.commit()
    db_session.refresh(t)
    return t


@pytest.fixture
def raw_key() -> str:
    return "test-secret-api-key-12345"


@pytest.fixture
def api_key_record(db_session, tenant, raw_key):
    """Active API key belonging to `tenant`."""
    record = Apikey(
        tenant_id=tenant.id,
        name="CI key",
        key_hash=_sha256(raw_key),
        is_active=True,
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    return record


@pytest.fixture
def revoked_key_record(db_session, tenant, raw_key):
    """Revoked (is_active=False) API key."""
    record = Apikey(
        tenant_id=tenant.id,
        name="Revoked key",
        key_hash=_sha256(raw_key),
        is_active=False,
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    return record


@pytest.fixture
def mock_redis():
    """
    Replace Redis calls with an in-process dict so tests are hermetic.
    """
    store: dict[str, str] = {}

    async def fake_get(cache_key: str):
        return store.get(cache_key)

    async def fake_setex(cache_key: str, ttl: int, value: str):
        store[cache_key] = value

    async def fake_delete(cache_key: str):
        store.pop(cache_key, None)

    mock = MagicMock()
    mock.get = fake_get
    mock.setex = fake_setex
    mock.delete = fake_delete

    with patch("app.middleware.api_key_middleware._get_redis", return_value=mock):
        yield store


@pytest.fixture
def no_redis():
    """Simulate Redis being unavailable (cache always misses)."""
    with patch("app.middleware.api_key_middleware._get_redis", return_value=None):
        yield


@pytest.fixture
def mock_db_lookup(tenant, api_key_record):
    """
    Patch _resolve_api_key so tests don't need an async DB engine.
    Returns a valid payload for the fixture tenant / api_key_record.
    """
    payload = {
        "api_key_id": str(api_key_record.id),
        "tenant_id": str(tenant.id),
        "key_is_active": api_key_record.is_active,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": tenant.schema_name,
            "status": tenant.status,
            "credits": float(tenant.credits or 0),
            "stripe_customer_id": tenant.stripe_customer_id,
            "stripe_subscription_id": tenant.stripe_subscription_id,
        },
    }

    async def _resolve(key_hash, workspace_id):
        if key_hash == api_key_record.key_hash:
            return payload
        return None

    with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
        yield payload
