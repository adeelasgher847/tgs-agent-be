"""
Unit tests for ApiKeyMiddleware logic.

All DB and Redis I/O is mocked so these tests run without any external services.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.api_key_middleware import ApiKeyMiddleware, _sha256


# ---------------------------------------------------------------------------
# Minimal app
# ---------------------------------------------------------------------------

def _app() -> FastAPI:
    mini = FastAPI()
    mini.add_middleware(ApiKeyMiddleware)

    @mini.get("/api/v1/protected")
    def protected():
        return {"ok": True}

    @mini.get("/api/v1/auth/login")
    def public_login():
        return {"ok": True}

    @mini.get("/health")
    def health():
        return {"ok": True}

    return mini


@pytest.fixture
def client():
    return TestClient(_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_payload(tenant_id: str | uuid.UUID, key_id: str | uuid.UUID) -> dict:
    return {
        "api_key_id": str(key_id),
        "tenant_id": str(tenant_id),
        "tenant_status": "active",
        "key_is_active": True,
    }


# ---------------------------------------------------------------------------
# Public / skip paths — no auth required
# ---------------------------------------------------------------------------

class TestSkipPaths:
    def test_health_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_auth_login_no_auth(self, client):
        resp = client.get("/api/v1/auth/login")
        assert resp.status_code == 200

    def test_root_no_auth(self, client):
        resp = client.get("/")
        # FastAPI 404 is fine — the point is middleware does not block it
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Missing headers
# ---------------------------------------------------------------------------

class TestMissingHeaders:
    def test_missing_both_headers(self, client):
        resp = client.get("/api/v1/protected")
        assert resp.status_code == 401
        assert "x-api-key" in resp.json()["detail"].lower() or "missing" in resp.json()["detail"].lower()

    def test_missing_api_key_only(self, client):
        resp = client.get("/api/v1/protected", headers={"x-workspace-id": str(uuid.uuid4())})
        assert resp.status_code == 401

    def test_missing_workspace_id_only(self, client):
        resp = client.get("/api/v1/protected", headers={"x-api-key": "somekey"})
        assert resp.status_code == 401

    def test_invalid_workspace_uuid(self, client):
        resp = client.get(
            "/api/v1/protected",
            headers={"x-api-key": "somekey", "x-workspace-id": "not-a-uuid"},
        )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Valid request
# ---------------------------------------------------------------------------

class TestValidKey:
    def test_valid_key_returns_200(self, client):
        tenant_id = uuid.uuid4()
        key_id = uuid.uuid4()
        raw = "my-valid-key"

        async def _resolve(key_hash, workspace_id):
            return _valid_payload(tenant_id, key_id)

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": raw, "x-workspace-id": str(tenant_id)},
            )
        assert resp.status_code == 200

    def test_x_request_id_injected(self, client):
        tenant_id = uuid.uuid4()
        key_id = uuid.uuid4()

        async def _resolve(kh, wid):
            return _valid_payload(tenant_id, key_id)

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "key", "x-workspace-id": str(tenant_id)},
            )
        assert "x-request-id" in resp.headers

    def test_custom_request_id_echoed(self, client):
        tenant_id = uuid.uuid4()
        custom_id = str(uuid.uuid4())

        async def _resolve(kh, wid):
            return _valid_payload(tenant_id, uuid.uuid4())

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={
                    "x-api-key": "key",
                    "x-workspace-id": str(tenant_id),
                    "x-request-id": custom_id,
                },
            )
        assert resp.headers["x-request-id"] == custom_id


# ---------------------------------------------------------------------------
# Invalid / non-existent key
# ---------------------------------------------------------------------------

class TestInvalidKey:
    def test_unknown_key_returns_401(self, client):
        async def _resolve(kh, wid):
            return None  # not found

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "bad-key", "x-workspace-id": str(uuid.uuid4())},
            )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Revoked key
# ---------------------------------------------------------------------------

class TestRevokedKey:
    def test_revoked_key_returns_401(self, client):
        tenant_id = uuid.uuid4()

        async def _resolve(kh, wid):
            return {
                "api_key_id": str(uuid.uuid4()),
                "tenant_id": str(tenant_id),
                "tenant_status": "active",
                "key_is_active": False,  # revoked
            }

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "revoked", "x-workspace-id": str(tenant_id)},
            )
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Workspace mismatch
# ---------------------------------------------------------------------------

class TestWorkspaceMismatch:
    def test_wrong_workspace_returns_401(self, client):
        real_tenant_id = uuid.uuid4()
        other_tenant_id = uuid.uuid4()

        async def _resolve(kh, wid):
            return _valid_payload(real_tenant_id, uuid.uuid4())

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "key", "x-workspace-id": str(other_tenant_id)},
            )
        assert resp.status_code == 401
        assert "workspace" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Inactive workspace
# ---------------------------------------------------------------------------

class TestInactiveWorkspace:
    def test_inactive_tenant_returns_401(self, client):
        tenant_id = uuid.uuid4()

        async def _resolve(kh, wid):
            return {
                "api_key_id": str(uuid.uuid4()),
                "tenant_id": str(tenant_id),
                "tenant_status": "pending_payment",
                "key_is_active": True,
            }

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "key", "x-workspace-id": str(tenant_id)},
            )
        assert resp.status_code == 401
        assert "not active" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# SHA-256 hashing
# ---------------------------------------------------------------------------

class TestSha256:
    def test_hash_is_64_hex_chars(self):
        h = _sha256("any-key")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_hash(self):
        assert _sha256("key") == _sha256("key")

    def test_different_inputs_different_hash(self):
        assert _sha256("key-a") != _sha256("key-b")
