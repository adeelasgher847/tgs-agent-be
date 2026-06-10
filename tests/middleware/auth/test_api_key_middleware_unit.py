"""
Unit tests for ApiKeyMiddleware logic.

All DB and Redis I/O is mocked so these tests run without any external services.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.workspace import Workspace

from app.core.security import create_user_token
from app.middleware.api_key_middleware import (
    ApiKeyMiddleware,
    _sha256,
    invalidate_api_key_cache,
    invalidate_api_key_cache_by_hash,
)
from app.middleware.request_id_middleware import RequestIdMiddleware

_UNAUTHORIZED = {
    "error": {
        "code": "unauthorized",
        "message": "Invalid or missing API key",
    }
}


# ---------------------------------------------------------------------------
# Minimal app
# ---------------------------------------------------------------------------

def _app() -> FastAPI:
    mini = FastAPI()
    mini.add_middleware(ApiKeyMiddleware)
    mini.add_middleware(RequestIdMiddleware)

    @mini.get("/api/v1/protected")
    def protected(request: Request):
        ws = request.state.workspace
        return {
            "ok": True,
            "workspace_id": str(ws.id),
            "workspace_name": ws.name,
            "auth_method": request.state.auth_method,
            "request_id": getattr(request.state, "request_id", None),
        }

    @mini.get("/api/v1/auth/clickup/callback")
    def public_oauth():
        return {"ok": True}

    @mini.post("/api/v1/users/login")
    def public_login():
        return {"ok": True}

    @mini.post("/api/v1/users/register")
    def public_register():
        return {"ok": True}

    @mini.post("/api/v1/users/forgot-password")
    def public_forgot():
        return {"ok": True}

    @mini.post("/api/v1/accept-invite")
    def public_accept_invite():
        return {"ok": True}

    @mini.post("/api/v1/tenants/create")
    def public_tenant_create():
        return {"ok": True}

    @mini.get("/api/v1/plans/public")
    def public_plans():
        return {"ok": True}

    @mini.post("/api/v1/api-keys/")
    def public_api_keys():
        return {"ok": True}

    @mini.get("/health")
    def health():
        return {"ok": True}

    @mini.get("/api/v2/protected")
    def v2_protected(request: Request):
        ws = request.state.workspace
        return {
            "ok": True,
            "workspace_id": str(ws.id),
            "auth_method": request.state.auth_method,
        }

    @mini.get("/api/v2/health")
    def v2_health():
        return {"status": "ok"}

    return mini


def _assert_unauthorized(resp) -> None:
    """All 401s from the middleware must share the canonical error envelope."""
    assert resp.status_code == 401
    data = resp.json()
    # Verify envelope shape without requiring requestId to be a specific value
    # (RequestIdMiddleware is not wired in these unit-test apps).
    assert data["error"]["code"] == _UNAUTHORIZED["error"]["code"]
    assert data["error"]["message"] == _UNAUTHORIZED["error"]["message"]
    assert "requestId" in data["error"]
    assert "x-request-id" in resp.headers


def _bearer_token(user_id: uuid.UUID, tenant_id: uuid.UUID) -> str:
    return create_user_token(
        user_id=user_id,
        email="test@example.com",
        tenant_id=tenant_id,
        role="admin",
    )


@pytest.fixture
def client():
    return TestClient(_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _workspace_dict(
    tenant_id: str | uuid.UUID,
    *,
    status: str = "active",
    name: str = "Test WS",
) -> dict:
    tid = str(tenant_id)
    return {
        "id": tid,
        "name": name,
        "schema_name": "test_schema",
        "status": status,
        "credits": 10.0,
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
    }


def _valid_payload(tenant_id: str | uuid.UUID, key_id: str | uuid.UUID) -> dict:
    return {
        "api_key_id": str(key_id),
        "tenant_id": str(tenant_id),
        "key_is_active": True,
        "workspace": _workspace_dict(tenant_id),
    }


# ---------------------------------------------------------------------------
# Public / skip paths — no auth required
# ---------------------------------------------------------------------------

class TestSkipPaths:
    def test_options_preflight_not_blocked_by_auth(self, client):
        """Browser CORS preflight must not receive 401 from ApiKey middleware."""
        resp = client.options("/api/v1/protected")
        assert resp.status_code != 401

    def test_health_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_v2_health_no_auth(self, client):
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200

    def test_clickup_oauth_no_auth(self, client):
        resp = client.get("/api/v1/auth/clickup/callback")
        assert resp.status_code == 200

    def test_user_login_no_auth(self, client):
        resp = client.post("/api/v1/users/login")
        assert resp.status_code == 200

    def test_user_register_no_auth(self, client):
        resp = client.post("/api/v1/users/register")
        assert resp.status_code == 200

    def test_user_forgot_password_no_auth(self, client):
        resp = client.post("/api/v1/users/forgot-password")
        assert resp.status_code == 200

    def test_accept_invite_no_auth(self, client):
        resp = client.post("/api/v1/accept-invite")
        assert resp.status_code == 200

    def test_tenant_create_no_auth(self, client):
        resp = client.post("/api/v1/tenants/create")
        assert resp.status_code == 200

    def test_plans_public_no_auth(self, client):
        resp = client.get("/api/v1/plans/public")
        assert resp.status_code == 200

    def test_api_keys_crud_no_auth(self, client):
        resp = client.post("/api/v1/api-keys/")
        assert resp.status_code == 200

    def test_root_no_auth(self, client):
        resp = client.get("/")
        # FastAPI 404 is fine — the point is middleware does not block it
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Missing headers
# ---------------------------------------------------------------------------

class TestMissingHeaders:
    def test_missing_both_headers_and_jwt(self, client):
        resp = client.get("/api/v1/protected")
        _assert_unauthorized(resp)

    def test_partial_api_key_headers_without_jwt(self, client):
        resp = client.get("/api/v1/protected", headers={"x-workspace-id": str(uuid.uuid4())})
        _assert_unauthorized(resp)

    def test_partial_api_key_only_without_jwt(self, client):
        resp = client.get("/api/v1/protected", headers={"x-api-key": "somekey"})
        _assert_unauthorized(resp)

    def test_invalid_workspace_uuid(self, client):
        resp = client.get(
            "/api/v1/protected",
            headers={"x-api-key": "somekey", "x-workspace-id": "not-a-uuid"},
        )
        _assert_unauthorized(resp)


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
        body = resp.json()
        assert body["workspace_id"] == str(tenant_id)
        assert body["workspace_name"] == "Test WS"
        assert body["auth_method"] == "api_key"

    def test_x_request_id_injected(self, client):
        tenant_id = uuid.uuid4()
        key_id = uuid.uuid4()

        async def _resolve(kh, workspace_id):
            return _valid_payload(tenant_id, key_id)

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "key", "x-workspace-id": str(tenant_id)},
            )
        assert resp.status_code == 200
        assert resp.json()["request_id"]

    def test_custom_request_id_echoed(self, client):
        tenant_id = uuid.uuid4()
        custom_id = "custom-nanoid-req-id01"

        async def _resolve(kh, workspace_id):
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
        assert resp.status_code == 200
        assert resp.json()["request_id"] == custom_id


# ---------------------------------------------------------------------------
# Invalid / non-existent key
# ---------------------------------------------------------------------------

class TestInvalidKey:
    def test_unknown_key_returns_401(self, client):
        async def _resolve(kh, workspace_id):
            return None  # not found

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "bad-key", "x-workspace-id": str(uuid.uuid4())},
            )
        _assert_unauthorized(resp)


# ---------------------------------------------------------------------------
# Revoked key
# ---------------------------------------------------------------------------

class TestRevokedKey:
    def test_revoked_key_returns_401(self, client):
        """Revoked key must 401 with the generic body (no enumeration signal)."""
        tenant_id = uuid.uuid4()

        async def _resolve(kh, workspace_id):
            return {
                "api_key_id": str(uuid.uuid4()),
                "tenant_id": str(tenant_id),
                "key_is_active": False,
                "workspace": _workspace_dict(tenant_id),
            }

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "revoked", "x-workspace-id": str(tenant_id)},
            )
        _assert_unauthorized(resp)


# ---------------------------------------------------------------------------
# Workspace mismatch
# ---------------------------------------------------------------------------

class TestWorkspaceMismatch:
    def test_wrong_workspace_returns_401(self, client):
        """Key from workspace A presented with workspace B's id must 401."""
        real_tenant_id = uuid.uuid4()
        other_tenant_id = uuid.uuid4()

        async def _resolve(kh, workspace_id):
            return _valid_payload(real_tenant_id, uuid.uuid4())

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "key", "x-workspace-id": str(other_tenant_id)},
            )
        _assert_unauthorized(resp)


# ---------------------------------------------------------------------------
# Inactive workspace
# ---------------------------------------------------------------------------

class TestInactiveWorkspace:
    def test_inactive_tenant_returns_401(self, client):
        tenant_id = uuid.uuid4()

        async def _resolve(kh, workspace_id):
            return {
                "api_key_id": str(uuid.uuid4()),
                "tenant_id": str(tenant_id),
                "key_is_active": True,
                "workspace": _workspace_dict(tenant_id, status="pending_payment"),
            }

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            resp = client.get(
                "/api/v1/protected",
                headers={"x-api-key": "key", "x-workspace-id": str(tenant_id)},
            )
        _assert_unauthorized(resp)


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


# ---------------------------------------------------------------------------
# Public cache invalidation helper
# ---------------------------------------------------------------------------

class TestJwtAuth:
    def test_valid_jwt_without_api_key_headers(self, client):
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        token = _bearer_token(user_id, tenant_id)
        workspace = Workspace.from_mapping(_workspace_dict(tenant_id))

        async def _load(wid):
            return workspace

        with patch("app.middleware.api_key_middleware._load_workspace", side_effect=_load):
            resp = client.get(
                "/api/v1/protected",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace_id"] == str(tenant_id)
        assert body["auth_method"] == "jwt"

    def test_valid_jwt_on_v2_protected_route(self, client):
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        token = _bearer_token(user_id, tenant_id)
        workspace = Workspace.from_mapping(_workspace_dict(tenant_id))

        async def _load(wid):
            return workspace

        with patch("app.middleware.api_key_middleware._load_workspace", side_effect=_load):
            resp = client.get(
                "/api/v2/protected",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["auth_method"] == "jwt"

    def test_invalid_api_key_headers_with_valid_jwt(self, client):
        """Invalid API key attempt must fall through to JWT."""
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        token = _bearer_token(user_id, tenant_id)
        workspace = Workspace.from_mapping(_workspace_dict(tenant_id))

        async def _resolve(kh, workspace_id):
            return None

        async def _load(wid):
            return workspace

        with patch("app.middleware.api_key_middleware._resolve_api_key", side_effect=_resolve):
            with patch("app.middleware.api_key_middleware._load_workspace", side_effect=_load):
                resp = client.get(
                    "/api/v1/protected",
                    headers={
                        "x-api-key": "bad",
                        "x-workspace-id": str(tenant_id),
                        "Authorization": f"Bearer {token}",
                    },
                )
        assert resp.status_code == 200
        assert resp.json()["auth_method"] == "jwt"

    def test_invalid_jwt_returns_401(self, client):
        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Bearer not-a-valid-token"},
        )
        _assert_unauthorized(resp)


class TestInvalidateApiKeyCache:
    def test_invalidate_hashes_then_deletes(self):
        """`invalidate_api_key_cache(raw, workspace)` must delete composite cache key."""
        raw = "some-raw-key"
        workspace_id = uuid.uuid4()
        with patch(
            "app.middleware.api_key_middleware._apikey_cache_delete",
            new_callable=AsyncMock,
        ) as mock_delete:
            asyncio.run(invalidate_api_key_cache(raw, workspace_id))
        mock_delete.assert_awaited_once_with(_sha256(raw), workspace_id)

    def test_invalidate_by_hash_deletes_directly(self):
        key_hash = "abc123hash"
        workspace_id = uuid.uuid4()
        with patch(
            "app.middleware.api_key_middleware._apikey_cache_delete",
            new_callable=AsyncMock,
        ) as mock_delete:
            asyncio.run(invalidate_api_key_cache_by_hash(key_hash, workspace_id))
        mock_delete.assert_awaited_once_with(key_hash, workspace_id)
