"""
API key CRUD endpoint tests.

Uses SQLite in-memory DB (from root conftest) and overrides admin auth.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import require_admin_or_owner
from app.main import app
from app.models.tenant import Tenant
from app.models.user import User
from app.services.api_key_service import sha256_hex


@pytest.fixture
def workspace(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(name=f"WS-{suffix}", schema_name=f"ws_{suffix}", status="active")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def admin_user(db, workspace) -> User:
    u = User(
        email=f"admin-{uuid.uuid4().hex[:8]}@test.com",
        first_name="Admin",
        last_name="User",
        hashed_password="",
        current_tenant_id=workspace.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def authed_client(client: TestClient, admin_user: User):
    app.dependency_overrides[require_admin_or_owner] = lambda: admin_user
    yield client
    app.dependency_overrides.pop(require_admin_or_owner, None)


@pytest.mark.usefixtures("db")
class TestApiKeyEndpoints:
    def test_create_returns_raw_key_once(self, authed_client: TestClient, workspace):
        resp = authed_client.post(
            "/api/v1/api-keys/",
            json={"name": "Integration key"},
            headers={"Authorization": "Bearer fake"},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        assert data["name"] == "Integration key"
        assert data["workspace_id"] == str(workspace.id)
        assert data["raw_key"].startswith("sk_")
        assert "••••" in data["masked_key"]
        assert data["is_active"] is True

    def test_list_never_includes_raw_key(self, authed_client: TestClient):
        authed_client.post(
            "/api/v1/api-keys/",
            json={"name": "Listed key"},
            headers={"Authorization": "Bearer fake"},
        )
        resp = authed_client.get(
            "/api/v1/api-keys/",
            headers={"Authorization": "Bearer fake"},
        )
        assert resp.status_code == 200, resp.text
        keys = resp.json()["data"]
        assert len(keys) >= 1
        for item in keys:
            assert "raw_key" not in item
            assert "masked_key" in item

    def test_revoke_invalidates_cache_and_deactivates(self, authed_client: TestClient):
        create_resp = authed_client.post(
            "/api/v1/api-keys/",
            json={"name": "To revoke"},
            headers={"Authorization": "Bearer fake"},
        )
        created = create_resp.json()["data"]
        key_id = created["id"]
        key_hash = sha256_hex(created["raw_key"])
        workspace_id = uuid.UUID(created["workspace_id"])

        with patch(
            "app.api.api_v1.endpoints.api_keys.invalidate_api_key_cache_by_hash",
            new_callable=AsyncMock,
        ) as mock_invalidate:
            resp = authed_client.delete(
                f"/api/v1/api-keys/{key_id}",
                headers={"Authorization": "Bearer fake"},
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["is_active"] is False
        mock_invalidate.assert_awaited_once_with(key_hash, workspace_id)

    def test_revoke_unknown_key_404(self, authed_client: TestClient):
        resp = authed_client.delete(
            f"/api/v1/api-keys/{uuid.uuid4()}",
            headers={"Authorization": "Bearer fake"},
        )
        assert resp.status_code == 404

    def test_create_requires_name(self, authed_client: TestClient):
        resp = authed_client.post(
            "/api/v1/api-keys/",
            json={"name": ""},
            headers={"Authorization": "Bearer fake"},
        )
        assert resp.status_code == 422
