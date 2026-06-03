"""Integration tests for the workspace (tenant) CRUD endpoints.

The middleware is bypassed for protected routes by mocking
``_resolve_api_key`` so the request reaches the endpoint with a real
``Workspace`` context attached to ``request.state``.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.request_auth import AUTH_METHOD_JWT
from app.core.workspace import Workspace
from app.middleware.api_key_middleware import _attach_workspace_context
from app.models.tenant import Tenant
_API_KEY = "test-workspace-key"


def _payload_for(tenant: Tenant) -> dict:
    return {
        "api_key_id": str(uuid.uuid4()),
        "tenant_id": str(tenant.id),
        "key_is_active": True,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": tenant.schema_name,
            "status": "active",
            "credits": 0.0,
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
        },
    }


def _headers(tenant: Tenant) -> dict:
    return {"x-api-key": _API_KEY, "x-workspace-id": str(tenant.id)}


@pytest.fixture
def auth_tenant(db) -> Tenant:
    t = Tenant(
        name=f"AuthWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"auth_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def authed_client(client: TestClient, auth_tenant: Tenant):
    payload = _payload_for(auth_tenant)

    async def _resolve(_key_hash, _workspace_id):
        return payload

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield client


@pytest.mark.usefixtures("db")
class TestCreateWorkspace:
    def test_create_returns_201_with_id_name_createdAt(self, authed_client, auth_tenant):
        name = f"NewWS-{uuid.uuid4().hex[:8]}"
        resp = authed_client.post(
            "/api/v1/workspace",
            json={"name": name},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert "data" not in body
        assert body["name"] == name
        assert "id" in body
        assert "createdAt" in body
        assert "deleted_at" not in body
        assert "schema_name" not in body

    def test_create_too_short_returns_400(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace",
            json={"name": "ab"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_too_long_returns_400(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace",
            json={"name": "x" * 51},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_missing_name_returns_400(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace",
            json={},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_duplicate_name_returns_409(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace",
            json={"name": auth_tenant.name},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 409


@pytest.mark.usefixtures("db")
class TestGetWorkspace:
    def test_get_own_returns_200(self, authed_client, auth_tenant):
        resp = authed_client.get(
            f"/api/v1/workspace/{auth_tenant.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["id"] == str(auth_tenant.id)
        assert data["name"] == auth_tenant.name
        assert "createdAt" in data
        assert "deleted_at" not in data
        assert "schema_name" not in data

    def test_get_other_workspace_returns_403(self, authed_client, auth_tenant):
        other_id = uuid.uuid4()
        resp = authed_client.get(
            f"/api/v1/workspace/{other_id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 403

    def test_get_nonexistent_workspace_returns_404(self, client, db):
        """Valid auth with a workspace that was soft-deleted → 404."""
        # Create a workspace, soft-delete it so find_by_id returns None, then assert 404.
        non_existent = Tenant(
            name=f"Gone-{uuid.uuid4().hex[:8]}",
            schema_name=f"gone_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(non_existent)
        db.commit()
        db.refresh(non_existent)

        payload = _payload_for(non_existent)

        # Soft-delete the tenant so the repository can't find it.
        from datetime import datetime, timezone
        non_existent.deleted_at = datetime.now(timezone.utc)
        db.commit()

        async def _resolve(_key_hash, _workspace_id):
            return payload

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            resp = client.get(
                f"/api/v1/workspace/{non_existent.id}",
                headers=_headers(non_existent),
            )
        assert resp.status_code == 404, resp.text


@pytest.mark.usefixtures("db")
class TestUpdateWorkspaceName:
    def test_put_updates_name(self, authed_client, auth_tenant):
        new_name = f"Renamed-{uuid.uuid4().hex[:6]}"
        resp = authed_client.put(
            "/api/v1/workspace/name",
            json={"name": new_name},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["name"] == new_name

    def test_put_duplicate_returns_409(self, authed_client, auth_tenant, db):
        other = Tenant(
            name=f"DupTarget-{uuid.uuid4().hex[:6]}",
            schema_name=f"dup_target_{uuid.uuid4().hex[:6]}",
            status="active",
        )
        db.add(other)
        db.commit()
        db.refresh(other)

        resp = authed_client.put(
            "/api/v1/workspace/name",
            json={"name": other.name},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 409

    def test_put_validation_returns_400(self, authed_client, auth_tenant):
        resp = authed_client.put(
            "/api/v1/workspace/name",
            json={"name": "ab"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400


@pytest.mark.usefixtures("db")
class TestDeleteWorkspace:
    def test_delete_other_returns_403(self, authed_client, auth_tenant):
        resp = authed_client.delete(
            f"/api/v1/workspace/{uuid.uuid4()}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 403

    def test_delete_own_returns_204_then_get_404(self, client, db):
        # Build an isolated tenant + auth context for this test so the
        # 'deleted' state doesn't bleed into other test methods.
        t = Tenant(
            name=f"DelMe-{uuid.uuid4().hex[:8]}",
            schema_name=f"del_me_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(t)
        db.commit()
        db.refresh(t)

        payload = _payload_for(t)

        async def _resolve(_key_hash, _workspace_id):
            return payload

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            resp = client.delete(
                f"/api/v1/workspace/{t.id}",
                headers=_headers(t),
            )
            assert resp.status_code == 204, resp.text

            follow_up = client.get(
                f"/api/v1/workspace/{t.id}",
                headers=_headers(t),
            )
            assert follow_up.status_code == 404


@pytest.mark.usefixtures("db")
class TestWorkspaceAuth:
    def test_missing_api_key_returns_401(self, client, auth_tenant):
        resp = client.get(f"/api/v1/workspace/{auth_tenant.id}")
        assert resp.status_code == 401

    def test_missing_workspace_header_returns_401(self, client, auth_tenant):
        resp = client.get(
            f"/api/v1/workspace/{auth_tenant.id}",
            headers={"x-api-key": _API_KEY},
        )
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, client, auth_tenant):
        async def _resolve(_key_hash, _workspace_id):
            return None

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            resp = client.get(
                f"/api/v1/workspace/{auth_tenant.id}",
                headers=_headers(auth_tenant),
            )
        assert resp.status_code == 401

    def test_jwt_auth_returns_403(self, client, auth_tenant):
        workspace = Workspace.from_tenant(auth_tenant)

        async def _jwt_auth(request):
            _attach_workspace_context(
                request,
                workspace=workspace,
                auth_method=AUTH_METHOD_JWT,
                user_id=uuid.uuid4(),
            )
            return True

        with patch(
            "app.middleware.api_key_middleware._try_api_key_auth",
            new_callable=AsyncMock,
            return_value=False,
        ), patch(
            "app.middleware.api_key_middleware._try_jwt_auth",
            side_effect=_jwt_auth,
        ):
            resp = client.get(
                f"/api/v1/workspace/{auth_tenant.id}",
                headers={
                    "Authorization": "Bearer test-token",
                    "x-workspace-id": str(auth_tenant.id),
                },
            )
        assert resp.status_code == 403
