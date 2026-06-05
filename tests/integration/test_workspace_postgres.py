"""Workspace CRUD integration tests against real PostgreSQL (isolated schema).

Requires TEST_DATABASE_URL. Uses real API keys — no _resolve_api_key mock.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.models.tenant import Tenant
from app.services.api_key_service import create_api_key, revoke_api_key
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]


def _headers(raw_key: str, tenant_id: uuid.UUID) -> dict[str, str]:
    return {"x-api-key": raw_key, "x-workspace-id": str(tenant_id)}


@pytest.fixture()
def auth_workspace(pg_session):
    """Tenant + real API key committed in the Postgres test schema."""
    tenant = Tenant(
        name=f"PGWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"pg_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    pg_session.add(tenant)
    pg_session.commit()
    pg_session.refresh(tenant)

    _record, raw_key = create_api_key(
        pg_session, tenant_id=tenant.id, name="integration-test"
    )
    return tenant, raw_key


class TestCreateWorkspacePostgres:
    def test_create_returns_201(self, pg_client: TestClient, auth_workspace):
        tenant, raw_key = auth_workspace
        name = f"NewPG-{uuid.uuid4().hex[:8]}"
        resp = pg_client.post(
            "/api/v1/workspace",
            json={"name": name},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == name
        assert "id" in body
        assert "createdAt" in body

    def test_create_too_short_returns_400(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.post(
            "/api/v1/workspace",
            json={"name": "ab"},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 400

    def test_create_missing_name_returns_400(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.post(
            "/api/v1/workspace",
            json={},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 400

    def test_create_duplicate_name_returns_409(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.post(
            "/api/v1/workspace",
            json={"name": tenant.name},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 409


class TestGetWorkspacePostgres:
    def test_get_own_returns_200(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.get(
            f"/api/v1/workspace/{tenant.id}",
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["id"] == str(tenant.id)

    def test_get_other_workspace_returns_403(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.get(
            f"/api/v1/workspace/{uuid.uuid4()}",
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 403

    def test_get_soft_deleted_returns_404(self, pg_client, pg_session, auth_workspace):
        tenant, raw_key = auth_workspace
        tenant.deleted_at = datetime.now(timezone.utc)
        pg_session.commit()

        resp = pg_client.get(
            f"/api/v1/workspace/{tenant.id}",
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 404


class TestUpdateWorkspacePostgres:
    def test_put_updates_name(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        new_name = f"RenamedPG-{uuid.uuid4().hex[:6]}"
        resp = pg_client.put(
            "/api/v1/workspace/name",
            json={"name": new_name},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["name"] == new_name

    def test_put_duplicate_returns_409(self, pg_client, pg_session, auth_workspace):
        tenant, raw_key = auth_workspace
        other = Tenant(
            name=f"DupPG-{uuid.uuid4().hex[:6]}",
            schema_name=f"dup_pg_{uuid.uuid4().hex[:6]}",
            status="active",
        )
        pg_session.add(other)
        pg_session.commit()

        resp = pg_client.put(
            "/api/v1/workspace/name",
            json={"name": other.name},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 409

    def test_put_validation_returns_400(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.put(
            "/api/v1/workspace/name",
            json={"name": "ab"},
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 400


class TestDeleteWorkspacePostgres:
    def test_delete_other_returns_403(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.delete(
            f"/api/v1/workspace/{uuid.uuid4()}",
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 403

    def test_delete_own_returns_204_then_get_404(self, pg_client, pg_session):
        tenant = Tenant(
            name=f"DelPG-{uuid.uuid4().hex[:8]}",
            schema_name=f"del_pg_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        pg_session.add(tenant)
        pg_session.commit()
        pg_session.refresh(tenant)

        _record, raw_key = create_api_key(
            pg_session, tenant_id=tenant.id, name="delete-test"
        )

        resp = pg_client.delete(
            f"/api/v1/workspace/{tenant.id}",
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 204, resp.text

        follow_up = pg_client.get(
            f"/api/v1/workspace/{tenant.id}",
            headers=_headers(raw_key, tenant.id),
        )
        assert follow_up.status_code == 404


class TestWorkspaceAuthPostgres:
    def test_missing_api_key_returns_401(self, pg_client, auth_workspace):
        tenant, _ = auth_workspace
        resp = pg_client.get(f"/api/v1/workspace/{tenant.id}")
        assert resp.status_code == 401

    def test_missing_workspace_header_returns_401(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.get(
            f"/api/v1/workspace/{tenant.id}",
            headers={"x-api-key": raw_key},
        )
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, pg_client, auth_workspace):
        tenant, _ = auth_workspace
        resp = pg_client.get(
            f"/api/v1/workspace/{tenant.id}",
            headers=_headers("sk_invalid_key_not_in_db", tenant.id),
        )
        assert resp.status_code == 401

    def test_revoked_api_key_returns_401(self, pg_client, pg_session, auth_workspace):
        tenant, raw_key = auth_workspace
        from app.models.api_key import Apikey
        from app.services.api_key_service import sha256_hex

        record = (
            pg_session.query(Apikey)
            .filter(
                Apikey.tenant_id == tenant.id,
                Apikey.key_hash == sha256_hex(raw_key),
            )
            .first()
        )
        assert record is not None
        revoke_api_key(pg_session, record)

        resp = pg_client.get(
            f"/api/v1/workspace/{tenant.id}",
            headers=_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 401
