"""Integration tests for /api/v1/workspace/allowed-domains.

Mirrors the auth-mocking pattern from tests/api/test_call_flows.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.models.allowed_domain import AllowedDomain
from app.models.tenant import Tenant

_API_KEY = "test-allowed-domains-key"


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
        name=f"DomainWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"domain_ws_{uuid.uuid4().hex[:8]}",
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
class TestCreateAllowedDomain:
    def test_create_returns_201_normalized(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace/allowed-domains",
            json={"domain": "https://App.Example.com:443/"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["domain"] == "https://app.example.com"
        assert "id" in body and "createdAt" in body

    def test_create_normalizes_path_before_storing(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace/allowed-domains",
            json={"domain": "https://example.com/some-path"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["domain"] == "https://example.com"

    def test_rejects_non_https(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/workspace/allowed-domains",
            json={"domain": "http://example.com"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400, resp.text

    def test_max_20_domains_returns_422(self, authed_client, auth_tenant, db):
        for i in range(20):
            db.add(
                AllowedDomain(
                    workspace_id=auth_tenant.id,
                    domain=f"https://site{i}.example.com",
                )
            )
        db.commit()

        resp = authed_client.post(
            "/api/v1/workspace/allowed-domains",
            json={"domain": "https://one-too-many.example.com"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 422, resp.text


@pytest.mark.usefixtures("db")
class TestListAllowedDomains:
    def test_list_returns_workspace_domains_only(self, authed_client, auth_tenant, db):
        other_tenant = Tenant(
            name=f"OtherWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        db.add(AllowedDomain(workspace_id=auth_tenant.id, domain="https://mine.example.com"))
        db.add(AllowedDomain(workspace_id=other_tenant.id, domain="https://theirs.example.com"))
        db.commit()

        resp = authed_client.get(
            "/api/v1/workspace/allowed-domains",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        domains = [d["domain"] for d in resp.json()["data"]]
        assert domains == ["https://mine.example.com"]


@pytest.mark.usefixtures("db")
class TestDeleteAllowedDomain:
    def test_delete_returns_204(self, authed_client, auth_tenant, db):
        domain = AllowedDomain(workspace_id=auth_tenant.id, domain="https://gone.example.com")
        db.add(domain)
        db.commit()
        db.refresh(domain)

        resp = authed_client.delete(
            f"/api/v1/workspace/allowed-domains/{domain.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text
        assert db.query(AllowedDomain).filter(AllowedDomain.id == domain.id).first() is None

    def test_delete_unknown_returns_404(self, authed_client, auth_tenant):
        resp = authed_client.delete(
            f"/api/v1/workspace/allowed-domains/{uuid.uuid4()}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404, resp.text

    def test_delete_other_workspace_domain_returns_404(self, authed_client, auth_tenant, db):
        other_tenant = Tenant(
            name=f"OtherWS2-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_ws2_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        domain = AllowedDomain(workspace_id=other_tenant.id, domain="https://notyours.example.com")
        db.add(domain)
        db.commit()
        db.refresh(domain)

        resp = authed_client.delete(
            f"/api/v1/workspace/allowed-domains/{domain.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404, resp.text
