"""Integration tests for tenant-facing demo-link CRUD:

POST/GET   /api/v1/call-flows/{flow_id}/demo-links
PATCH      /api/v1/call-flows/demo-links/{link_id}
DELETE     /api/v1/call-flows/demo-links/{link_id}

Mirrors the auth-mocking pattern from tests/api/test_call_flows.py.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.workspace import Workspace
from app.middleware.api_key_middleware import _attach_workspace_context
from app.core.request_auth import AUTH_METHOD_JWT
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_flow_demo_link import CallFlowDemoLink
from app.models.call_flow_demo_link_visitor_usage import CallFlowDemoLinkVisitorUsage
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association

_API_KEY = "test-demolinks-key"


# ─────────────────────────────────────────────────────────────── helpers ──


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


def _make_jwt_user(db, tenant_id: uuid.UUID, role_name: str) -> User:
    role = db.query(Role).filter(Role.name == role_name).first()
    if role is None:
        role = Role(name=role_name, description=role_name)
        db.add(role)
        db.commit()
        db.refresh(role)

    user = User(
        email=f"{role_name}-{uuid.uuid4().hex[:8]}@example.com",
        first_name=role_name.title(),
        last_name="User",
        hashed_password="",
        current_tenant_id=tenant_id,
    )
    db.add(user)
    db.flush()
    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id, tenant_id=tenant_id, role_id=role.id, is_creator=False
        )
    )
    db.commit()
    db.refresh(user)
    return user


@contextmanager
def _jwt_auth_ctx(workspace: Workspace, user_id: uuid.UUID):
    async def _jwt_auth(request):
        _attach_workspace_context(
            request, workspace=workspace, auth_method=AUTH_METHOD_JWT, user_id=user_id,
        )
        return True

    with patch(
        "app.middleware.api_key_middleware._try_jwt_auth", side_effect=_jwt_auth
    ):
        yield


# ─────────────────────────────────────────────────────────────── fixtures ──


@pytest.fixture
def auth_tenant(db) -> Tenant:
    t = Tenant(
        name=f"DemoLinkWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"demo_link_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def other_tenant(db) -> Tenant:
    t = Tenant(
        name=f"OtherWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def test_agent(db, auth_tenant: Tenant) -> Agent:
    a = Agent(
        tenant_id=auth_tenant.id,
        name="Demo Link Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@pytest.fixture
def test_flow(db, auth_tenant: Tenant, test_agent: Agent) -> CallFlow:
    flow = CallFlow(
        tenant_id=auth_tenant.id,
        agent_id=test_agent.id,
        name="Demo Link Flow",
        direction="inbound",
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


@pytest.fixture
def other_flow(db, other_tenant: Tenant) -> CallFlow:
    other_agent = Agent(
        tenant_id=other_tenant.id,
        name="Other Tenant Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
    )
    db.add(other_agent)
    db.commit()
    db.refresh(other_agent)

    flow = CallFlow(
        tenant_id=other_tenant.id,
        agent_id=other_agent.id,
        name="Other Tenant Flow",
        direction="inbound",
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


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


# ──────────────────────────────────────────────────────────────── tests ──


@pytest.mark.usefixtures("db")
class TestCreateDemoLink:
    def test_create_happy_path_returns_token_and_share_url(
        self, authed_client, auth_tenant, test_flow
    ):
        resp = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "My Demo Link"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "My Demo Link"
        assert body["callFlowId"] == str(test_flow.id)
        assert body["token"]
        # shareUrl is derived server-side from settings.FRONTEND_URL + token
        assert "shareUrl" in body
        assert body["isActive"] is True
        assert float(body["totalMinutesUsed"]) == 0

    def test_client_supplied_token_is_rejected(
        self, authed_client, auth_tenant, test_flow
    ):
        """CallFlowDemoLinkCreate uses extra='forbid' — a 'token' field in the
        body must be rejected outright, not silently ignored."""
        resp = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "Hack", "token": "attacker-supplied-token"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400, resp.text

    def test_created_by_user_id_set_for_jwt_user(
        self, authed_client, auth_tenant, test_flow, db
    ):
        admin = _make_jwt_user(db, auth_tenant.id, "admin")
        workspace = Workspace.from_tenant(auth_tenant)

        with _jwt_auth_ctx(workspace, admin.id):
            resp = authed_client.post(
                f"/api/v1/call-flows/{test_flow.id}/demo-links",
                json={"name": "JWT Created Link"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["createdByUserId"] == str(admin.id)

    def test_created_by_user_id_null_for_api_key_principal(
        self, authed_client, auth_tenant, test_flow
    ):
        resp = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "API Key Created Link"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["createdByUserId"] is None

    def test_create_cross_tenant_flow_returns_404(
        self, authed_client, auth_tenant, other_flow
    ):
        resp = authed_client.post(
            f"/api/v1/call-flows/{other_flow.id}/demo-links",
            json={"name": "Cross Tenant"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404, resp.text


@pytest.mark.usefixtures("db")
class TestListDemoLinks:
    def test_list_is_tenant_and_flow_scoped(
        self, authed_client, auth_tenant, test_flow, test_agent, db
    ):
        authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "Link A"},
            headers=_headers(auth_tenant),
        )
        authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "Link B"},
            headers=_headers(auth_tenant),
        )

        # A second flow (same tenant) with its own demo link must not appear
        other_flow_same_tenant = CallFlow(
            tenant_id=auth_tenant.id,
            agent_id=test_agent.id,
            name="Second Flow",
            direction="inbound",
        )
        db.add(other_flow_same_tenant)
        db.commit()
        db.refresh(other_flow_same_tenant)
        authed_client.post(
            f"/api/v1/call-flows/{other_flow_same_tenant.id}/demo-links",
            json={"name": "Unrelated Link"},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.get(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        names = {item["name"] for item in resp.json()["data"]}
        assert names == {"Link A", "Link B"}

    def test_list_cross_tenant_flow_returns_404(
        self, authed_client, auth_tenant, other_flow
    ):
        resp = authed_client.get(
            f"/api/v1/call-flows/{other_flow.id}/demo-links",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404, resp.text


@pytest.mark.usefixtures("db")
class TestUpdateDemoLink:
    def _create_link(self, authed_client, auth_tenant, flow, **overrides) -> dict:
        body = {"name": "Original Name"}
        body.update(overrides)
        resp = authed_client.post(
            f"/api/v1/call-flows/{flow.id}/demo-links",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        return resp.json()

    def test_patch_name_only(self, authed_client, auth_tenant, test_flow):
        link = self._create_link(authed_client, auth_tenant, test_flow)
        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{link['id']}",
            json={"name": "New Name"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "New Name"
        assert body["isActive"] is True

    def test_patch_is_active_toggle(self, authed_client, auth_tenant, test_flow):
        link = self._create_link(authed_client, auth_tenant, test_flow)
        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{link['id']}",
            json={"isActive": False},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["isActive"] is False

        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{link['id']}",
            json={"isActive": True},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["isActive"] is True

    def test_patch_expires_at(self, authed_client, auth_tenant, test_flow):
        link = self._create_link(authed_client, auth_tenant, test_flow)
        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{link['id']}",
            json={"expiresAt": "2099-01-01T00:00:00Z"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["expiresAt"].startswith("2099-01-01")

    def test_patch_per_user_limit_minutes(self, authed_client, auth_tenant, test_flow):
        link = self._create_link(authed_client, auth_tenant, test_flow)
        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{link['id']}",
            json={"perUserLimitMinutes": 5},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert float(resp.json()["perUserLimitMinutes"]) == 5.0

    def test_patch_total_budget_minutes(self, authed_client, auth_tenant, test_flow):
        link = self._create_link(authed_client, auth_tenant, test_flow)
        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{link['id']}",
            json={"totalBudgetMinutes": 100},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert float(resp.json()["totalBudgetMinutes"]) == 100.0

    def test_patch_cross_tenant_link_returns_404(
        self, authed_client, auth_tenant, other_flow, db
    ):
        other_link = CallFlowDemoLink(
            tenant_id=other_flow.tenant_id,
            call_flow_id=other_flow.id,
            token=uuid.uuid4().hex,
        )
        db.add(other_link)
        db.commit()
        db.refresh(other_link)

        resp = authed_client.patch(
            f"/api/v1/call-flows/demo-links/{other_link.id}",
            json={"name": "Hijack"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404, resp.text


@pytest.mark.usefixtures("db")
class TestDeleteDemoLink:
    def test_delete_hard_deletes_row(self, authed_client, auth_tenant, test_flow, db):
        created = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "To Delete"},
            headers=_headers(auth_tenant),
        ).json()
        link_id = uuid.UUID(created["id"])

        resp = authed_client.delete(
            f"/api/v1/call-flows/demo-links/{link_id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text

        row = db.query(CallFlowDemoLink).filter(CallFlowDemoLink.id == link_id).first()
        assert row is None

    def test_delete_cascades_visitor_usage(self, authed_client, auth_tenant, test_flow, db):
        created = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "With Usage"},
            headers=_headers(auth_tenant),
        ).json()
        link_id = uuid.UUID(created["id"])

        usage = CallFlowDemoLinkVisitorUsage(
            demo_link_id=link_id,
            visitor_id="visitor-abc",
        )
        db.add(usage)
        db.commit()
        db.refresh(usage)
        usage_id = usage.id

        resp = authed_client.delete(
            f"/api/v1/call-flows/demo-links/{link_id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text

        remaining = (
            db.query(CallFlowDemoLinkVisitorUsage)
            .filter(CallFlowDemoLinkVisitorUsage.id == usage_id)
            .first()
        )
        assert remaining is None

    def test_delete_cross_tenant_link_returns_404(
        self, authed_client, auth_tenant, other_flow, db
    ):
        other_link = CallFlowDemoLink(
            tenant_id=other_flow.tenant_id,
            call_flow_id=other_flow.id,
            token=uuid.uuid4().hex,
        )
        db.add(other_link)
        db.commit()
        db.refresh(other_link)

        resp = authed_client.delete(
            f"/api/v1/call-flows/demo-links/{other_link.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404, resp.text


@pytest.mark.usefixtures("db")
class TestDemoLinkRbac:
    """require_config_or_api_key gates all four endpoints — read_only (below
    config tier) must be rejected on every one; API-key principals bypass
    role checks entirely (covered by the happy-path tests above)."""

    def test_read_only_forbidden_on_create(self, authed_client, auth_tenant, test_flow, db):
        reader = _make_jwt_user(db, auth_tenant.id, "read_only")
        workspace = Workspace.from_tenant(auth_tenant)

        with _jwt_auth_ctx(workspace, reader.id):
            resp = authed_client.post(
                f"/api/v1/call-flows/{test_flow.id}/demo-links",
                json={"name": "Should Fail"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 403, resp.text

    def test_read_only_forbidden_on_list(self, authed_client, auth_tenant, test_flow, db):
        reader = _make_jwt_user(db, auth_tenant.id, "read_only")
        workspace = Workspace.from_tenant(auth_tenant)

        with _jwt_auth_ctx(workspace, reader.id):
            resp = authed_client.get(
                f"/api/v1/call-flows/{test_flow.id}/demo-links",
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 403, resp.text

    def test_read_only_forbidden_on_patch(self, authed_client, auth_tenant, test_flow, db):
        link = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "Target"},
            headers=_headers(auth_tenant),
        ).json()

        reader = _make_jwt_user(db, auth_tenant.id, "read_only")
        workspace = Workspace.from_tenant(auth_tenant)

        with _jwt_auth_ctx(workspace, reader.id):
            resp = authed_client.patch(
                f"/api/v1/call-flows/demo-links/{link['id']}",
                json={"name": "Should Fail"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 403, resp.text

    def test_read_only_forbidden_on_delete(self, authed_client, auth_tenant, test_flow, db):
        link = authed_client.post(
            f"/api/v1/call-flows/{test_flow.id}/demo-links",
            json={"name": "Target"},
            headers=_headers(auth_tenant),
        ).json()

        reader = _make_jwt_user(db, auth_tenant.id, "read_only")
        workspace = Workspace.from_tenant(auth_tenant)

        with _jwt_auth_ctx(workspace, reader.id):
            resp = authed_client.delete(
                f"/api/v1/call-flows/demo-links/{link['id']}",
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 403, resp.text
