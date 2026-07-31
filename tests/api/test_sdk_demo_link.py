"""Integration tests for the unauthenticated POST /api/v1/sdk/demo/{token}/call-token.

No x-api-key / x-workspace-id / Authorization headers are ever sent here.
Unlike public-call-token, this endpoint is deliberately origin-unrestricted —
access is controlled purely by possession of the opaque demo-link token plus
its active/expiry/usage-budget fields. See app/routers/sdk.py and
app/models/call_flow_demo_link.py for the security rationale.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_flow_demo_link import CallFlowDemoLink
from app.models.call_flow_demo_link_visitor_usage import CallFlowDemoLinkVisitorUsage
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"DemoSdkWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"demo_sdk_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def creator_user(db) -> User:
    # Every real tenant/agent has an owning user; CallSession.user_id is a
    # NOT NULL FK, so a demo-link call session needs one to attribute to.
    u = User(
        first_name="Demo",
        last_name="Creator",
        email=f"demo-creator-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="not-a-real-hash",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def agent(db, tenant: Tenant, creator_user: User) -> Agent:
    a = Agent(
        tenant_id=tenant.id,
        name="Demo SDK Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
        created_by=creator_user.id,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@pytest.fixture
def flow(db, tenant: Tenant, agent: Agent) -> CallFlow:
    f = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Demo SDK Flow",
        direction="inbound",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _make_link(db, tenant: Tenant, flow: CallFlow, **overrides) -> CallFlowDemoLink:
    kwargs = dict(
        tenant_id=tenant.id,
        call_flow_id=flow.id,
        name="Demo Link",
        token=uuid.uuid4().hex,
        is_active=True,
    )
    kwargs.update(overrides)
    link = CallFlowDemoLink(**kwargs)
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


@pytest.fixture
def demo_link(db, tenant: Tenant, flow: CallFlow) -> CallFlowDemoLink:
    return _make_link(db, tenant, flow)


@pytest.fixture
def mock_livekit():
    mock = MagicMock()
    mock.generate_caller_token = MagicMock(return_value="signed.demo.jwt")
    with patch("app.services.livekit_service.livekit_service", mock):
        yield mock


def _body(visitor_id: str = "visitor-1") -> dict:
    return {"visitor_id": visitor_id}


@pytest.mark.usefixtures("db")
class TestDemoCallToken:
    def test_happy_path_no_limits(self, client: TestClient, db, tenant, flow, demo_link, mock_livekit):
        resp = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body("visitor-happy"),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["livekit_token"] == "signed.demo.jwt"
        assert body["room_name"].startswith("room_")
        assert body["flow_id"] == str(flow.id)
        assert body["agent_id"] == str(flow.agent_id)
        assert "call_session_id" in body
        mock_livekit.generate_caller_token.assert_called_once()

        # A CallSession row was created and tagged with demo_link metadata
        session = (
            db.query(CallSession)
            .filter(CallSession.id == uuid.UUID(body["call_session_id"]))
            .first()
        )
        assert session is not None
        assert session.call_type == "web"
        assert session.call_metadata["demo_link"]["demo_link_id"] == str(demo_link.id)
        assert session.call_metadata["demo_link"]["visitor_id"] == "visitor-happy"

    def test_unknown_token_returns_404(self, client: TestClient, mock_livekit):
        resp = client.post(
            "/api/v1/sdk/demo/does-not-exist/call-token",
            json=_body(),
        )
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "demo_link_not_found"

    def test_inactive_link_returns_403(self, client: TestClient, db, tenant, flow, mock_livekit):
        link = _make_link(db, tenant, flow, is_active=False)
        resp = client.post(
            f"/api/v1/sdk/demo/{link.token}/call-token",
            json=_body(),
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "demo_link_inactive"

    def test_expired_link_returns_410(self, client: TestClient, db, tenant, flow, mock_livekit):
        link = _make_link(
            db, tenant, flow,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        resp = client.post(
            f"/api/v1/sdk/demo/{link.token}/call-token",
            json=_body(),
        )
        assert resp.status_code == 410, resp.text
        assert resp.json()["error"]["code"] == "demo_link_expired"

    def test_deleted_parent_flow_returns_403(self, client: TestClient, db, tenant, flow, demo_link, mock_livekit):
        flow.is_deleted = True
        db.add(flow)
        db.commit()

        resp = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body(),
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "call_flow_unavailable"

    def test_inactive_parent_flow_returns_403(self, client: TestClient, db, tenant, flow, demo_link, mock_livekit):
        flow.status = "inactive"
        db.add(flow)
        db.commit()

        resp = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body(),
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "call_flow_unavailable"

    def test_deleted_agent_returns_403(self, client: TestClient, db, tenant, agent, flow, demo_link, mock_livekit):
        agent.is_deleted = True
        db.add(agent)
        db.commit()

        resp = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body(),
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "agent_unavailable"

    def test_total_budget_exceeded_returns_403(self, client: TestClient, db, tenant, flow, mock_livekit):
        link = _make_link(
            db, tenant, flow,
            total_budget_minutes=Decimal("10"),
            total_minutes_used=Decimal("10"),
        )
        resp = client.post(
            f"/api/v1/sdk/demo/{link.token}/call-token",
            json=_body(),
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "demo_link_budget_exceeded"

    def test_per_visitor_limit_exceeded_returns_403(
        self, client: TestClient, db, tenant, flow, mock_livekit
    ):
        link = _make_link(db, tenant, flow, per_user_limit_minutes=Decimal("5"))
        usage = CallFlowDemoLinkVisitorUsage(
            demo_link_id=link.id,
            visitor_id="visitor-capped",
            minutes_used=Decimal("5"),
        )
        db.add(usage)
        db.commit()

        resp = client.post(
            f"/api/v1/sdk/demo/{link.token}/call-token",
            json=_body("visitor-capped"),
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "demo_link_visitor_limit_exceeded"

    def test_first_time_visitor_usage_row_created(
        self, client: TestClient, db, tenant, flow, demo_link, mock_livekit
    ):
        assert (
            db.query(CallFlowDemoLinkVisitorUsage)
            .filter(
                CallFlowDemoLinkVisitorUsage.demo_link_id == demo_link.id,
                CallFlowDemoLinkVisitorUsage.visitor_id == "brand-new-visitor",
            )
            .first()
            is None
        )

        resp = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body("brand-new-visitor"),
        )
        assert resp.status_code == 200, resp.text

        usage = (
            db.query(CallFlowDemoLinkVisitorUsage)
            .filter(
                CallFlowDemoLinkVisitorUsage.demo_link_id == demo_link.id,
                CallFlowDemoLinkVisitorUsage.visitor_id == "brand-new-visitor",
            )
            .first()
        )
        assert usage is not None
        assert float(usage.minutes_used) == 0

    def test_origin_is_not_enforced(self, client: TestClient, demo_link, mock_livekit):
        """Unlike public-call-token, no AllowedDomain/Origin check gates this
        endpoint — any Origin (or none) must be allowed through."""
        resp_no_origin = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body("visitor-no-origin"),
        )
        assert resp_no_origin.status_code == 200, resp_no_origin.text

        resp_evil_origin = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body("visitor-evil-origin"),
            headers={"Origin": "https://totally-unrelated-and-unwhitelisted.example.com"},
        )
        assert resp_evil_origin.status_code == 200, resp_evil_origin.text

    def test_reachable_without_any_auth_headers(self, client: TestClient, demo_link, mock_livekit):
        """No x-api-key, x-workspace-id, or Authorization header is sent —
        confirms the auth-bypass middleware change covers this path."""
        resp = client.post(
            f"/api/v1/sdk/demo/{demo_link.token}/call-token",
            json=_body("visitor-no-auth"),
        )
        assert resp.status_code == 200, resp.text
        assert "livekit_token" in resp.json()
