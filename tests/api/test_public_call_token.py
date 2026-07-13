"""Integration tests for the unauthenticated POST /api/v1/sdk/public-call-token.

No x-api-key / x-workspace-id / Authorization headers are ever sent here —
that's the point of this endpoint. Security is exercised via flow.public_access
and the allowed_domains Origin check instead.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.agent import Agent
from app.models.allowed_domain import AllowedDomain
from app.models.call_flow import CallFlow
from app.models.tenant import Tenant


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"SdkWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"sdk_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def agent(db, tenant: Tenant) -> Agent:
    a = Agent(
        tenant_id=tenant.id,
        name="SDK Test Agent",
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
def public_flow(db, tenant: Tenant, agent: Agent) -> CallFlow:
    flow = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Public Flow",
        direction="inbound",
        public_access=True,
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


@pytest.fixture
def private_flow(db, tenant: Tenant, agent: Agent) -> CallFlow:
    flow = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Private Flow",
        direction="inbound",
        public_access=False,
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


@pytest.fixture
def mock_livekit():
    mock = MagicMock()
    mock.generate_caller_token = MagicMock(return_value="signed.jwt.token")
    with patch("app.services.livekit_service.livekit_service", mock):
        yield mock


def _body(flow: CallFlow) -> dict:
    return {"flow_id": str(flow.id), "agent_id": str(flow.agent_id)}


@pytest.mark.usefixtures("db")
class TestPublicCallToken:
    def test_allowed_origin_returns_token(
        self, client: TestClient, db, tenant, public_flow, mock_livekit
    ):
        db.add(AllowedDomain(workspace_id=tenant.id, domain="https://embed.example.com"))
        db.commit()

        resp = client.post(
            "/api/v1/sdk/public-call-token",
            json=_body(public_flow),
            headers={"Origin": "https://embed.example.com"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["livekit_token"] == "signed.jwt.token"
        assert body["flow_id"] == str(public_flow.id)
        assert body["room_name"].startswith("room_")
        assert "expires_at" in body
        mock_livekit.generate_caller_token.assert_called_once()

    def test_disallowed_origin_returns_403(self, client: TestClient, db, tenant, public_flow, mock_livekit):
        db.add(AllowedDomain(workspace_id=tenant.id, domain="https://embed.example.com"))
        db.commit()

        resp = client.post(
            "/api/v1/sdk/public-call-token",
            json=_body(public_flow),
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "domain_not_allowed"

    def test_flow_without_public_access_returns_403(
        self, client: TestClient, db, tenant, private_flow, mock_livekit
    ):
        db.add(AllowedDomain(workspace_id=tenant.id, domain="https://embed.example.com"))
        db.commit()

        resp = client.post(
            "/api/v1/sdk/public-call-token",
            json=_body(private_flow),
            headers={"Origin": "https://embed.example.com"},
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "public_access_disabled"

    def test_missing_origin_header_returns_403(self, client: TestClient, db, tenant, public_flow, mock_livekit):
        db.add(AllowedDomain(workspace_id=tenant.id, domain="https://embed.example.com"))
        db.commit()

        resp = client.post("/api/v1/sdk/public-call-token", json=_body(public_flow))
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "domain_not_allowed"

    def test_unknown_flow_returns_404(self, client: TestClient, mock_livekit):
        resp = client.post(
            "/api/v1/sdk/public-call-token",
            json={"flow_id": str(uuid.uuid4()), "agent_id": str(uuid.uuid4())},
            headers={"Origin": "https://embed.example.com"},
        )
        assert resp.status_code == 404, resp.text

    def test_localhost_allowed_in_development_without_whitelist(
        self, client: TestClient, public_flow, mock_livekit
    ):
        """No allowed_domains row needed — localhost:* always passes in development."""
        resp = client.post(
            "/api/v1/sdk/public-call-token",
            json=_body(public_flow),
            headers={"Origin": "http://localhost:5173"},
        )
        assert resp.status_code == 200, resp.text

    def test_normalized_origin_matches_stored_domain(
        self, client: TestClient, db, tenant, public_flow, mock_livekit
    ):
        """Stored as https://app.example.com; browser sends with trailing slash
        stripped and default port — both normalize to the same value."""
        db.add(AllowedDomain(workspace_id=tenant.id, domain="https://app.example.com"))
        db.commit()

        resp = client.post(
            "/api/v1/sdk/public-call-token",
            json=_body(public_flow),
            headers={"Origin": "https://App.Example.com:443"},
        )
        assert resp.status_code == 200, resp.text

    def test_does_not_log_token(self, client: TestClient, db, tenant, public_flow, mock_livekit, caplog):
        db.add(AllowedDomain(workspace_id=tenant.id, domain="https://embed.example.com"))
        db.commit()

        with caplog.at_level("INFO"):
            resp = client.post(
                "/api/v1/sdk/public-call-token",
                json=_body(public_flow),
                headers={"Origin": "https://embed.example.com"},
            )
        assert resp.status_code == 200
        assert "signed.jwt.token" not in caplog.text
