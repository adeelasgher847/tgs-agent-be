"""Tests for PUT /api/v2/flows/{flow_id}/post-call-actions-settings.

Coverage:
  - Admin/API-key principal can enable both toggles and set recipients
  - email_summary_recipients rejects >10 entries and invalid email strings
  - Unknown extra fields are rejected (extra="forbid")
  - Config-rank (non-admin) principal is forbidden — mirrors
    tests/api/v2/test_caller_memory_settings.py's admin-gate coverage
  - Unknown flow_id / other-tenant flow both return 404 (tenant isolation)
  - A successful update fires an audit event with the expected shape
  - The response echoes back the persisted three-field shape
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal, *, forbidden=False):
    from app.api.deps import get_db, require_admin_or_api_key
    from app.api.v2.routers.flows import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

        mini.dependency_overrides[require_admin_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"PostCallWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"post_call_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def agent(db, workspace):
    from app.models.agent import Agent

    a = Agent(
        tenant_id=workspace.id,
        name="Post Call Actions Test Agent",
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
def flow(db, workspace, agent):
    from app.models.call_flow import CallFlow

    f = CallFlow(
        tenant_id=workspace.id,
        agent_id=agent.id,
        name="Post Call Actions Test Flow",
        direction="inbound",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _admin_principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    return principal


@pytest.mark.usefixtures("db")
class TestUpdatePostCallActionsSettings:
    def test_admin_can_enable_both_toggles_with_recipients(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": ["a@example.com", "b@example.com"],
                "summary_to_business_owner_enabled": True,
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["email_summary_enabled"] is True
        assert body["email_summary_recipients"] == ["a@example.com", "b@example.com"]
        assert body["summary_to_business_owner_enabled"] is True

        db.refresh(flow)
        assert flow.email_summary_enabled is True
        assert flow.email_summary_recipients == ["a@example.com", "b@example.com"]
        assert flow.summary_to_business_owner_enabled is True

    def test_disabling_both_toggles_persists(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["email_summary_enabled"] is False
        assert body["email_summary_recipients"] == []
        assert body["summary_to_business_owner_enabled"] is False

    def test_more_than_ten_recipients_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        recipients = [f"user{i}@example.com" for i in range(11)]
        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": recipients,
                "summary_to_business_owner_enabled": False,
            },
        )

        assert resp.status_code == 400

    def test_exactly_ten_recipients_accepted(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        recipients = [f"user{i}@example.com" for i in range(10)]
        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": recipients,
                "summary_to_business_owner_enabled": False,
            },
        )

        assert resp.status_code == 200, resp.text
        assert len(resp.json()["email_summary_recipients"]) == 10

    def test_invalid_email_string_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": ["not-an-email"],
                "summary_to_business_owner_enabled": False,
            },
        )

        assert resp.status_code == 400

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "unexpected_field": "nope",
            },
        )

        assert resp.status_code == 400

    def test_non_admin_principal_is_forbidden(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal, forbidden=True)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
            },
        )

        assert resp.status_code == 403

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
            },
        )

        assert resp.status_code == 404

    def test_flow_from_other_tenant_returns_404(self, db, flow):
        """Tenant isolation: a principal from another tenant must not be
        able to update — or even discover the existence of — this flow."""
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        principal = _admin_principal(other_tenant.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": True,
                "email_summary_recipients": ["x@example.com"],
                "summary_to_business_owner_enabled": True,
            },
        )

        assert resp.status_code == 404

        db.refresh(flow)
        assert flow.email_summary_enabled is False
        assert flow.summary_to_business_owner_enabled is False

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_log_audit:
            resp = client.put(
                f"/flows/{flow.id}/post-call-actions-settings",
                json={
                    "email_summary_enabled": True,
                    "email_summary_recipients": ["owner@example.com"],
                    "summary_to_business_owner_enabled": True,
                },
            )

        assert resp.status_code == 200, resp.text
        mock_log_audit.assert_called_once()
        kwargs = mock_log_audit.call_args.kwargs
        assert kwargs["action"] == "post_call_actions_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"] == {
            "email_summary_enabled": True,
            "email_summary_recipients": ["owner@example.com"],
            "summary_to_business_owner_enabled": True,
        }
        assert kwargs["actor_user_id"] == principal.id
