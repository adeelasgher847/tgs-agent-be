"""Tests for PUT and GET /api/v2/flows/{flow_id}/post-call-actions-settings.

Coverage:
  - Admin/API-key principal can enable both toggles and set recipients
  - email_summary_recipients rejects >10 entries and invalid email strings
  - Unknown extra fields are rejected (extra="forbid")
  - Config-rank (non-admin) principal is forbidden on PUT — mirrors
    tests/api/v2/test_caller_memory_settings.py's admin-gate coverage
  - Unknown flow_id / other-tenant flow both return 404 (tenant isolation) on PUT and GET
  - A successful update fires an audit event with the expected shape
  - The response echoes back the persisted 6-field shape (email + slack settings)
  - GET returns default unconfigured state on fresh flows and handles null DB columns
  - GET round-trips prior PUT updates accurately
  - Read-only rank is sufficient to view post-call actions settings via GET
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
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )

        mini.dependency_overrides[require_admin_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _build_readonly_app(db_override, principal, *, forbidden=False):
    from app.api.deps import get_db, require_readonly_or_api_key
    from app.api.v2.routers.flows import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )

        mini.dependency_overrides[require_readonly_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_readonly_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _build_app_for_real_rank_check(db_override, principal):
    """Companion to `_with_effective_role` below — builds an app where only
    `require_tenant` is overridden (as upstream auth normally would be
    resolved), leaving the REAL rank-checking logic in
    `_require_rank_or_api_key` (app/api/deps/rbac.py) to run unmodified for
    both the admin- and readonly-gated routes on this router.
    """
    from app.api.deps import get_db, require_tenant
    from app.api.v2.routers.flows import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    mini.dependency_overrides[require_tenant] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _with_effective_role(role_name: str):
    """Context manager: pretend `rbac_cache_service.get_effective_role`
    resolves the caller's role to `role_name`, so the real rank-checking
    dependency (`_require_rank_or_api_key`) can be exercised end-to-end
    against a controlled rank without standing up real
    user_tenant_association rows."""
    return patch(
        "app.api.deps.rbac.rbac_cache_service.get_effective_role",
        return_value=role_name,
    )


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
    principal.role = "admin"
    return principal


def _readonly_principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    principal.role = "readonly"
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

    def test_enables_slack_summary_with_channel_override(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": True,
                "slack_channel_id": "C123",
                "slack_channel_name": "sales-calls",
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["slack_summary_enabled"] is True
        assert body["slack_channel_id"] == "C123"
        assert body["slack_channel_name"] == "sales-calls"

        db.refresh(flow)
        assert flow.slack_summary_enabled is True
        assert flow.slack_channel_id == "C123"
        assert flow.slack_channel_name == "sales-calls"

    def test_slack_summary_defaults_to_disabled_with_no_channel_override(
        self, db, workspace, flow
    ):
        """When slack_summary_enabled/channel fields are omitted entirely,
        the flow should round-trip to the documented defaults (disabled, no
        override — falls back to the workspace default channel at send time)."""
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
        assert body["slack_summary_enabled"] is False
        assert body["slack_channel_id"] is None
        assert body["slack_channel_name"] is None

    def test_slack_channel_id_without_name_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": True,
                "slack_channel_id": "C123",
                "slack_channel_name": None,
            },
        )

        assert resp.status_code == 400

    def test_slack_channel_name_without_id_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": True,
                "slack_channel_id": None,
                "slack_channel_name": "sales-calls",
            },
        )

        assert resp.status_code == 400

    def test_disabling_slack_summary_clears_channel_override(self, db, workspace, flow):
        """A previously-set channel override must be clearable independently
        of the email settings on the same request."""
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": True,
                "slack_channel_id": "C123",
                "slack_channel_name": "sales-calls",
            },
        )

        resp = client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": False,
                "slack_channel_id": None,
                "slack_channel_name": None,
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["slack_summary_enabled"] is False
        assert body["slack_channel_id"] is None
        assert body["slack_channel_name"] is None

    def test_two_flows_in_same_tenant_have_independent_slack_settings(
        self, db, workspace, agent, flow
    ):
        """Agent/call-flow-specific config: a second call flow in the same
        tenant must be able to enable Slack with a different channel without
        affecting the first flow."""
        from app.models.call_flow import CallFlow

        other_flow = CallFlow(
            tenant_id=workspace.id,
            agent_id=agent.id,
            name="Second Flow",
            direction="inbound",
        )
        db.add(other_flow)
        db.commit()
        db.refresh(other_flow)

        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": True,
                "slack_channel_id": "C-FLOW-1",
                "slack_channel_name": "flow-one-channel",
            },
        )
        client.put(
            f"/flows/{other_flow.id}/post-call-actions-settings",
            json={
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": False,
            },
        )

        db.refresh(flow)
        db.refresh(other_flow)
        assert flow.slack_summary_enabled is True
        assert flow.slack_channel_id == "C-FLOW-1"
        assert other_flow.slack_summary_enabled is False
        assert other_flow.slack_channel_id is None

    def test_flow_from_other_tenant_cannot_toggle_slack_settings(self, db, flow):
        """Tenant isolation: a principal from another tenant must not be able
        to change this flow's Slack settings."""
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherSlackWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_slack_ws_{uuid.uuid4().hex[:8]}",
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
                "email_summary_enabled": False,
                "email_summary_recipients": [],
                "summary_to_business_owner_enabled": False,
                "slack_summary_enabled": True,
                "slack_channel_id": "C123",
                "slack_channel_name": "sales-calls",
            },
        )

        assert resp.status_code == 404
        db.refresh(flow)
        assert flow.slack_summary_enabled is False
        assert flow.slack_channel_id is None

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
            "slack_summary_enabled": False,
            "slack_channel_id": None,
            "slack_channel_name": None,
        }
        assert kwargs["actor_user_id"] == principal.id


@pytest.mark.usefixtures("db")
class TestGetPostCallActionsSettings:
    def test_get_returns_defaults_for_unconfigured_flow(self, db, workspace, flow):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-actions-settings")

        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "email_summary_enabled": False,
            "email_summary_recipients": [],
            "summary_to_business_owner_enabled": False,
            "slack_summary_enabled": False,
            "slack_channel_id": None,
            "slack_channel_name": None,
        }

    def test_get_round_trips_a_prior_put(self, db, workspace, flow):
        admin_client = _build_app(db, _admin_principal(workspace.id))
        put_payload = {
            "email_summary_enabled": True,
            "email_summary_recipients": ["ops@example.com", "alerts@example.com"],
            "summary_to_business_owner_enabled": True,
            "slack_summary_enabled": True,
            "slack_channel_id": "C12345",
            "slack_channel_name": "ops-channel",
        }
        put_resp = admin_client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json=put_payload,
        )
        assert put_resp.status_code == 200, put_resp.text
        put_body = put_resp.json()

        readonly_client = _build_readonly_app(db, _readonly_principal(workspace.id))
        get_resp = readonly_client.get(f"/flows/{flow.id}/post-call-actions-settings")

        assert get_resp.status_code == 200, get_resp.text
        get_body = get_resp.json()
        assert get_body == put_body
        assert get_body["email_summary_enabled"] is True
        assert get_body["email_summary_recipients"] == ["ops@example.com", "alerts@example.com"]
        assert get_body["summary_to_business_owner_enabled"] is True
        assert get_body["slack_summary_enabled"] is True
        assert get_body["slack_channel_id"] == "C12345"
        assert get_body["slack_channel_name"] == "ops-channel"

    def test_get_flow_from_other_tenant_returns_404(self, db, flow):
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        principal = _readonly_principal(other_tenant.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-actions-settings")
        assert resp.status_code == 404

    def test_get_unknown_flow_returns_404(self, db, workspace):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{uuid.uuid4()}/post-call-actions-settings")
        assert resp.status_code == 404

    def test_readonly_principal_can_get(self, db, workspace, flow):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-actions-settings")
        assert resp.status_code == 200, resp.text

    def test_readonly_rank_can_access_get(self, db, workspace, flow):
        """Readonly is the floor for this endpoint — a genuinely readonly-rank
        principal (real rank-check dependency, not just a dependency-swap)
        must still succeed."""
        principal = _admin_principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        with _with_effective_role("read_only"):
            resp = client.get(f"/flows/{flow.id}/post-call-actions-settings")

        assert resp.status_code == 200, resp.text

    def test_forbidden_principal_cannot_get(self, db, workspace, flow):
        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal, forbidden=True)

        resp = client.get(f"/flows/{flow.id}/post-call-actions-settings")
        assert resp.status_code == 403

    def test_get_handles_null_database_columns(self, db, workspace, agent):
        from app.models.call_flow import CallFlow

        flow = CallFlow(
            tenant_id=workspace.id,
            agent_id=agent.id,
            name="Null Columns Flow",
            direction="inbound",
            email_summary_enabled=None,
            email_summary_recipients=None,
            summary_to_business_owner_enabled=None,
            slack_summary_enabled=None,
            slack_channel_id=None,
            slack_channel_name=None,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)

        principal = _readonly_principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/post-call-actions-settings")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "email_summary_enabled": False,
            "email_summary_recipients": [],
            "summary_to_business_owner_enabled": False,
            "slack_summary_enabled": False,
            "slack_channel_id": None,
            "slack_channel_name": None,
        }

    def test_get_round_trips_slack_enabled_without_channel_override(
        self, db, workspace, flow
    ):
        admin_client = _build_app(db, _admin_principal(workspace.id))
        put_payload = {
            "email_summary_enabled": False,
            "email_summary_recipients": [],
            "summary_to_business_owner_enabled": False,
            "slack_summary_enabled": True,
            "slack_channel_id": None,
            "slack_channel_name": None,
        }
        put_resp = admin_client.put(
            f"/flows/{flow.id}/post-call-actions-settings",
            json=put_payload,
        )
        assert put_resp.status_code == 200, put_resp.text

        readonly_client = _build_readonly_app(db, _readonly_principal(workspace.id))
        get_resp = readonly_client.get(f"/flows/{flow.id}/post-call-actions-settings")

        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json() == {
            "email_summary_enabled": False,
            "email_summary_recipients": [],
            "summary_to_business_owner_enabled": False,
            "slack_summary_enabled": True,
            "slack_channel_id": None,
            "slack_channel_name": None,
        }

