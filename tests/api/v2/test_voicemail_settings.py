"""Tests for PUT and GET /api/v2/flows/{flow_id}/voicemail-settings.

Coverage:
  - Admin/API-key principal can enable voicemail detection and configure actions
  - Action options: hang_up, leave_message (with voicemail_message), continue
  - Advanced detection toggle (Twilio AMD) and detection timeout (1–30s)
  - Validation: invalid action rejected (400), timeout out-of-range (<1 or >30) rejected (400)
  - Validation: voicemail_message > 500 chars rejected (400), whitespace trimmed
  - Extra fields rejected (extra="forbid")
  - Non-admin (config, read_only) principals forbidden on PUT (403)
  - Tenant isolation: unknown flow_id or other tenant flow returns 404 on PUT and GET
  - Audit event logged on successful PUT (action="voicemail_settings.updated")
  - GET returns default unconfigured state on fresh flows and handles null columns
  - GET round-trips prior PUT updates accurately
  - Read-only rank is sufficient to view settings via GET
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
    from app.api.deps import get_db, require_tenant
    from app.api.v2.routers.flows import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    mini.dependency_overrides[require_tenant] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _with_effective_role(role_name: str):
    return patch(
        "app.api.deps.rbac.rbac_cache_service.get_effective_role",
        return_value=role_name,
    )


def _principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    return principal


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"VMSettingsWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"vm_settings_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def other_workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"OtherVMSettingsWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_vm_settings_ws_{uuid.uuid4().hex[:8]}",
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
        name="VM Settings Test Agent",
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
        name="VM Settings Test Flow",
        direction="inbound",
        status="active",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestUpdateVoicemailSettings:
    def test_admin_can_enable_voicemail_detection(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "hang_up",
                "voicemail_message": None,
                "voicemail_advanced_detection_enabled": False,
                "voicemail_detection_timeout": 5,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["voicemail_detection_enabled"] is True
        assert body["voicemail_action"] == "hang_up"
        assert body["voicemail_message"] is None
        assert body["voicemail_advanced_detection_enabled"] is False
        assert body["voicemail_detection_timeout"] == 5

        db.refresh(flow)
        assert flow.voicemail_detection_enabled is True
        assert flow.voicemail_action == "hang_up"

    def test_admin_can_configure_leave_message_action(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "leave_message",
                "voicemail_message": "Hello, please call us back at 555-1234.",
                "voicemail_advanced_detection_enabled": True,
                "voicemail_detection_timeout": 10,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["voicemail_detection_enabled"] is True
        assert body["voicemail_action"] == "leave_message"
        assert body["voicemail_message"] == "Hello, please call us back at 555-1234."
        assert body["voicemail_advanced_detection_enabled"] is True
        assert body["voicemail_detection_timeout"] == 10

        db.refresh(flow)
        assert flow.voicemail_action == "leave_message"
        assert flow.voicemail_message == "Hello, please call us back at 555-1234."
        assert flow.voicemail_advanced_detection_enabled is True
        assert flow.voicemail_detection_timeout == 10

    def test_admin_can_configure_continue_action(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "continue",
                "voicemail_message": None,
                "voicemail_advanced_detection_enabled": False,
                "voicemail_detection_timeout": 3,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["voicemail_action"] == "continue"

        db.refresh(flow)
        assert flow.voicemail_action == "continue"

    @pytest.mark.parametrize("timeout", [0, 31, -1, 100])
    def test_timeout_out_of_range_returns_422(self, db, workspace, flow, timeout):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_detection_timeout": timeout,
            },
        )
        assert resp.status_code in (400, 422)

    @pytest.mark.parametrize("timeout", [1, 30, 15])
    def test_timeout_boundary_values_accepted(self, db, workspace, flow, timeout):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_detection_timeout": timeout,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["voicemail_detection_timeout"] == timeout

    def test_invalid_action_returns_422(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "invalid_action_value",
            },
        )
        assert resp.status_code in (400, 422)

    def test_message_over_max_length_returns_422(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "leave_message",
                "voicemail_message": "x" * 501,
            },
        )
        assert resp.status_code in (400, 422)

    def test_message_whitespace_is_trimmed(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "leave_message",
                "voicemail_message": "   Trimmed message   ",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["voicemail_message"] == "Trimmed message"

    def test_empty_string_message_becomes_none(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "hang_up",
                "voicemail_message": "    ",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["voicemail_message"] is None

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "unknown_extra_field": "disallowed",
            },
        )
        assert resp.status_code in (400, 422)

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/voicemail-settings",
            json={"voicemail_detection_enabled": True},
        )
        assert resp.status_code == 404

    def test_flow_from_other_tenant_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_app(db, foreign_principal)

        resp = client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={"voicemail_detection_enabled": True},
        )
        assert resp.status_code == 404

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_audit:
            resp = client.put(
                f"/flows/{flow.id}/voicemail-settings",
                json={
                    "voicemail_detection_enabled": True,
                    "voicemail_action": "leave_message",
                    "voicemail_message": "Audit test voicemail",
                    "voicemail_advanced_detection_enabled": True,
                    "voicemail_detection_timeout": 8,
                },
            )
            assert resp.status_code == 200

        mock_audit.assert_called_once()
        _, kwargs = mock_audit.call_args
        assert kwargs["action"] == "voicemail_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"] == {
            "voicemail_detection_enabled": True,
            "voicemail_action": "leave_message",
            "voicemail_message": "Audit test voicemail",
            "voicemail_advanced_detection_enabled": True,
            "voicemail_detection_timeout": 8,
        }

    def test_non_admin_forbidden_on_put(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for non_admin_role in ["config_only", "read_only", "billing_only"]:
            with _with_effective_role(non_admin_role):
                resp = client.put(
                    f"/flows/{flow.id}/voicemail-settings",
                    json={"voicemail_detection_enabled": True},
                )
                assert resp.status_code == 403, (
                    f"Expected 403 for role '{non_admin_role}', got {resp.status_code}"
                )


class TestGetVoicemailSettings:
    def test_fresh_flow_returns_default_settings(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/voicemail-settings")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {
            "voicemail_detection_enabled": False,
            "voicemail_action": "hang_up",
            "voicemail_message": None,
            "voicemail_advanced_detection_enabled": False,
            "voicemail_detection_timeout": 5,
        }

    def test_get_round_trips_prior_put_update(self, db, workspace, flow):
        admin_principal = _principal(workspace.id)
        admin_client = _build_app(db, admin_principal)

        put_resp = admin_client.put(
            f"/flows/{flow.id}/voicemail-settings",
            json={
                "voicemail_detection_enabled": True,
                "voicemail_action": "leave_message",
                "voicemail_message": "Round-trip message",
                "voicemail_advanced_detection_enabled": True,
                "voicemail_detection_timeout": 12,
            },
        )
        assert put_resp.status_code == 200

        readonly_principal = _principal(workspace.id)
        readonly_client = _build_readonly_app(db, readonly_principal)

        get_resp = readonly_client.get(f"/flows/{flow.id}/voicemail-settings")
        assert get_resp.status_code == 200
        assert get_resp.json() == put_resp.json()

    def test_readonly_user_can_access_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for role in ["read_only", "config_only", "manager", "admin", "owner"]:
            with _with_effective_role(role):
                resp = client.get(f"/flows/{flow.id}/voicemail-settings")
                assert resp.status_code == 200, (
                    f"Expected 200 for role '{role}', got {resp.status_code}"
                )

    def test_get_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{uuid.uuid4()}/voicemail-settings")
        assert resp.status_code == 404

    def test_get_flow_from_other_tenant_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_readonly_app(db, foreign_principal)

        resp = client.get(f"/flows/{flow.id}/voicemail-settings")
        assert resp.status_code == 404

    def test_get_handles_none_attributes_on_flow_object(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        mock_flow = MagicMock()
        mock_flow.id = flow.id
        mock_flow.tenant_id = workspace.id
        mock_flow.voicemail_detection_enabled = None
        mock_flow.voicemail_action = None
        mock_flow.voicemail_message = None
        mock_flow.voicemail_advanced_detection_enabled = None
        mock_flow.voicemail_detection_timeout = None

        with patch("app.services.call_flow_service.call_flow_service._get_flow_or_404", return_value=mock_flow):
            resp = client.get(f"/flows/{flow.id}/voicemail-settings")
            assert resp.status_code == 200
            body = resp.json()
            assert body["voicemail_detection_enabled"] is False
            assert body["voicemail_action"] == "hang_up"
            assert body["voicemail_message"] is None
            assert body["voicemail_advanced_detection_enabled"] is False
            assert body["voicemail_detection_timeout"] == 5
