"""Tests for PUT and GET /api/v2/flows/{flow_id}/ivr-dtmf-settings.

Coverage:
  - Admin/API-key principal can update all IVR and DTMF settings
  - Default values on fresh flows
  - Boundary validations (attempts 1-10, delay 0-15s, hold time 15-900s, button delay 0-10s, digits 1-100, exceeded attempts 1-50)
  - Enum validations for ivr_action, ivr_navigation_mode, dtmf_exceeded_action
  - dtmf_end_call_message string handling and max length
  - Extra fields rejected (extra="forbid")
  - Non-admin (config_only, read_only, billing_only) forbidden on PUT (403)
  - Read-only rank is sufficient for GET (200)
  - Tenant isolation: unknown flow_id or other tenant flow returns 404
  - Audit event logged on PUT (action="ivr_dtmf_settings.updated")
  - GET round-trips prior PUT updates
  - GET handles null/none attribute fallback on flow object
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
        name=f"IVRWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"ivr_ws_{uuid.uuid4().hex[:8]}",
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
        name=f"OtherIVRWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_ivr_ws_{uuid.uuid4().hex[:8]}",
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
        name="IVR Test Agent",
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
        name="IVR DTMF Test Flow",
        direction="outbound",
        status="active",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestUpdateIVRDTMFSettings:
    def test_admin_can_update_all_ivr_dtmf_settings(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "ivr_enabled": True,
            "ivr_action": "dial_through",
            "ivr_navigation_mode": "auto_detect",
            "ivr_max_attempts": 5,
            "ivr_keypress_delay": 10,
            "ivr_priority_list": ["support", "billing"],
            "ivr_wait_on_hold": True,
            "ivr_max_hold_time": 300,
            "dtmf_enabled": True,
            "dtmf_button_press_delay": 3,
            "dtmf_allow_caller_interruption": True,
            "dtmf_max_digits": 20,
            "dtmf_allowed_exceeded_attempts": 5,
            "dtmf_exceeded_action": "end_call",
            "dtmf_end_call_message": "Goodbye - too many inputs.",
        }

        resp = client.put(f"/flows/{flow.id}/ivr-dtmf-settings", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for k, v in payload.items():
            assert body[k] == v

        db.refresh(flow)
        assert flow.ivr_enabled is True
        assert flow.ivr_action == "dial_through"
        assert flow.ivr_navigation_mode == "auto_detect"
        assert flow.ivr_max_attempts == 5
        assert flow.ivr_keypress_delay == 10
        assert flow.ivr_priority_list == ["support", "billing"]
        assert flow.ivr_wait_on_hold is True
        assert flow.ivr_max_hold_time == 300
        assert flow.dtmf_enabled is True
        assert flow.dtmf_button_press_delay == 3
        assert flow.dtmf_allow_caller_interruption is True
        assert flow.dtmf_max_digits == 20
        assert flow.dtmf_allowed_exceeded_attempts == 5
        assert flow.dtmf_exceeded_action == "end_call"
        assert flow.dtmf_end_call_message == "Goodbye - too many inputs."

    @pytest.mark.parametrize(
        "field,value",
        [
            ("ivr_max_attempts", 0),
            ("ivr_max_attempts", 11),
            ("ivr_keypress_delay", -1),
            ("ivr_keypress_delay", 16),
            ("ivr_max_hold_time", 14),
            ("ivr_max_hold_time", 901),
            ("dtmf_button_press_delay", -1),
            ("dtmf_button_press_delay", 11),
            ("dtmf_max_digits", 0),
            ("dtmf_max_digits", 101),
            ("dtmf_allowed_exceeded_attempts", 0),
            ("dtmf_allowed_exceeded_attempts", 51),
            ("ivr_action", "invalid_action"),
            ("ivr_navigation_mode", "unknown_mode"),
            ("dtmf_exceeded_action", "invalid_exceeded"),
        ],
    )
    def test_boundary_and_enum_validations_rejected(self, db, workspace, flow, field, value):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/ivr-dtmf-settings",
            json={field: value},
        )
        assert resp.status_code in (400, 422)

    def test_dtmf_end_call_message_blank_converts_to_none(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/ivr-dtmf-settings",
            json={"dtmf_end_call_message": "   "},
        )
        assert resp.status_code == 200
        assert resp.json()["dtmf_end_call_message"] is None

    def test_dtmf_end_call_message_exceeds_max_length_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/ivr-dtmf-settings",
            json={"dtmf_end_call_message": "a" * 501},
        )
        assert resp.status_code in (400, 422)

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/ivr-dtmf-settings",
            json={
                "ivr_enabled": True,
                "unknown_setting": "bad",
            },
        )
        assert resp.status_code in (400, 422)

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/ivr-dtmf-settings",
            json={"ivr_enabled": True},
        )
        assert resp.status_code == 404

    def test_other_tenant_flow_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_app(db, foreign_principal)

        resp = client.put(
            f"/flows/{flow.id}/ivr-dtmf-settings",
            json={"ivr_enabled": True},
        )
        assert resp.status_code == 404

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_audit:
            resp = client.put(
                f"/flows/{flow.id}/ivr-dtmf-settings",
                json={"ivr_enabled": True, "ivr_action": "hang_up"},
            )
            assert resp.status_code == 200

        mock_audit.assert_called_once()
        _, kwargs = mock_audit.call_args
        assert kwargs["action"] == "ivr_dtmf_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"]["ivr_enabled"] is True
        assert kwargs["new_value"]["ivr_action"] == "hang_up"

    def test_non_admin_forbidden_on_put(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for non_admin_role in ["config_only", "read_only", "billing_only"]:
            with _with_effective_role(non_admin_role):
                resp = client.put(
                    f"/flows/{flow.id}/ivr-dtmf-settings",
                    json={"ivr_enabled": True},
                )
                assert resp.status_code == 403


class TestGetIVRDTMFSettings:
    def test_fresh_flow_returns_defaults(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/ivr-dtmf-settings")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ivr_enabled"] is False
        assert body["ivr_action"] == "dial_through"
        assert body["ivr_navigation_mode"] == "let_ai_converse"
        assert body["ivr_max_attempts"] == 3
        assert body["ivr_keypress_delay"] == 8
        assert body["ivr_priority_list"] == []
        assert body["ivr_wait_on_hold"] is False
        assert body["ivr_max_hold_time"] == 120
        assert body["dtmf_enabled"] is False
        assert body["dtmf_button_press_delay"] == 2
        assert body["dtmf_allow_caller_interruption"] is False
        assert body["dtmf_max_digits"] == 50
        assert body["dtmf_allowed_exceeded_attempts"] == 10
        assert body["dtmf_exceeded_action"] == "end_call"
        assert body["dtmf_end_call_message"] == (
            "You've reached the maximum number of inputs allowed for this call."
        )

    def test_get_round_trips_prior_put_update(self, db, workspace, flow):
        admin_principal = _principal(workspace.id)
        admin_client = _build_app(db, admin_principal)

        put_resp = admin_client.put(
            f"/flows/{flow.id}/ivr-dtmf-settings",
            json={
                "ivr_enabled": True,
                "ivr_action": "hang_up",
                "dtmf_enabled": True,
                "dtmf_button_press_delay": 5,
            },
        )
        assert put_resp.status_code == 200

        readonly_principal = _principal(workspace.id)
        readonly_client = _build_readonly_app(db, readonly_principal)

        get_resp = readonly_client.get(f"/flows/{flow.id}/ivr-dtmf-settings")
        assert get_resp.status_code == 200
        assert get_resp.json() == put_resp.json()

    def test_readonly_user_can_access_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for role in ["read_only", "config_only", "manager", "admin", "owner"]:
            with _with_effective_role(role):
                resp = client.get(f"/flows/{flow.id}/ivr-dtmf-settings")
                assert resp.status_code == 200

    def test_get_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{uuid.uuid4()}/ivr-dtmf-settings")
        assert resp.status_code == 404

    def test_get_other_tenant_flow_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_readonly_app(db, foreign_principal)

        resp = client.get(f"/flows/{flow.id}/ivr-dtmf-settings")
        assert resp.status_code == 404

    def test_get_handles_null_attributes_fallback(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        mock_flow = MagicMock()
        mock_flow.id = flow.id
        mock_flow.tenant_id = workspace.id
        mock_flow.ivr_enabled = None
        mock_flow.ivr_action = None
        mock_flow.ivr_navigation_mode = None
        mock_flow.ivr_max_attempts = None
        mock_flow.ivr_keypress_delay = None
        mock_flow.ivr_priority_list = None
        mock_flow.ivr_wait_on_hold = None
        mock_flow.ivr_max_hold_time = None
        mock_flow.dtmf_enabled = None
        mock_flow.dtmf_button_press_delay = None
        mock_flow.dtmf_allow_caller_interruption = None
        mock_flow.dtmf_max_digits = None
        mock_flow.dtmf_allowed_exceeded_attempts = None
        mock_flow.dtmf_exceeded_action = None
        mock_flow.dtmf_end_call_message = None

        with patch(
            "app.services.call_flow_service.call_flow_service._get_flow_or_404",
            return_value=mock_flow,
        ):
            resp = client.get(f"/flows/{flow.id}/ivr-dtmf-settings")
            assert resp.status_code == 200
            body = resp.json()
            assert body["ivr_enabled"] is False
            assert body["ivr_action"] == "dial_through"
            assert body["ivr_navigation_mode"] == "let_ai_converse"
            assert body["ivr_max_attempts"] == 3
            assert body["ivr_keypress_delay"] == 8
            assert body["ivr_priority_list"] == []
            assert body["ivr_wait_on_hold"] is False
            assert body["ivr_max_hold_time"] == 120
            assert body["dtmf_enabled"] is False
            assert body["dtmf_button_press_delay"] == 2
            assert body["dtmf_allow_caller_interruption"] is False
            assert body["dtmf_max_digits"] == 50
            assert body["dtmf_allowed_exceeded_attempts"] == 10
            assert body["dtmf_exceeded_action"] == "end_call"
            assert body["dtmf_end_call_message"] is None
