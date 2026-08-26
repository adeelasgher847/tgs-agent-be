"""Tests for PUT and GET /api/v2/flows/{flow_id}/inbound-redirect-settings.

Coverage:
  - Admin/API-key principal can update all Inbound Call Redirection settings
  - Default values on fresh flows
  - Boundary validations (operators: exists, not_empty, equals, not_equals; max 20 conditions; max 500 chars message)
  - String handling for redirect_message and redirect_forward_phone_number (trimming, blank to None)
  - Extra fields rejected (extra="forbid")
  - Non-admin (config_only, read_only, billing_only) forbidden on PUT (403)
  - Read-only rank is sufficient for GET (200)
  - Tenant isolation: unknown flow_id or other tenant flow returns 404
  - Audit event logged on PUT (action="inbound_redirect_settings.updated")
  - GET round-trips prior PUT updates
  - Partial updates preserve unmodified fields
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
        name=f"RedirectWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"redirect_ws_{uuid.uuid4().hex[:8]}",
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
        name=f"OtherRedirectWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_redirect_ws_{uuid.uuid4().hex[:8]}",
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
        name="Redirect Test Agent",
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
        name="Redirect Test Flow",
        direction="inbound",
        status="active",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestUpdateInboundRedirectSettings:
    def test_admin_can_update_all_redirect_settings(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "redirect_inbound_calls_enabled": True,
            "redirect_forward_phone_number": "+14155552671",
            "redirect_conditions": [
                {
                    "variable": "tier",
                    "operator": "equals",
                    "value": "vip",
                },
                {
                    "variable": "{{_metadata.company}}",
                    "operator": "exists",
                    "value": None,
                },
            ],
            "redirect_speak_message_enabled": True,
            "redirect_message": "Please hold while we forward you to a specialist at {{_metadata.company}}.",
        }

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings", json=payload
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["redirect_inbound_calls_enabled"] is True
        assert body["redirect_forward_phone_number"] == "+14155552671"
        assert len(body["redirect_conditions"]) == 2
        assert body["redirect_conditions"][0]["variable"] == "tier"
        assert body["redirect_conditions"][0]["operator"] == "equals"
        assert body["redirect_conditions"][0]["value"] == "vip"
        assert body["redirect_speak_message_enabled"] is True
        assert (
            body["redirect_message"]
            == "Please hold while we forward you to a specialist at {{_metadata.company}}."
        )

        db.refresh(flow)
        assert flow.redirect_inbound_calls_enabled is True
        assert flow.redirect_forward_phone_number == "+14155552671"
        assert len(flow.redirect_conditions) == 2
        assert flow.redirect_speak_message_enabled is True

    def test_invalid_operator_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_conditions": [
                    {
                        "variable": "tier",
                        "operator": "invalid_op",
                        "value": "vip",
                    }
                ]
            },
        )
        assert resp.status_code in (400, 422)

    def test_redirect_message_blank_converts_to_none(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_message": "   "},
        )
        assert resp.status_code == 200
        assert resp.json()["redirect_message"] is None

    def test_redirect_forward_phone_number_blank_converts_to_none(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_forward_phone_number": "   "},
        )
        assert resp.status_code == 200
        assert resp.json()["redirect_forward_phone_number"] is None

    def test_redirect_message_over_max_length_rejected(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_message": "a" * 501},
        )
        assert resp.status_code in (400, 422)

    def test_conditions_over_max_limit_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        conditions = [
            {"variable": f"var_{i}", "operator": "exists"} for i in range(21)
        ]
        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_conditions": conditions},
        )
        assert resp.status_code in (400, 422)

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_inbound_calls_enabled": True,
                "unexpected_field": "val",
            },
        )
        assert resp.status_code in (400, 422)

    def test_extra_fields_inside_condition_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_conditions": [
                    {
                        "variable": "tier",
                        "operator": "equals",
                        "value": "vip",
                        "unsupported_field": "disallowed",
                    }
                ]
            },
        )
        assert resp.status_code in (400, 422)

    def test_condition_variable_empty_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_conditions": [
                    {
                        "variable": "   ",
                        "operator": "exists",
                    }
                ]
            },
        )
        assert resp.status_code in (400, 422)

    def test_condition_variable_over_max_length_rejected(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_conditions": [
                    {
                        "variable": "a" * 101,
                        "operator": "exists",
                    }
                ]
            },
        )
        assert resp.status_code in (400, 422)

    def test_condition_value_over_max_length_rejected(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_conditions": [
                    {
                        "variable": "tier",
                        "operator": "equals",
                        "value": "v" * 256,
                    }
                ]
            },
        )
        assert resp.status_code in (400, 422)

    def test_redirect_forward_phone_number_over_max_length_rejected(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_forward_phone_number": "+1" + "9" * 50},
        )
        assert resp.status_code in (400, 422)

    def test_clearing_fields_with_explicit_null(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        # First set fields
        client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_inbound_calls_enabled": True,
                "redirect_forward_phone_number": "+14155552671",
                "redirect_speak_message_enabled": True,
                "redirect_message": "Connecting call.",
            },
        )

        # Clear them with None / empty
        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_forward_phone_number": None,
                "redirect_message": None,
                "redirect_conditions": [],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["redirect_forward_phone_number"] is None
        assert body["redirect_message"] is None
        assert body["redirect_conditions"] == []

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/inbound-redirect-settings",
            json={"redirect_inbound_calls_enabled": True},
        )
        assert resp.status_code == 404

    def test_other_tenant_flow_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_app(db, foreign_principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_inbound_calls_enabled": True},
        )
        assert resp.status_code == 404

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_audit:
            resp = client.put(
                f"/flows/{flow.id}/inbound-redirect-settings",
                json={
                    "redirect_inbound_calls_enabled": True,
                    "redirect_forward_phone_number": "+14155552671",
                },
            )
            assert resp.status_code == 200

        mock_audit.assert_called_once()
        _, kwargs = mock_audit.call_args
        assert kwargs["action"] == "inbound_redirect_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"]["redirect_inbound_calls_enabled"] is True
        assert (
            kwargs["new_value"]["redirect_forward_phone_number"]
            == "+14155552671"
        )

    def test_non_admin_forbidden_on_put(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for non_admin_role in ["config_only", "read_only", "billing_only"]:
            with _with_effective_role(non_admin_role):
                resp = client.put(
                    f"/flows/{flow.id}/inbound-redirect-settings",
                    json={"redirect_inbound_calls_enabled": True},
                )
                assert resp.status_code == 403


class TestGetInboundRedirectSettings:
    def test_fresh_flow_returns_defaults(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/inbound-redirect-settings")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["redirect_inbound_calls_enabled"] is False
        assert body["redirect_forward_phone_number"] is None
        assert body["redirect_conditions"] == []
        assert body["redirect_speak_message_enabled"] is False
        assert body["redirect_message"] is None

    def test_get_round_trips_prior_put_update(self, db, workspace, flow):
        admin_principal = _principal(workspace.id)
        admin_client = _build_app(db, admin_principal)

        put_resp = admin_client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_inbound_calls_enabled": True,
                "redirect_forward_phone_number": "+14155552671",
                "redirect_conditions": [
                    {
                        "variable": "account_type",
                        "operator": "not_empty",
                        "value": None,
                    }
                ],
                "redirect_speak_message_enabled": True,
                "redirect_message": "Connecting you now.",
            },
        )
        assert put_resp.status_code == 200

        readonly_principal = _principal(workspace.id)
        readonly_client = _build_readonly_app(db, readonly_principal)

        get_resp = readonly_client.get(
            f"/flows/{flow.id}/inbound-redirect-settings"
        )
        assert get_resp.status_code == 200
        assert get_resp.json() == put_resp.json()

    def test_readonly_user_can_access_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for role in ["read_only", "config_only", "manager", "admin", "owner"]:
            with _with_effective_role(role):
                resp = client.get(f"/flows/{flow.id}/inbound-redirect-settings")
                assert resp.status_code == 200

    def test_get_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{uuid.uuid4()}/inbound-redirect-settings")
        assert resp.status_code == 404

    def test_get_other_tenant_flow_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_readonly_app(db, foreign_principal)

        resp = client.get(f"/flows/{flow.id}/inbound-redirect-settings")
        assert resp.status_code == 404

    def test_partial_update_preserves_unmodified_fields(
        self, db, workspace, flow
    ):
        admin_principal = _principal(workspace.id)
        admin_client = _build_app(db, admin_principal)

        admin_client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={
                "redirect_inbound_calls_enabled": True,
                "redirect_forward_phone_number": "+14155552671",
                "redirect_speak_message_enabled": True,
                "redirect_message": "Connecting you.",
            },
        )

        resp = admin_client.put(
            f"/flows/{flow.id}/inbound-redirect-settings",
            json={"redirect_inbound_calls_enabled": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["redirect_inbound_calls_enabled"] is False
        assert body["redirect_forward_phone_number"] == "+14155552671"
        assert body["redirect_message"] == "Connecting you."

    def test_get_handles_null_attributes_fallback(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        mock_flow = MagicMock()
        mock_flow.id = flow.id
        mock_flow.tenant_id = workspace.id
        mock_flow.redirect_inbound_calls_enabled = None
        mock_flow.redirect_forward_phone_number = None
        mock_flow.redirect_conditions = None
        mock_flow.redirect_speak_message_enabled = None
        mock_flow.redirect_message = None

        with patch(
            "app.services.call_flow_service.call_flow_service._get_flow_or_404",
            return_value=mock_flow,
        ):
            resp = client.get(f"/flows/{flow.id}/inbound-redirect-settings")
            assert resp.status_code == 200
            body = resp.json()
            assert body["redirect_inbound_calls_enabled"] is False
            assert body["redirect_forward_phone_number"] is None
            assert body["redirect_conditions"] == []
            assert body["redirect_speak_message_enabled"] is False
            assert body["redirect_message"] is None
