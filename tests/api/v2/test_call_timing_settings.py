"""Tests for PUT and GET /api/v2/flows/{flow_id}/call-timing-settings.

Coverage:
  - Admin/API-key principal can update all Call Timing settings
  - Default values on fresh flows
  - Boundary validations (silence_timeout 3-60, end_call_after_reminder 3-60, reminder_retries 1-3, max_call_duration 60-7200)
  - String handling for max_duration_message (trimming, blank to None, max 500 length)
  - List handling for reminder_messages (filtering blank strings, max 10 entries)
  - Extra fields rejected (extra="forbid")
  - Non-admin (config_only, read_only, billing_only) forbidden on PUT (403)
  - Read-only rank is sufficient for GET (200)
  - Tenant isolation: unknown flow_id or other tenant flow returns 404
  - Audit event logged on PUT (action="call_timing_settings.updated")
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
        name=f"TimingWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"timing_ws_{uuid.uuid4().hex[:8]}",
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
        name=f"OtherTimingWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_timing_ws_{uuid.uuid4().hex[:8]}",
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
        name="Timing Test Agent",
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
        name="Call Timing Test Flow",
        direction="outbound",
        status="active",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestUpdateCallTimingSettings:
    def test_admin_can_update_all_call_timing_settings(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "silence_timeout": 15,
            "end_call_after_reminder": 20,
            "reminder_retries": 2,
            "reminder_messages": [
                "Hello, are you still there?",
                "I haven't heard from you in a moment.",
            ],
            "max_call_duration": 3600,
            "max_duration_message": "Our scheduled call time is up. Have a great day!",
        }

        resp = client.put(f"/flows/{flow.id}/call-timing-settings", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for k, v in payload.items():
            assert body[k] == v

        db.refresh(flow)
        assert flow.silence_timeout == 15
        assert flow.end_call_after_reminder == 20
        assert flow.reminder_retries == 2
        assert flow.reminder_messages == [
            "Hello, are you still there?",
            "I haven't heard from you in a moment.",
        ]
        assert flow.max_call_duration == 3600
        assert (
            flow.max_duration_message
            == "Our scheduled call time is up. Have a great day!"
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("silence_timeout", 2),
            ("silence_timeout", 61),
            ("end_call_after_reminder", 2),
            ("end_call_after_reminder", 61),
            ("reminder_retries", 0),
            ("reminder_retries", 4),
            ("max_call_duration", 59),
            ("max_call_duration", 7201),
        ],
    )
    def test_boundary_validations_rejected(
        self, db, workspace, flow, field, value
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={field: value},
        )
        assert resp.status_code in (400, 422)

    def test_max_duration_message_blank_converts_to_none(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"max_duration_message": "   "},
        )
        assert resp.status_code == 200
        assert resp.json()["max_duration_message"] is None

    def test_max_duration_message_over_max_length_rejected(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"max_duration_message": "a" * 501},
        )
        assert resp.status_code in (400, 422)

    def test_reminder_messages_strips_blanks(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={
                "reminder_messages": [
                    "Are you there?",
                    "   ",
                    "Hello?",
                    "",
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["reminder_messages"] == ["Are you there?", "Hello?"]

    def test_reminder_messages_max_limit_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"reminder_messages": [f"Msg {i}" for i in range(11)]},
        )
        assert resp.status_code in (400, 422)

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={
                "silence_timeout": 10,
                "unexpected_field": "val",
            },
        )
        assert resp.status_code in (400, 422)

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/call-timing-settings",
            json={"silence_timeout": 10},
        )
        assert resp.status_code == 404

    def test_other_tenant_flow_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_app(db, foreign_principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"silence_timeout": 10},
        )
        assert resp.status_code == 404

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_audit:
            resp = client.put(
                f"/flows/{flow.id}/call-timing-settings",
                json={"silence_timeout": 20, "max_call_duration": 900},
            )
            assert resp.status_code == 200

        mock_audit.assert_called_once()
        _, kwargs = mock_audit.call_args
        assert kwargs["action"] == "call_timing_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"]["silence_timeout"] == 20
        assert kwargs["new_value"]["max_call_duration"] == 900

    def test_non_admin_forbidden_on_put(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for non_admin_role in ["config_only", "read_only", "billing_only"]:
            with _with_effective_role(non_admin_role):
                resp = client.put(
                    f"/flows/{flow.id}/call-timing-settings",
                    json={"silence_timeout": 15},
                )
                assert resp.status_code == 403


class TestGetCallTimingSettings:
    def test_fresh_flow_returns_defaults(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/call-timing-settings")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["silence_timeout"] == 10
        assert body["end_call_after_reminder"] == 10
        assert body["reminder_retries"] == 1
        assert body["reminder_messages"] == []
        assert body["max_call_duration"] == 1800
        assert body["max_duration_message"] == (
            "I appreciate the conversation, but we've reached our time limit for this call."
        )

    def test_get_round_trips_prior_put_update(self, db, workspace, flow):
        admin_principal = _principal(workspace.id)
        admin_client = _build_app(db, admin_principal)

        put_resp = admin_client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={
                "silence_timeout": 25,
                "reminder_retries": 3,
                "reminder_messages": ["Still there?"],
                "max_call_duration": 2400,
            },
        )
        assert put_resp.status_code == 200

        readonly_principal = _principal(workspace.id)
        readonly_client = _build_readonly_app(db, readonly_principal)

        get_resp = readonly_client.get(f"/flows/{flow.id}/call-timing-settings")
        assert get_resp.status_code == 200
        assert get_resp.json() == put_resp.json()

    def test_readonly_user_can_access_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app_for_real_rank_check(db, principal)

        for role in ["read_only", "config_only", "manager", "admin", "owner"]:
            with _with_effective_role(role):
                resp = client.get(f"/flows/{flow.id}/call-timing-settings")
                assert resp.status_code == 200

    def test_get_unknown_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{uuid.uuid4()}/call-timing-settings")
        assert resp.status_code == 404

    def test_get_other_tenant_flow_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        client = _build_readonly_app(db, foreign_principal)

        resp = client.get(f"/flows/{flow.id}/call-timing-settings")
        assert resp.status_code == 404

    def test_get_handles_null_attributes_fallback(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        mock_flow = MagicMock()
        mock_flow.id = flow.id
        mock_flow.tenant_id = workspace.id
        mock_flow.silence_timeout = None
        mock_flow.end_call_after_reminder = None
        mock_flow.reminder_retries = None
        mock_flow.reminder_messages = None
        mock_flow.max_call_duration = None
        mock_flow.max_duration_message = None

        with patch(
            "app.services.call_flow_service.call_flow_service._get_flow_or_404",
            return_value=mock_flow,
        ):
            resp = client.get(f"/flows/{flow.id}/call-timing-settings")
            assert resp.status_code == 200
            body = resp.json()
            assert body["silence_timeout"] == 10
            assert body["end_call_after_reminder"] == 10
            assert body["reminder_retries"] == 1
            assert body["reminder_messages"] == []
            assert body["max_call_duration"] == 1800
            assert body["max_duration_message"] is None

    def test_partial_update_preserves_unmodified_fields(
        self, db, workspace, flow
    ):
        admin_principal = _principal(workspace.id)
        admin_client = _build_app(db, admin_principal)

        resp1 = admin_client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={
                "silence_timeout": 20,
                "reminder_messages": ["Custom reminder 1"],
                "max_call_duration": 2400,
            },
        )
        assert resp1.status_code == 200
        assert resp1.json()["silence_timeout"] == 20
        assert resp1.json()["reminder_messages"] == ["Custom reminder 1"]
        assert resp1.json()["max_call_duration"] == 2400

        # Partial update: only change silence_timeout
        resp2 = admin_client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"silence_timeout": 30},
        )
        assert resp2.status_code == 200
        assert resp2.json()["silence_timeout"] == 30
        assert resp2.json()["reminder_messages"] == ["Custom reminder 1"]
        assert resp2.json()["max_call_duration"] == 2400

    def test_reminder_messages_invalid_type_rejected(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"reminder_messages": "not-a-list"},
        )
        assert resp.status_code in (400, 422)

        resp_items = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"reminder_messages": [123, 456]},
        )
        assert resp_items.status_code in (400, 422)

    def test_null_reminder_messages_coerces_to_empty_list(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/call-timing-settings",
            json={"reminder_messages": None},
        )
        assert resp.status_code == 200
        assert resp.json()["reminder_messages"] == []
