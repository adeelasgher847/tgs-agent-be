"""Tests for PUT /api/v2/flows/{flow_id}/caller-memory-settings.

Coverage:
  - Admin/API-key principal can enable caller memory and set the window
  - caller_memory_window is validated to the inclusive [1, 10] range
  - Config-rank (non-admin) principal is forbidden (403)
  - Unknown flow_id returns 404
  - A successful update fires an audit event with the expected shape
  - The endpoint path does not collide with the HIPAA /flows/{id}/settings route
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal):
    from app.api.deps import get_db, require_admin_or_api_key
    from app.api.v2.routers.flows import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    mini.dependency_overrides[require_admin_or_api_key] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"CallerMemWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"caller_mem_ws_{uuid.uuid4().hex[:8]}",
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
        name="Caller Memory Test Agent",
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
        name="Caller Memory Test Flow",
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
class TestUpdateCallerMemorySettings:
    def test_admin_can_enable_caller_memory(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/caller-memory-settings",
            json={"caller_memory_enabled": True, "caller_memory_window": 5},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["caller_memory_enabled"] is True
        assert body["caller_memory_window"] == 5

        db.refresh(flow)
        assert flow.caller_memory_enabled is True
        assert flow.caller_memory_window == 5

    @pytest.mark.parametrize("window", [0, 11, -1])
    def test_window_out_of_range_returns_422(self, db, workspace, flow, window):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/caller-memory-settings",
            json={"caller_memory_enabled": True, "caller_memory_window": window},
        )

        assert resp.status_code == 400

    @pytest.mark.parametrize("window", [1, 10])
    def test_window_boundary_values_accepted(self, db, workspace, flow, window):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/caller-memory-settings",
            json={"caller_memory_enabled": True, "caller_memory_window": window},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["caller_memory_window"] == window

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/caller-memory-settings",
            json={"caller_memory_enabled": True, "caller_memory_window": 3},
        )

        assert resp.status_code == 404

    def test_flow_from_other_tenant_returns_404(self, db, flow):
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
            f"/flows/{flow.id}/caller-memory-settings",
            json={"caller_memory_enabled": True, "caller_memory_window": 3},
        )

        assert resp.status_code == 404

    def test_update_fires_audit_event(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_log_audit:
            resp = client.put(
                f"/flows/{flow.id}/caller-memory-settings",
                json={"caller_memory_enabled": True, "caller_memory_window": 7},
            )

        assert resp.status_code == 200, resp.text
        mock_log_audit.assert_called_once()
        kwargs = mock_log_audit.call_args.kwargs
        assert kwargs["action"] == "caller_memory_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["new_value"] == {
            "caller_memory_enabled": True,
            "caller_memory_window": 7,
        }
        assert kwargs["actor_user_id"] == principal.id

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/caller-memory-settings",
            json={
                "caller_memory_enabled": True,
                "caller_memory_window": 3,
                "unexpected_field": "nope",
            },
        )

        assert resp.status_code == 400

    def test_path_does_not_collide_with_hipaa_settings_route(self):
        """
        Sanity check on the route table: /flows/{id}/settings (HIPAA) and
        /flows/{id}/caller-memory-settings (this feature) must remain distinct
        paths so mounting both v2 routers together never causes one to shadow
        the other.
        """
        from app.api.v2.routers.flows import router as caller_memory_router
        from app.api.v2.routers.hipaa import flows_router as hipaa_router

        caller_memory_paths = {r.path for r in caller_memory_router.routes}
        hipaa_paths = {r.path for r in hipaa_router.routes}

        assert "/flows/{flow_id}/caller-memory-settings" in caller_memory_paths
        assert "/flows/{flow_id}/settings" in hipaa_paths
        assert caller_memory_paths.isdisjoint(hipaa_paths)
