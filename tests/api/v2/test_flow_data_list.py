"""Tests for GET /api/v2/flows/flow-data — paginated visual/pre-compiled
flow-graph listing scoped to the caller's workspace.

Coverage:
  - Happy path: paginated list, correct camelCase field names.
  - Tenant isolation: another tenant's flows never appear.
  - Soft-deleted flows are excluded.
  - Pagination: page/pageSize control the slice, total reflects full count.
  - Auth: unauthenticated request is rejected by require_readonly_or_api_key.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal, *, override_auth=True):
    from app.api.deps import (
        get_db,
        require_readonly_or_api_key,
    )
    from app.api.v2.routers.flow_data import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    if override_auth:
        mini.dependency_overrides[require_readonly_or_api_key] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    return principal


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"FlowDataListWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"flow_data_list_ws_{uuid.uuid4().hex[:8]}",
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
        name=f"FlowDataListWS-Other-{uuid.uuid4().hex[:8]}",
        schema_name=f"flow_data_list_ws_other_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


def _agent(db, tenant):
    from app.models.agent import Agent

    a = Agent(
        tenant_id=tenant.id,
        name="Flow Data List Test Agent",
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


def _flow(db, tenant, agent, *, name="Flow", is_deleted=False, flow_data=None, flow_data_compiled=None):
    from app.models.call_flow import CallFlow

    f = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name=name,
        direction="inbound",
        is_deleted=is_deleted,
        flow_data=flow_data,
        flow_data_compiled=flow_data_compiled,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestListFlowData:
    def test_happy_path_returns_paginated_camel_case_response(self, db, workspace):
        agent = _agent(db, workspace)
        flow = _flow(
            db,
            workspace,
            agent,
            name="My Flow",
            flow_data={"nodes": [{"id": "start", "type": "start", "data": {}}]},
            flow_data_compiled={"start": {"node": {"type": "start"}}},
        )

        client = _build_app(db, _principal(workspace.id))
        resp = client.get("/flows/flow-data")

        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["total"] >= 1
        assert body["page"] == 1
        assert body["pageSize"] == 20

        item = next(i for i in body["data"] if i["flowId"] == str(flow.id))
        assert item["name"] == "My Flow"
        assert item["flowData"] == {"nodes": [{"id": "start", "type": "start", "data": {}}]}
        assert item["flowDataCompiled"] == {"start": {"node": {"type": "start"}}}
        assert "updatedAt" in item
        # confirm no snake_case leakage
        assert "flow_id" not in item
        assert "flow_data" not in item
        assert "flow_data_compiled" not in item
        assert "updated_at" not in item

    def test_tenant_isolation_excludes_other_tenants_flows(
        self, db, workspace, other_workspace
    ):
        agent_a = _agent(db, workspace)
        agent_b = _agent(db, other_workspace)
        flow_a = _flow(db, workspace, agent_a, name="Tenant A Flow")
        flow_b = _flow(db, other_workspace, agent_b, name="Tenant B Flow")

        client = _build_app(db, _principal(workspace.id))
        resp = client.get("/flows/flow-data")

        assert resp.status_code == 200, resp.text
        ids = [i["flowId"] for i in resp.json()["data"]]
        assert str(flow_a.id) in ids
        assert str(flow_b.id) not in ids

    def test_soft_deleted_flows_are_excluded(self, db, workspace):
        agent = _agent(db, workspace)
        active_flow = _flow(db, workspace, agent, name="Active Flow")
        deleted_flow = _flow(
            db, workspace, agent, name="Deleted Flow", is_deleted=True
        )

        client = _build_app(db, _principal(workspace.id))
        resp = client.get("/flows/flow-data")

        assert resp.status_code == 200, resp.text
        ids = [i["flowId"] for i in resp.json()["data"]]
        assert str(active_flow.id) in ids
        assert str(deleted_flow.id) not in ids

    def test_pagination_slices_results_and_total_reflects_full_count(
        self, db, workspace
    ):
        agent = _agent(db, workspace)
        flows = [
            _flow(db, workspace, agent, name=f"Paginated Flow {i}") for i in range(5)
        ]

        client = _build_app(db, _principal(workspace.id))

        resp_page1 = client.get("/flows/flow-data?page=1&pageSize=2")
        assert resp_page1.status_code == 200, resp_page1.text
        body1 = resp_page1.json()
        assert body1["page"] == 1
        assert body1["pageSize"] == 2
        assert len(body1["data"]) == 2
        assert body1["total"] >= 5

        resp_page2 = client.get("/flows/flow-data?page=2&pageSize=2")
        assert resp_page2.status_code == 200, resp_page2.text
        body2 = resp_page2.json()
        assert body2["page"] == 2
        assert len(body2["data"]) == 2

        # No overlap between page 1 and page 2 for this tenant's flows
        ids_page1 = {i["flowId"] for i in body1["data"]}
        ids_page2 = {i["flowId"] for i in body2["data"]}
        assert ids_page1.isdisjoint(ids_page2)

    def test_unauthenticated_request_is_rejected(self, db, workspace):
        client = _build_app(db, _principal(workspace.id), override_auth=False)

        resp = client.get("/flows/flow-data")

        assert resp.status_code == 401, resp.text
