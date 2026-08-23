"""Tests for the Visual Flow Editor (Sprint 6).

Coverage:
  - PUT /api/v2/flows/{flow_id}/flow-data: valid graphs save + pre-compile;
    invalid graphs (missing greeting, invalid entry_node_id, cycle, orphan
    node, missing outgoing edge) are rejected with 422 and a detailed error
    array.
  - PUT /api/v2/flows/{flow_id}/flow-data: version increments and
    compiled_at is stamped on every save, per the ticket's flow_data schema.
  - GET /api/v2/flows/{flow_id}/flow-data: returns saved flow_data +
    flow_data_compiled.
  - POST /api/v2/flows/{flow_id}/validate: validates without saving.
  - Graph compilation: next_nodes handle map (the ticket's flat
    ``{node_id: {type, data, next_nodes}}`` compiled-plan shape).
  - FlowExecutor: isolated node-executor sequence greeting -> collect_input ->
    branch -> transfer, entry point resolved via entry_node_id (not a
    separate "start" node type).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers
from app.services.flow_graph_service import compile_graph, validate_graph
from app.voice.flow_executor import FlowExecutor, PipelineState


def _build_app(db_override, principal):
    from app.api.deps import (
        get_db,
        require_config_or_api_key,
        require_readonly_or_api_key,
    )
    from app.api.v2.routers.flow_data import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    mini.dependency_overrides[require_config_or_api_key] = lambda: principal
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
        name=f"FlowEditorWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"flow_editor_ws_{uuid.uuid4().hex[:8]}",
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
        name="Flow Editor Test Agent",
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
        name="Flow Editor Test Flow",
        direction="inbound",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _valid_flow_data() -> dict:
    return {
        "entry_node_id": "greet",
        "nodes": [
            {"id": "greet", "type": "greeting", "data": {"message": "Hi there!"}},
            {
                "id": "collect",
                "type": "collect_input",
                "data": {
                    "variable_name": "call_reason",
                    "prompt": "What can I help you with?",
                },
            },
            {
                "id": "branch",
                "type": "branch",
                "data": {
                    "condition_variable": "call_reason",
                    "operator": "contains",
                    "condition_value": "human",
                },
            },
            {"id": "transfer", "type": "transfer", "data": {}},
            {"id": "end", "type": "end_call", "data": {}},
        ],
        "edges": [
            {"id": "e1", "source": "greet", "target": "collect"},
            {"id": "e2", "source": "collect", "target": "branch"},
            {
                "id": "e3",
                "source": "branch",
                "target": "transfer",
                "sourceHandle": "yes",
            },
            {"id": "e4", "source": "branch", "target": "end", "sourceHandle": "no"},
        ],
    }


# ── PUT /flow-data — validation gate ──────────────────────────────────────


class TestUpdateFlowData:
    def test_valid_graph_saves_and_compiles(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))

        resp = client.put(
            f"/flows/{flow.id}/flow-data", json={"flowData": _valid_flow_data()}
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # PUT response is the ticket-literal {version, validated} shape.
        assert body == {"version": 1, "validated": True}

        db.refresh(flow)
        assert flow.flow_data is not None
        assert flow.flow_data["version"] == 1
        assert flow.flow_data["compiled_at"]
        assert flow.compiled_plan is not None
        assert flow.compiled_plan["greet"]["type"] == "greeting"
        assert flow.compiled_plan["greet"]["next_nodes"] == {"default": "collect"}

    def test_version_increments_on_every_save(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))

        first = client.put(
            f"/flows/{flow.id}/flow-data", json={"flowData": _valid_flow_data()}
        )
        second = client.put(
            f"/flows/{flow.id}/flow-data", json={"flowData": _valid_flow_data()}
        )

        assert first.json()["version"] == 1
        assert second.json()["version"] == 2

    def test_missing_greeting_node_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"][0]["type"] = "collect_input"  # no greeting node left

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "no_greeting_node" in codes

    def test_multiple_greeting_nodes_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"][1]["type"] = "greeting"  # collect -> a second greeting node

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "multiple_greeting_nodes" in codes

    def test_missing_entry_node_id_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        del data["entry_node_id"]

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "missing_entry_node_id" in codes

    def test_entry_node_id_not_greeting_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["entry_node_id"] = "collect"  # points at a non-greeting node

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "entry_node_not_greeting" in codes

    def test_cycle_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        # Introduce a cycle: end -> branch (end was previously terminal)
        data["edges"].append({"id": "e5", "source": "end", "target": "branch"})

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "cycle_detected" in codes

    def test_orphan_node_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"].append(
            {"id": "orphan", "type": "kb_lookup", "data": {}}
        )
        data["edges"].append({"id": "e6", "source": "orphan", "target": "end"})

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "orphan_node" in codes

    def test_missing_outgoing_edge_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        # Drop branch's edges entirely — non-terminal node with no outgoing edge.
        data["edges"] = [e for e in data["edges"] if e["source"] != "branch"]

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "missing_outgoing_edge" in codes

    def test_unknown_flow_returns_404(self, db, workspace):
        client = _build_app(db, _principal(workspace.id))

        resp = client.put(
            f"/flows/{uuid.uuid4()}/flow-data", json={"flowData": _valid_flow_data()}
        )

        assert resp.status_code == 404


# ── GET /flow-data + POST /validate ────────────────────────────────────────


class TestGetAndValidateFlowData:
    def test_get_returns_saved_flow_and_compiled_graph(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        client.put(f"/flows/{flow.id}/flow-data", json={"flowData": _valid_flow_data()})

        resp = client.get(f"/flows/{flow.id}/flow-data")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["flowData"] is not None
        assert body["flowDataCompiled"] is not None
        assert body["validationErrors"] == []

    def test_get_on_empty_flow_returns_nulls(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))

        resp = client.get(f"/flows/{flow.id}/flow-data")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["flowData"] is None
        assert body["flowDataCompiled"] is None

    def test_validate_endpoint_reports_errors_without_saving(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"][0]["type"] = "collect_input"  # break the greeting invariant

        resp = client.post(f"/flows/{flow.id}/validate", json={"flowData": data})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["valid"] is False
        assert any("greeting" in e["message"] for e in body["errors"])

        db.refresh(flow)
        assert flow.flow_data is None  # nothing was persisted

    def test_validate_endpoint_valid_graph(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))

        resp = client.post(
            f"/flows/{flow.id}/validate",
            json={"flowData": _valid_flow_data()},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"valid": True, "errors": []}


# ── Graph compilation: compiled-plan shape ────────────────────────────────
class TestCompileGraphShape:
    def test_compiled_plan_is_flat_type_data_next_nodes_lookup(self):
        data = _valid_flow_data()

        compiled = compile_graph(data)

        assert compiled["branch"]["type"] == "branch"
        assert compiled["branch"]["data"]["condition_variable"] == "call_reason"
        assert compiled["branch"]["next_nodes"] == {"yes": "transfer", "no": "end"}
        assert compiled["greet"]["next_nodes"] == {"default": "collect"}

    def test_validate_graph_accepts_well_formed_flow(self):
        assert validate_graph(_valid_flow_data()) == []


# ── FlowExecutor: isolated node-executor sequence ─────────────────────────


class TestFlowExecutor:
    def test_full_sequence_greeting_collect_branch_transfer(self):
        data = _valid_flow_data()
        compiled = compile_graph(data)
        executor = FlowExecutor(compiled, data["entry_node_id"])
        state = PipelineState(current_node_id=executor.start_node_id())

        # entry_node_id resolves directly to the greeting node — no "start" node.
        assert executor.start_node_id() == "greet"
        greet_result = executor.execute_node(state.current_node_id, state)
        assert greet_result.node_type == "greeting"
        assert greet_result.action == "speak"
        assert greet_result.speech_text == "Hi there!"

        # greeting -> collect_input (single default edge)
        next_id = executor.next_node_id(state.current_node_id, None, state.variables)
        collect_result = executor.execute_node(next_id, state)
        assert collect_result.node_type == "collect_input"
        assert collect_result.action == "wait_for_input"

        # caller's answer to collect_input is stored by the pipeline mixin;
        # simulate it directly here since FlowExecutor itself doesn't store variables.
        state.variables["call_reason"] = "I'd like to speak to a human please"

        # collect_input -> branch (single default edge)
        next_id = executor.next_node_id(state.current_node_id, None, state.variables)
        branch_result = executor.execute_node(next_id, state)
        assert branch_result.node_type == "branch"
        assert branch_result.action == "branch"

        # branch evaluates condition_variable/operator/condition_value from its
        # own data against flow_variables — "human" is contained -> yes -> transfer
        next_id = executor.next_node_id(state.current_node_id, None, state.variables)
        transfer_result = executor.execute_node(next_id, state)
        assert transfer_result.node_type == "transfer"
        assert transfer_result.action == "transfer"

        assert state.history == ["greet", "collect", "branch", "transfer"]

    def test_branch_no_path_when_condition_does_not_match(self):
        data = _valid_flow_data()
        compiled = compile_graph(data)
        executor = FlowExecutor(compiled, data["entry_node_id"])

        target = executor.next_node_id(
            "branch", None, {"call_reason": "billing question"}
        )

        assert target == "end"
