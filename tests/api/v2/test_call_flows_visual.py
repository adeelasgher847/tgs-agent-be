"""Tests for the Visual Flow Editor (Sprint 6).

Coverage:
  - PUT /api/v2/flows/{flow_id}/flow-data: valid graphs save + pre-compile;
    invalid graphs (missing start, cycle, orphan node, missing outgoing
    edge) are rejected with 422 and a detailed error array.
  - GET /api/v2/flows/{flow_id}/flow-data: returns saved flow_data +
    flow_data_compiled.
  - GET /api/v2/flows/{flow_id}/flow-data/validate: validates without saving.
  - Graph compilation: edge priority ordering (intent_match < keyword < fallback).
  - FlowExecutor: isolated node-executor sequence greeting -> collect_input ->
    branch -> transfer.
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
        "nodes": [
            {"id": "start", "type": "start", "data": {}},
            {"id": "greet", "type": "greeting", "data": {"message": "Hi there!"}},
            {
                "id": "collect",
                "type": "collect_input",
                "data": {
                    "timeout_seconds": 10,
                    "silence_threshold_ms": 700,
                    "max_attempts": 3,
                },
            },
            {"id": "branch", "type": "branch", "data": {"variable": "transcript"}},
            {"id": "transfer", "type": "transfer", "data": {}},
            {"id": "end", "type": "end_call", "data": {}},
        ],
        "edges": [
            {
                "id": "e1",
                "source": "start",
                "target": "greet",
                "condition": {"type": "always"},
            },
            {
                "id": "e2",
                "source": "greet",
                "target": "collect",
                "condition": {"type": "always"},
            },
            {
                "id": "e3",
                "source": "collect",
                "target": "branch",
                "condition": {"type": "always"},
            },
            {
                "id": "e4",
                "source": "branch",
                "target": "transfer",
                "condition": {
                    "type": "intent_match",
                    "pattern": "(?i)speak to (a )?human",
                },
            },
            {
                "id": "e5",
                "source": "branch",
                "target": "end",
                "condition": {"type": "keyword", "keyword": "bye"},
            },
            {
                "id": "e6",
                "source": "branch",
                "target": "end",
                "condition": {"type": "fallback"},
            },
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
        assert body["validationErrors"] == []
        assert body["flowData"]["nodes"]
        assert "greet" in body["flowDataCompiled"]
        assert body["flowDataCompiled"]["greet"]["node"]["type"] == "greeting"

        db.refresh(flow)
        assert flow.flow_data is not None
        assert flow.flow_data_compiled is not None

    def test_missing_start_node_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"][0]["type"] = "greeting"  # no node left marked as start

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "no_start_node" in codes

    def test_multiple_start_nodes_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"][1]["data"]["isStart"] = True

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "multiple_start_nodes" in codes

    def test_cycle_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        # Introduce a cycle: end -> branch (end was previously terminal)
        data["edges"].append(
            {
                "id": "e7",
                "source": "end",
                "target": "branch",
                "condition": {"type": "always"},
            }
        )

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "cycle_detected" in codes

    def test_orphan_node_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        data["nodes"].append(
            {"id": "orphan", "type": "greeting", "data": {"message": "unreachable"}}
        )
        data["edges"].append(
            {
                "id": "e8",
                "source": "orphan",
                "target": "end",
                "condition": {"type": "always"},
            }
        )

        resp = client.put(f"/flows/{flow.id}/flow-data", json={"flowData": data})

        assert resp.status_code == 422
        codes = [e["code"] for e in resp.json()["error"]["validationErrors"]]
        assert "orphan_node" in codes

    def test_missing_outgoing_edge_returns_422(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        data = _valid_flow_data()
        # Drop branch's edges entirely — non-end node with no outgoing edge.
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


# ── GET /flow-data + /flow-data/validate ──────────────────────────────────


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
        data["nodes"][0]["type"] = "greeting"  # break the start-node invariant

        resp = client.request(
            "GET", f"/flows/{flow.id}/flow-data/validate", json={"flowData": data}
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["valid"] is False
        assert any(e["code"] == "no_start_node" for e in body["validationErrors"])

        db.refresh(flow)
        assert flow.flow_data is None  # nothing was persisted

    def test_validate_endpoint_valid_graph(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))

        resp = client.request(
            "GET",
            f"/flows/{flow.id}/flow-data/validate",
            json={"flowData": _valid_flow_data()},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["valid"] is True


# ── Graph compilation: edge priority ordering ─────────────────────────────


class TestCompileGraphPriority:
    def test_edges_sorted_by_condition_specificity(self):
        data = _valid_flow_data()

        compiled = compile_graph(data)

        branch_edges = compiled["branch"]["outgoing_edges"]
        types = [e["condition"]["type"] for e in branch_edges]
        assert types == ["intent_match", "keyword", "fallback"]

    def test_validate_graph_accepts_well_formed_flow(self):
        assert validate_graph(_valid_flow_data()) == []


# ── FlowExecutor: isolated node-executor sequence ─────────────────────────


class TestFlowExecutor:
    def test_full_sequence_greeting_collect_branch_transfer(self):
        compiled = compile_graph(_valid_flow_data())
        executor = FlowExecutor(compiled)
        state = PipelineState(current_node_id=executor.start_node_id())

        # start -> greeting (always edge, no transcript needed)
        assert executor.start_node_id() == "start"
        greet_result = executor.execute_node(
            executor.next_node_id(state.current_node_id, None, state.variables), state
        )
        assert greet_result.node_type == "greeting"
        assert greet_result.action == "speak"
        assert greet_result.speech_text == "Hi there!"

        # greeting -> collect_input (always edge)
        next_id = executor.next_node_id(state.current_node_id, None, state.variables)
        collect_result = executor.execute_node(next_id, state)
        assert collect_result.node_type == "collect_input"
        assert collect_result.action == "wait_for_input"

        # collect_input -> branch (always edge)
        next_id = executor.next_node_id(state.current_node_id, None, state.variables)
        branch_result = executor.execute_node(next_id, state)
        assert branch_result.node_type == "branch"
        assert branch_result.action == "branch"

        # branch -> transfer (transcript matches intent_match over keyword/fallback)
        next_id = executor.next_node_id(
            state.current_node_id,
            "I'd like to speak to a human please",
            state.variables,
        )
        transfer_result = executor.execute_node(next_id, state)
        assert transfer_result.node_type == "transfer"
        assert transfer_result.action == "transfer"

        assert state.history == ["greet", "collect", "branch", "transfer"]

    def test_keyword_condition_matches_exact_word(self):
        compiled = compile_graph(_valid_flow_data())
        executor = FlowExecutor(compiled)

        target = executor.next_node_id("branch", "ok bye then", {})

        assert target == "end"

    def test_fallback_used_when_nothing_else_matches(self):
        compiled = compile_graph(_valid_flow_data())
        executor = FlowExecutor(compiled)

        target = executor.next_node_id("branch", "totally unrelated text", {})

        assert target == "end"

    def test_intent_match_takes_priority_over_keyword(self):
        compiled = compile_graph(_valid_flow_data())
        executor = FlowExecutor(compiled)

        # Contains both an intent phrase and a keyword; intent_match must win.
        target = executor.next_node_id("branch", "speak to a human, then bye", {})

        assert target == "transfer"
