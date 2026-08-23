"""API tests for Visual Flow Editor: POST validate endpoint and GET flow-data?mode=readonly."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


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
        name=f"FlowDataV2WS-{uuid.uuid4().hex[:8]}",
        schema_name=f"flow_data_v2_ws_{uuid.uuid4().hex[:8]}",
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
        name="Flow Data Test Agent",
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
        name="Flow Data Test Flow",
        direction="inbound",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _full_sensitive_flow_data() -> dict:
    return {
        "nodes": [
            {
                "id": "node_greet",
                "type": "greeting",
                "position": {"x": 100, "y": 100},
                "data": {
                    "label": "Welcome Node",
                    "message": "Welcome to secret service {customer_name}",
                },
            },
            {
                "id": "node_collect",
                "type": "collect_input",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Collect SSN",
                    "prompt": "Please enter your secret pin",
                    "variable_name": "secret_pin",
                    "validation_regex": r"^\d{4}$",
                    "fallback_message": "Invalid pin format",
                },
            },
            {
                "id": "node_transfer",
                "type": "transfer",
                "position": {"x": 100, "y": 300},
                "data": {
                    "label": "Transfer to Agent",
                    "transfer_to": "+15551234567",
                    "agent_message": "Connecting you now",
                    "whisper_message": "Secret VIP caller",
                },
            },
            {
                "id": "node_end",
                "type": "end_call",
                "position": {"x": 100, "y": 400},
                "data": {
                    "label": "Hangup",
                    "closing_message": "Goodbye",
                    "post_call_webhook_override": "https://secret.webhook.org/endpoint",
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "node_greet", "target": "node_collect", "condition": {"type": "always"}},
            {"id": "e2", "source": "node_collect", "target": "node_transfer", "condition": {"type": "always"}},
            {"id": "e3", "source": "node_transfer", "target": "node_end", "condition": {"type": "always"}},
        ],
        "entry_node_id": "node_greet",
        "version": 1,
        "compiled_at": "2026-08-23T12:00:00Z",
    }


class TestValidatePostEndpoint:
    def test_post_validate_valid_graph(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        resp = client.post(
            f"/flows/{flow.id}/validate",
            json={"flowData": _full_sensitive_flow_data()},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["valid"] is True
        assert body["errors"] == []

    def test_post_validate_invalid_graph_returns_errors(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        invalid_data = _full_sensitive_flow_data()
        # Break the entry-node invariant: greeting node retyped away
        invalid_data["nodes"][0]["type"] = "collect_input"

        resp = client.post(
            f"/flows/{flow.id}/validate",
            json={"flowData": invalid_data},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["valid"] is False
        assert any(e["node_id"] is None for e in body["errors"])
        assert any("greeting" in e["message"] for e in body["errors"])

    def test_post_validate_saved_graph_when_body_omitted(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        # Save flow first
        client.put(
            f"/flows/{flow.id}/flow-data",
            json={"flowData": _full_sensitive_flow_data()},
        )

        # Validate with empty body
        resp = client.post(f"/flows/{flow.id}/validate")
        assert resp.status_code == 200, resp.text
        assert resp.json()["valid"] is True

    def test_get_validate_returns_405_method_not_allowed(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        resp = client.get(f"/flows/{flow.id}/validate")
        assert resp.status_code == 405


class TestGetFlowDataReadonlyMode:
    def test_readonly_mode_strips_sensitive_fields_and_nulls_compiled(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        # Save full flow
        client.put(
            f"/flows/{flow.id}/flow-data",
            json={"flowData": _full_sensitive_flow_data()},
        )

        # Request readonly mode
        resp = client.get(f"/flows/{flow.id}/flow-data?mode=readonly")
        assert resp.status_code == 200, resp.text
        body = resp.json()

        # flowDataCompiled must be null in readonly mode
        assert body["flowDataCompiled"] is None

        nodes = body["flowData"]["nodes"]
        assert len(nodes) == 4

        # Verify structural fields are preserved
        assert nodes[0]["id"] == "node_greet"
        assert nodes[0]["type"] == "greeting"
        assert nodes[0]["position"] == {"x": 100, "y": 100}
        assert nodes[0]["data"]["label"] == "Welcome Node"

        # Verify none of the 8 sensitive fields are present in any node's data
        sensitive_fields = {
            "message",
            "prompt",
            "transfer_to",
            "whisper_message",
            "agent_message",
            "post_call_webhook_override",
            "validation_regex",
            "fallback_message",
        }
        for node in nodes:
            node_data = node.get("data", {})
            for sensitive_key in sensitive_fields:
                assert sensitive_key not in node_data, f"Field '{sensitive_key}' leaked in node '{node['id']}'"

        # Verify edges and top-level fields are intact
        assert len(body["flowData"]["edges"]) == 3
        assert body["flowData"]["entry_node_id"] == "node_greet"
        assert body["flowData"]["version"] == 1

    def test_readonly_mode_on_empty_flow_returns_null(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        resp = client.get(f"/flows/{flow.id}/flow-data?mode=readonly")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["flowData"] is None
        assert body["flowDataCompiled"] is None
        assert body["validationErrors"] == []

    def test_unrecognized_mode_falls_through_to_normal(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        client.put(
            f"/flows/{flow.id}/flow-data",
            json={"flowData": _full_sensitive_flow_data()},
        )

        resp = client.get(f"/flows/{flow.id}/flow-data?mode=preview")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Full payload returned
        assert body["flowDataCompiled"] is not None
        assert "message" in body["flowData"]["nodes"][0]["data"]

    def test_default_mode_returns_full_payload(self, db, workspace, flow):
        client = _build_app(db, _principal(workspace.id))
        client.put(
            f"/flows/{flow.id}/flow-data",
            json={"flowData": _full_sensitive_flow_data()},
        )

        resp = client.get(f"/flows/{flow.id}/flow-data")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["flowDataCompiled"] is not None
        assert body["flowData"]["nodes"][0]["data"]["message"] == "Welcome to secret service {customer_name}"
