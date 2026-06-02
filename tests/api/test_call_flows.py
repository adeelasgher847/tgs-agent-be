"""Integration tests for /api/v1/call-flows.

Mirrors the auth-mocking pattern from tests/api/test_agents.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.workspace import Workspace
from app.middleware.api_key_middleware import _attach_workspace_context
from app.core.request_auth import AUTH_METHOD_JWT
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.prompt_version import PromptVersion
from app.models.tenant import Tenant

_API_KEY = "test-callflows-key"


# ─────────────────────────────────────────────────────────────── helpers ──


def _payload_for(tenant: Tenant) -> dict:
    return {
        "api_key_id": str(uuid.uuid4()),
        "tenant_id": str(tenant.id),
        "key_is_active": True,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": tenant.schema_name,
            "status": "active",
            "credits": 0.0,
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
        },
    }


def _headers(tenant: Tenant) -> dict:
    return {"x-api-key": _API_KEY, "x-workspace-id": str(tenant.id)}


# ─────────────────────────────────────────────────────────────── fixtures ──


@pytest.fixture
def auth_tenant(db) -> Tenant:
    t = Tenant(
        name=f"FlowWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"flow_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def test_agent(db, auth_tenant: Tenant) -> Agent:
    a = Agent(
        tenant_id=auth_tenant.id,
        name="Flow Test Agent",
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
def authed_client(client: TestClient, auth_tenant: Tenant):
    payload = _payload_for(auth_tenant)

    async def _resolve(_key_hash, _workspace_id):
        return payload

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield client


def _create_body(agent_id: uuid.UUID, **overrides) -> dict:
    body = {
        "name": "My Flow",
        "direction": "inbound",
        "agentId": str(agent_id),
    }
    body.update(overrides)
    return body


# ──────────────────────────────────────────────────────────────── tests ──


@pytest.mark.usefixtures("db")
class TestCreateCallFlow:
    def test_create_with_prompt_returns_201(self, authed_client, auth_tenant, test_agent, db):
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(
                test_agent.id,
                prompt="You are a helpful assistant.",
            ),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "My Flow"
        assert body["direction"] == "inbound"
        assert body["agentId"] == str(test_agent.id)
        assert body["currentPromptId"] is not None
        assert len(body["promptVersions"]) == 1
        pv = body["promptVersions"][0]
        assert pv["promptText"] == "You are a helpful assistant."
        assert pv["flowId"] == body["id"]
        assert "createdAt" in pv
        # gemini_prompt must never appear
        assert "geminiPrompt" not in pv
        assert "gemini_prompt" not in pv

    def test_create_without_prompt_has_empty_versions(
        self, authed_client, auth_tenant, test_agent
    ):
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["currentPromptId"] is None
        assert body["promptVersions"] == []

    def test_create_missing_required_fields_returns_400(
        self, authed_client, auth_tenant
    ):
        # Global exception handler converts Pydantic validation errors to 400
        resp = authed_client.post(
            "/api/v1/call-flows",
            json={"name": "Bad Flow"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_unknown_agent_returns_404(
        self, authed_client, auth_tenant
    ):
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(uuid.uuid4()),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_create_returns_agent_embedded(
        self, authed_client, auth_tenant, test_agent
    ):
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["agent"]["id"] == str(test_agent.id)
        assert body["agent"]["name"] == test_agent.name


@pytest.mark.usefixtures("db")
class TestUpdateCallFlow:
    def _create_flow(self, client, tenant, agent) -> dict:
        resp = client.post(
            "/api/v1/call-flows",
            json=_create_body(agent.id, prompt="v1 prompt"),
            headers=_headers(tenant),
        )
        assert resp.status_code == 201
        return resp.json()

    def test_update_prompt_creates_new_version(
        self, authed_client, auth_tenant, test_agent
    ):
        created = self._create_flow(authed_client, auth_tenant, test_agent)
        v1_id = created["currentPromptId"]

        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"prompt": "v2 prompt"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["promptVersions"]) == 2
        assert body["currentPromptId"] != v1_id
        # Newest version is first
        assert body["promptVersions"][0]["promptText"] == "v2 prompt"

    def test_update_same_prompt_does_not_create_version(
        self, authed_client, auth_tenant, test_agent
    ):
        created = self._create_flow(authed_client, auth_tenant, test_agent)
        v1_id = created["currentPromptId"]

        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"prompt": "v1 prompt"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200
        body = resp.json()
        # Still only 1 version; no new row created
        assert len(body["promptVersions"]) == 1
        assert body["currentPromptId"] == v1_id

    def test_rollback_via_current_prompt_id(
        self, authed_client, auth_tenant, test_agent
    ):
        created = self._create_flow(authed_client, auth_tenant, test_agent)
        v1_id = created["currentPromptId"]

        # Create v2
        authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"prompt": "v2 prompt"},
            headers=_headers(auth_tenant),
        )

        # Rollback to v1
        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"currentPromptId": v1_id},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["currentPromptId"] == v1_id
        # Still 2 versions (no new version created)
        assert len(body["promptVersions"]) == 2

    def test_rollback_with_wrong_flow_version_returns_400(
        self, authed_client, auth_tenant, test_agent, db
    ):
        created = self._create_flow(authed_client, auth_tenant, test_agent)
        # Create a second flow and get its version id
        other = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, prompt="other flow prompt"),
            headers=_headers(auth_tenant),
        ).json()
        other_version_id = other["currentPromptId"]

        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"currentPromptId": other_version_id},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_update_name_direction(
        self, authed_client, auth_tenant, test_agent
    ):
        created = self._create_flow(authed_client, auth_tenant, test_agent)
        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"name": "Renamed Flow", "direction": "outbound"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Renamed Flow"
        assert body["direction"] == "outbound"


@pytest.mark.usefixtures("db")
class TestGetCallFlow:
    def test_get_returns_full_shape(self, authed_client, auth_tenant, test_agent):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, prompt="hello"),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.get(
            f"/api/v1/call-flows/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == created["id"]
        assert len(body["promptVersions"]) == 1
        assert body["agent"]["id"] == str(test_agent.id)

    def test_get_unknown_returns_404(self, authed_client, auth_tenant):
        resp = authed_client.get(
            f"/api/v1/call-flows/{uuid.uuid4()}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404


@pytest.mark.usefixtures("db")
class TestListCallFlows:
    def test_list_paginated_with_agent_name(
        self, authed_client, auth_tenant, test_agent
    ):
        for i in range(3):
            authed_client.post(
                "/api/v1/call-flows",
                json=_create_body(test_agent.id, name=f"Flow {i}"),
                headers=_headers(auth_tenant),
            )

        resp = authed_client.get(
            "/api/v1/call-flows?pageSize=2",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["pageSize"] == 2
        assert len(body["data"]) == 2
        assert body["total"] >= 3
        assert body["page"] == 1
        for item in body["data"]:
            assert "agent" in item
            assert item["agent"]["name"] == test_agent.name


@pytest.mark.usefixtures("db")
class TestPromptVersions:
    def test_get_versions_newest_first(
        self, authed_client, auth_tenant, test_agent
    ):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, prompt="first"),
            headers=_headers(auth_tenant),
        ).json()
        authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"prompt": "second"},
            headers=_headers(auth_tenant),
        )
        authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"prompt": "third"},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.get(
            f"/api/v1/call-flows/{created['id']}/prompt-versions",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        versions = resp.json()
        assert len(versions) == 3
        # Newest first
        assert versions[0]["promptText"] == "third"
        assert versions[2]["promptText"] == "first"
        # gemini_prompt must not appear
        for v in versions:
            assert "geminiPrompt" not in v
            assert "gemini_prompt" not in v


@pytest.mark.usefixtures("db")
class TestDeleteCallFlow:
    def test_delete_without_active_call_returns_204(
        self, authed_client, auth_tenant, test_agent, db
    ):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.delete(
            f"/api/v1/call-flows/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text

        # Soft deleted — GET returns 404
        get_resp = authed_client.get(
            f"/api/v1/call-flows/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert get_resp.status_code == 404

        row = db.query(CallFlow).filter(CallFlow.id == uuid.UUID(created["id"])).first()
        assert row is not None
        assert row.is_deleted is True

    def test_delete_with_active_call_returns_409(
        self, authed_client, auth_tenant, test_agent, db
    ):
        from app.models.user import User

        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        ).json()
        flow_id = uuid.UUID(created["id"])

        # Seed a minimal User so CallSession FK is satisfied
        u = User(
            email=f"cs-user-{uuid.uuid4().hex[:6]}@example.com",
            first_name="CS",
            last_name="User",
            hashed_password="",
            current_tenant_id=auth_tenant.id,
        )
        db.add(u)
        db.flush()

        session = CallSession(
            user_id=u.id,
            agent_id=test_agent.id,
            tenant_id=auth_tenant.id,
            call_flow_id=flow_id,
            status="active",
            start_time=__import__("datetime").datetime.utcnow(),
        )
        db.add(session)
        db.commit()

        resp = authed_client.delete(
            f"/api/v1/call-flows/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["code"] == "flow_has_active_calls"


@pytest.mark.usefixtures("db")
class TestVersionCap:
    def test_51st_version_prunes_to_50(
        self, authed_client, auth_tenant, test_agent, db
    ):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, prompt="version 0"),
            headers=_headers(auth_tenant),
        ).json()
        flow_id = created["id"]

        # Create 50 more versions (total 51)
        for i in range(1, 51):
            resp = authed_client.put(
                f"/api/v1/call-flows/{flow_id}",
                json={"prompt": f"version {i}"},
                headers=_headers(auth_tenant),
            )
            assert resp.status_code == 200, resp.text

        versions = db.query(PromptVersion).filter(
            PromptVersion.flow_id == uuid.UUID(flow_id)
        ).all()
        assert len(versions) == 50


@pytest.mark.usefixtures("db")
class TestFullAgentEmbed:
    def test_get_returns_full_agent_embed(self, authed_client, auth_tenant, test_agent):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, prompt="hello"),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.get(
            f"/api/v1/call-flows/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        agent = resp.json()["agent"]
        # Full AgentOut shape — not just {id, name}
        assert "llmModel" in agent
        assert "status" in agent
        assert "createdAt" in agent
        assert agent["id"] == str(test_agent.id)

    def test_post_201_returns_full_agent_embed(self, authed_client, auth_tenant, test_agent):
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201
        agent = resp.json()["agent"]
        assert "llmModel" in agent
        assert "status" in agent

    def test_put_returns_full_agent_embed(self, authed_client, auth_tenant, test_agent):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"name": "Updated Name"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200
        agent = resp.json()["agent"]
        assert "llmModel" in agent

    def test_list_agent_is_slim(self, authed_client, auth_tenant, test_agent):
        authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        )
        resp = authed_client.get("/api/v1/call-flows", headers=_headers(auth_tenant))
        assert resp.status_code == 200
        agent = resp.json()["data"][0]["agent"]
        # List keeps slim {id, name} only
        assert "id" in agent
        assert "name" in agent
        assert "llmModel" not in agent
        assert "status" not in agent


@pytest.mark.usefixtures("db")
class TestFlowData:
    def test_create_with_flow_data(self, authed_client, auth_tenant, test_agent):
        flow_data = {"nodes": [{"id": "1", "type": "start"}], "edges": []}
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, flowData=flow_data),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["flowData"]["nodes"] == [{"id": "1", "type": "start"}]
        assert body["flowData"]["edges"] == []

    def test_update_flow_data(self, authed_client, auth_tenant, test_agent):
        created = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id),
            headers=_headers(auth_tenant),
        ).json()

        new_flow_data = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"source": "n1", "target": "n2"}],
        }
        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"flowData": new_flow_data},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["flowData"]["nodes"]) == 2
        assert len(body["flowData"]["edges"]) == 1

    def test_invalid_flow_data_missing_nodes_returns_400(
        self, authed_client, auth_tenant, test_agent
    ):
        # nodes has a default value so missing it alone is valid.
        # A non-list value for nodes should be rejected.
        resp = authed_client.post(
            "/api/v1/call-flows",
            json=_create_body(test_agent.id, flowData={"nodes": "not-a-list", "edges": []}),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400


@pytest.mark.usefixtures("db")
class TestVersionCapProtection:
    def _create_flow_with_prompt(self, client, tenant, agent, prompt="v0") -> dict:
        resp = client.post(
            "/api/v1/call-flows",
            json=_create_body(agent.id, prompt=prompt),
            headers=_headers(tenant),
        )
        assert resp.status_code == 201
        return resp.json()

    def test_cap_does_not_delete_current_prompt_version(
        self, authed_client, auth_tenant, test_agent, db
    ):
        """After rollback to v0 the pruner must not delete v0 when cap triggers."""
        created = self._create_flow_with_prompt(authed_client, auth_tenant, test_agent, "v0")
        flow_id = created["id"]
        v0_id = created["currentPromptId"]

        # Add 49 more versions so count = 50
        for i in range(1, 50):
            r = authed_client.put(
                f"/api/v1/call-flows/{flow_id}",
                json={"prompt": f"unique version {i} - {uuid.uuid4().hex}"},
                headers=_headers(auth_tenant),
            )
            assert r.status_code == 200

        # Rollback to v0 — now current = v0
        r = authed_client.put(
            f"/api/v1/call-flows/{flow_id}",
            json={"currentPromptId": v0_id},
            headers=_headers(auth_tenant),
        )
        assert r.status_code == 200
        assert r.json()["currentPromptId"] == v0_id

        # Add one more prompt → triggers pruning (count goes 50→51→prune→50)
        # Pruner must protect v0 (current) and delete the next-oldest instead
        r = authed_client.put(
            f"/api/v1/call-flows/{flow_id}",
            json={"prompt": f"trigger prune {uuid.uuid4().hex}"},
            headers=_headers(auth_tenant),
        )
        assert r.status_code == 200

        # v0 must still exist in DB
        v0_row = db.query(PromptVersion).filter(PromptVersion.id == uuid.UUID(v0_id)).first()
        assert v0_row is not None, "v0 was wrongly deleted by the pruner"

        # Total versions still capped at 50
        count = db.query(PromptVersion).filter(
            PromptVersion.flow_id == uuid.UUID(flow_id)
        ).count()
        assert count == 50

        # GET flow lists v0 in promptVersions
        versions = authed_client.get(
            f"/api/v1/call-flows/{flow_id}/prompt-versions",
            headers=_headers(auth_tenant),
        ).json()
        version_ids = {v["id"] for v in versions}
        assert v0_id in version_ids

    def test_prune_deletes_oldest_non_current(
        self, authed_client, auth_tenant, test_agent, db
    ):
        """When cap triggers normally, the true oldest (v0) is pruned."""
        created = self._create_flow_with_prompt(authed_client, auth_tenant, test_agent, "baseline")
        flow_id = created["id"]
        v0_id = created["currentPromptId"]

        # Add 50 more unique versions; on the 51st current is v49, so v0 is pruned
        for i in range(1, 51):
            r = authed_client.put(
                f"/api/v1/call-flows/{flow_id}",
                json={"prompt": f"prune-test version {i} - {uuid.uuid4().hex}"},
                headers=_headers(auth_tenant),
            )
            assert r.status_code == 200

        count = db.query(PromptVersion).filter(
            PromptVersion.flow_id == uuid.UUID(flow_id)
        ).count()
        assert count == 50

        # v0 (oldest, was not current when 51st was added) must be gone
        v0_row = db.query(PromptVersion).filter(PromptVersion.id == uuid.UUID(v0_id)).first()
        assert v0_row is None, "oldest non-current version was not pruned"
