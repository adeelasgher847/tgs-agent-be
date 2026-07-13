"""Integration tests for /api/v1/folders.

Mirrors the auth-mocking pattern from tests/api/test_agents.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.folder import Folder
from app.models.folder_flow import FolderFlow
from app.models.tenant import Tenant

_API_KEY = "test-folders-key"


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
        name=f"FolderWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"folder_ws_{uuid.uuid4().hex[:8]}",
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
        name="Folder Test Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-y",
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


def _make_flow(db, tenant: Tenant, agent: Agent, name: str = "Test Flow") -> CallFlow:
    flow = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name=name,
        direction="inbound",
    )
    db.add(flow)
    db.commit()
    db.refresh(flow)
    return flow


# ──────────────────────────────────────────────────────────────── tests ──


@pytest.mark.usefixtures("db")
class TestFolderCRUD:
    def test_create_folder_returns_201(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/folders",
            json={"name": "Inbound Flows"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "Inbound Flows"
        assert "id" in body
        assert body["isDeleted"] is False

    def test_list_folders(self, authed_client, auth_tenant):
        authed_client.post(
            "/api/v1/folders",
            json={"name": "List Folder A"},
            headers=_headers(auth_tenant),
        )
        authed_client.post(
            "/api/v1/folders",
            json={"name": "List Folder B"},
            headers=_headers(auth_tenant),
        )
        resp = authed_client.get("/api/v1/folders", headers=_headers(auth_tenant))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "data" in body
        assert body["total"] >= 2
        names = {f["name"] for f in body["data"]}
        assert "List Folder A" in names
        assert "List Folder B" in names

    def test_patch_rename_folder(self, authed_client, auth_tenant):
        created = authed_client.post(
            "/api/v1/folders",
            json={"name": "Old Name"},
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.patch(
            f"/api/v1/folders/{created['id']}",
            json={"name": "New Name"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "New Name"

    def test_delete_folder_soft_deletes_only_folder(
        self, authed_client, auth_tenant, test_agent, db
    ):
        created = authed_client.post(
            "/api/v1/folders",
            json={"name": "To Delete"},
            headers=_headers(auth_tenant),
        ).json()
        folder_id = uuid.UUID(created["id"])

        # Add a flow to the folder
        flow = _make_flow(db, auth_tenant, test_agent, "Preserved Flow")
        authed_client.post(
            f"/api/v1/folders/{folder_id}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )

        # Delete the folder
        resp = authed_client.delete(
            f"/api/v1/folders/{folder_id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text

        # Folder is soft-deleted
        folder_row = db.query(Folder).filter(Folder.id == folder_id).first()
        assert folder_row is not None
        assert folder_row.is_deleted is True

        # Flow is NOT deleted
        flow_row = db.query(CallFlow).filter(CallFlow.id == flow.id).first()
        assert flow_row is not None
        assert flow_row.is_deleted is False

    def test_delete_unknown_folder_returns_404(self, authed_client, auth_tenant):
        resp = authed_client.delete(
            f"/api/v1/folders/{uuid.uuid4()}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404


@pytest.mark.usefixtures("db")
class TestAddFlowToFolder:
    def test_add_flow_to_folder_success(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "My Folder"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent)

        resp = authed_client.post(
            f"/api/v1/folders/{folder['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["folderId"] == folder["id"]
        assert body["flowId"] == str(flow.id)

        # DB row exists
        link = (
            db.query(FolderFlow)
            .filter(
                FolderFlow.folder_id == uuid.UUID(folder["id"]),
                FolderFlow.flow_id == flow.id,
            )
            .first()
        )
        assert link is not None

    def test_add_flow_to_folder_idempotent(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Idem Folder"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Idem Flow")

        for _ in range(2):
            resp = authed_client.post(
                f"/api/v1/folders/{folder['id']}/flows",
                json={"flowId": str(flow.id)},
                headers=_headers(auth_tenant),
            )
            assert resp.status_code == 200

        count = (
            db.query(FolderFlow)
            .filter(
                FolderFlow.folder_id == uuid.UUID(folder["id"]),
                FolderFlow.flow_id == flow.id,
            )
            .count()
        )
        assert count == 1

    def test_add_unknown_flow_returns_404(self, authed_client, auth_tenant):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Folder X"},
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.post(
            f"/api/v1/folders/{folder['id']}/flows",
            json={"flowId": str(uuid.uuid4())},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404
