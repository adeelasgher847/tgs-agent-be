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


@pytest.mark.usefixtures("db")
class TestListFolderFlows:
    def test_list_flows_in_folder_returns_linked_flows(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Listing Folder"},
            headers=_headers(auth_tenant),
        ).json()
        linked_flow = _make_flow(db, auth_tenant, test_agent, "Linked Flow")
        unlinked_flow = _make_flow(db, auth_tenant, test_agent, "Unlinked Flow")

        authed_client.post(
            f"/api/v1/folders/{folder['id']}/flows",
            json={"flowId": str(linked_flow.id)},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.get(
            f"/api/v1/folders/{folder['id']}/flows",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["folderId"] == folder["id"]
        assert body["total"] == 1
        flow_ids = {f["id"] for f in body["data"]}
        assert str(linked_flow.id) in flow_ids
        assert str(unlinked_flow.id) not in flow_ids
        # The returned flow should also report the folder it belongs to
        returned = body["data"][0]
        assert folder["id"] in returned["folderIds"]

    def test_list_flows_in_unknown_folder_returns_404(self, authed_client, auth_tenant):
        resp = authed_client.get(
            f"/api/v1/folders/{uuid.uuid4()}/flows",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_list_flows_in_empty_folder_returns_empty(
        self, authed_client, auth_tenant
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Empty Folder"},
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.get(
            f"/api/v1/folders/{folder['id']}/flows",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 0
        assert body["data"] == []


@pytest.mark.usefixtures("db")
class TestRemoveFlowFromFolder:
    def test_remove_flow_deletes_join_row_not_flow(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Removal Folder"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Removable Flow")
        authed_client.post(
            f"/api/v1/folders/{folder['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.delete(
            f"/api/v1/folders/{folder['id']}/flows/{flow.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text

        # Join row is gone
        link = (
            db.query(FolderFlow)
            .filter(
                FolderFlow.folder_id == uuid.UUID(folder["id"]),
                FolderFlow.flow_id == flow.id,
            )
            .first()
        )
        assert link is None

        # The call flow itself is untouched
        flow_row = db.query(CallFlow).filter(CallFlow.id == flow.id).first()
        assert flow_row is not None
        assert flow_row.is_deleted is False

        # No longer listed under this folder
        list_resp = authed_client.get(
            f"/api/v1/folders/{folder['id']}/flows",
            headers=_headers(auth_tenant),
        )
        assert list_resp.json()["total"] == 0

        # Reappears with an empty folderIds list — i.e. "All Files"
        get_resp = authed_client.get(
            f"/api/v1/call-flows/{flow.id}",
            headers=_headers(auth_tenant),
        )
        assert get_resp.json()["folderIds"] == []

    def test_remove_flow_not_in_folder_returns_404(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "No Link Folder"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Never Linked Flow")

        resp = authed_client.delete(
            f"/api/v1/folders/{folder['id']}/flows/{flow.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_remove_flow_from_unknown_folder_returns_404(
        self, authed_client, auth_tenant, test_agent, db
    ):
        flow = _make_flow(db, auth_tenant, test_agent)
        resp = authed_client.delete(
            f"/api/v1/folders/{uuid.uuid4()}/flows/{flow.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_remove_flow_keeps_other_folder_links(
        self, authed_client, auth_tenant, test_agent, db
    ):
        """A flow linked to two folders should only lose the one it's removed from."""
        folder_a = authed_client.post(
            "/api/v1/folders",
            json={"name": "Folder A"},
            headers=_headers(auth_tenant),
        ).json()
        folder_b = authed_client.post(
            "/api/v1/folders",
            json={"name": "Folder B"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Multi-folder Flow")

        for folder in (folder_a, folder_b):
            authed_client.post(
                f"/api/v1/folders/{folder['id']}/flows",
                json={"flowId": str(flow.id)},
                headers=_headers(auth_tenant),
            )

        resp = authed_client.delete(
            f"/api/v1/folders/{folder_a['id']}/flows/{flow.id}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204

        get_resp = authed_client.get(
            f"/api/v1/call-flows/{flow.id}",
            headers=_headers(auth_tenant),
        )
        assert get_resp.json()["folderIds"] == [folder_b["id"]]

    def test_put_call_flow_rejects_folder_id_field(
        self, authed_client, auth_tenant, test_agent
    ):
        """folderId must never be accepted via the call-flow update endpoint —
        folder membership is only managed through the dedicated folder endpoints."""
        created = authed_client.post(
            "/api/v1/call-flows",
            json={
                "name": "My Flow",
                "direction": "inbound",
                "agentId": str(test_agent.id),
            },
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.put(
            f"/api/v1/call-flows/{created['id']}",
            json={"folderId": None},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400


@pytest.mark.usefixtures("db")
class TestMoveFlowToFolder:
    def _link(self, db, folder_id: uuid.UUID, flow_id: uuid.UUID) -> None:
        db.add(FolderFlow(folder_id=folder_id, flow_id=flow_id))
        db.commit()

    def test_move_flow_success(self, authed_client, auth_tenant, test_agent, db):
        folder_a = authed_client.post(
            "/api/v1/folders",
            json={"name": "Move Source"},
            headers=_headers(auth_tenant),
        ).json()
        folder_b = authed_client.post(
            "/api/v1/folders",
            json={"name": "Move Target"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Move Flow")
        authed_client.post(
            f"/api/v1/folders/{folder_a['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.put(
            f"/api/v1/folders/{folder_a['id']}/flows/{flow.id}/move",
            json={"targetFolderId": folder_b["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["flowId"] == str(flow.id)
        assert body["sourceFolderId"] == folder_a["id"]
        assert body["targetFolderId"] == folder_b["id"]

        # No longer in source
        source_flows = authed_client.get(
            f"/api/v1/folders/{folder_a['id']}/flows",
            headers=_headers(auth_tenant),
        ).json()
        assert source_flows["total"] == 0

        # Now in target
        target_flows = authed_client.get(
            f"/api/v1/folders/{folder_b['id']}/flows",
            headers=_headers(auth_tenant),
        ).json()
        assert target_flows["total"] == 1
        assert target_flows["data"][0]["id"] == str(flow.id)

    def test_move_flow_same_folder_returns_400(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Same Folder"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Same Folder Flow")
        authed_client.post(
            f"/api/v1/folders/{folder['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.put(
            f"/api/v1/folders/{folder['id']}/flows/{flow.id}/move",
            json={"targetFolderId": folder["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_move_flow_unknown_source_folder_returns_404(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder_b = authed_client.post(
            "/api/v1/folders",
            json={"name": "Target Only"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Orphan Flow")

        resp = authed_client.put(
            f"/api/v1/folders/{uuid.uuid4()}/flows/{flow.id}/move",
            json={"targetFolderId": folder_b["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_move_flow_unknown_target_folder_returns_404(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder_a = authed_client.post(
            "/api/v1/folders",
            json={"name": "Source Only"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Source Flow")
        authed_client.post(
            f"/api/v1/folders/{folder_a['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.put(
            f"/api/v1/folders/{folder_a['id']}/flows/{flow.id}/move",
            json={"targetFolderId": str(uuid.uuid4())},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_move_unknown_flow_returns_404(self, authed_client, auth_tenant):
        folder_a = authed_client.post(
            "/api/v1/folders",
            json={"name": "Move Src Unknown Flow"},
            headers=_headers(auth_tenant),
        ).json()
        folder_b = authed_client.post(
            "/api/v1/folders",
            json={"name": "Move Tgt Unknown Flow"},
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.put(
            f"/api/v1/folders/{folder_a['id']}/flows/{uuid.uuid4()}/move",
            json={"targetFolderId": folder_b["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_move_flow_not_in_source_folder_returns_404(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder_a = authed_client.post(
            "/api/v1/folders",
            json={"name": "Not Linked Source"},
            headers=_headers(auth_tenant),
        ).json()
        folder_b = authed_client.post(
            "/api/v1/folders",
            json={"name": "Not Linked Target"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Not Linked Flow")

        resp = authed_client.put(
            f"/api/v1/folders/{folder_a['id']}/flows/{flow.id}/move",
            json={"targetFolderId": folder_b["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_move_flow_already_in_target_dedupes(
        self, authed_client, auth_tenant, test_agent, db
    ):
        folder_a = authed_client.post(
            "/api/v1/folders",
            json={"name": "Dedup Source"},
            headers=_headers(auth_tenant),
        ).json()
        folder_b = authed_client.post(
            "/api/v1/folders",
            json={"name": "Dedup Target"},
            headers=_headers(auth_tenant),
        ).json()
        flow = _make_flow(db, auth_tenant, test_agent, "Dedup Flow")

        # Flow already linked to both source and target
        authed_client.post(
            f"/api/v1/folders/{folder_a['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )
        authed_client.post(
            f"/api/v1/folders/{folder_b['id']}/flows",
            json={"flowId": str(flow.id)},
            headers=_headers(auth_tenant),
        )

        resp = authed_client.put(
            f"/api/v1/folders/{folder_a['id']}/flows/{flow.id}/move",
            json={"targetFolderId": folder_b["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text

        # No duplicate row in target, and source link removed
        target_links = (
            db.query(FolderFlow)
            .filter(
                FolderFlow.folder_id == uuid.UUID(folder_b["id"]),
                FolderFlow.flow_id == flow.id,
            )
            .count()
        )
        assert target_links == 1

        source_link = (
            db.query(FolderFlow)
            .filter(
                FolderFlow.folder_id == uuid.UUID(folder_a["id"]),
                FolderFlow.flow_id == flow.id,
            )
            .first()
        )
        assert source_link is None

        target_flows = authed_client.get(
            f"/api/v1/folders/{folder_b['id']}/flows",
            headers=_headers(auth_tenant),
        ).json()
        assert target_flows["total"] == 1

    def test_move_flow_cross_tenant_returns_404(
        self, authed_client, auth_tenant, test_agent, db
    ):
        """A tenant must not be able to move a flow using folder/flow ids
        belonging to a different tenant — every combination should 404."""
        other_tenant = Tenant(
            name=f"OtherWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        other_agent = Agent(
            tenant_id=other_tenant.id,
            name="Other Tenant Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-y",
            tts_language="en",
        )
        db.add(other_agent)
        db.commit()
        db.refresh(other_agent)

        # Folders/flow owned by the "other" tenant, created directly via DB
        # (no authed client available for them in this test).
        other_folder_a = Folder(tenant_id=other_tenant.id, name="Other Source")
        other_folder_b = Folder(tenant_id=other_tenant.id, name="Other Target")
        db.add_all([other_folder_a, other_folder_b])
        db.commit()
        db.refresh(other_folder_a)
        db.refresh(other_folder_b)

        other_flow = _make_flow(db, other_tenant, other_agent, "Other Tenant Flow")
        self._link(db, other_folder_a.id, other_flow.id)

        # Our own tenant's folders/flow for cross-combinations
        own_folder = authed_client.post(
            "/api/v1/folders",
            json={"name": "Own Folder"},
            headers=_headers(auth_tenant),
        ).json()
        own_folder_2 = authed_client.post(
            "/api/v1/folders",
            json={"name": "Own Folder Target"},
            headers=_headers(auth_tenant),
        ).json()
        own_flow = _make_flow(db, auth_tenant, test_agent, "Own Flow")
        authed_client.post(
            f"/api/v1/folders/{own_folder['id']}/flows",
            json={"flowId": str(own_flow.id)},
            headers=_headers(auth_tenant),
        )

        # 1. Source folder belongs to other tenant -> 404
        resp = authed_client.put(
            f"/api/v1/folders/{other_folder_a.id}/flows/{own_flow.id}/move",
            json={"targetFolderId": own_folder["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

        # 2. Target folder belongs to other tenant -> 404
        resp = authed_client.put(
            f"/api/v1/folders/{own_folder['id']}/flows/{own_flow.id}/move",
            json={"targetFolderId": str(other_folder_b.id)},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

        # 3. Flow belongs to other tenant -> 404
        resp = authed_client.put(
            f"/api/v1/folders/{own_folder['id']}/flows/{other_flow.id}/move",
            json={"targetFolderId": own_folder_2["id"]},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

        # Sanity: the other tenant's flow is still linked to its own folder,
        # untouched by any of the failed attempts above.
        still_linked = (
            db.query(FolderFlow)
            .filter(
                FolderFlow.folder_id == other_folder_a.id,
                FolderFlow.flow_id == other_flow.id,
            )
            .first()
        )
        assert still_linked is not None
