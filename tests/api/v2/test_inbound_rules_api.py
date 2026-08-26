"""Tests for Inbound Rules & Blocklist Rule Sets API and Flow Assignment endpoints.

Coverage:
  - CRUD for /api/v2/inbound-rules/sets
  - Bulk import via /api/v2/inbound-rules/sets/import
  - Flow assignment /api/v2/flows/{flow_id}/inbound-rules (PUT/GET)
  - RBAC (Admin required for mutations, Read-Only sufficient for GET)
  - Tenant isolation (404 for other workspace resources)
  - Audit logging events
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal, *, forbidden=False):
    from app.api.deps import (
        get_db,
        require_admin_or_api_key,
        require_readonly_or_api_key,
    )
    from app.api.v2.routers.flows import router as flows_router
    from app.api.v2.routers.inbound_rules import router as inbound_rules_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(flows_router)
    mini.include_router(inbound_rules_router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )

        mini.dependency_overrides[require_admin_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal
        mini.dependency_overrides[require_readonly_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


def _build_readonly_app(db_override, principal, *, forbidden=False):
    from app.api.deps import get_db, require_readonly_or_api_key
    from app.api.v2.routers.flows import router as flows_router
    from app.api.v2.routers.inbound_rules import router as inbound_rules_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(flows_router)
    mini.include_router(inbound_rules_router)

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
    from app.api.v2.routers.flows import router as flows_router
    from app.api.v2.routers.inbound_rules import router as inbound_rules_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(flows_router)
    mini.include_router(inbound_rules_router)

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
        name=f"InboundRulesWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"inbound_rules_ws_{uuid.uuid4().hex[:8]}",
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
        name=f"OtherInboundRulesWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_inbound_rules_ws_{uuid.uuid4().hex[:8]}",
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
        name="Inbound Rules Agent",
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
        name="Inbound Rules Flow",
        direction="inbound",
        status="active",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestInboundRuleSetsCRUD:
    def test_create_rule_set_with_rules(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "name": "Global Robocallers",
            "description": "Blocklist of known spam numbers",
            "rules": [
                {
                    "phone_number_pattern": "+1 (555) 987-6543",
                    "label": "Telemarketer",
                },
                {
                    "phone_number_pattern": "+15551234567",
                    "label": "Scam Caller",
                },
            ],
        }

        resp = client.post("/inbound-rules/sets", json=payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["name"] == "Global Robocallers"
        assert data["description"] == "Blocklist of known spam numbers"
        assert data["rules_count"] == 2
        assert len(data["rules"]) == 2
        assert any(r["normalized_digits"] == "15559876543" for r in data["rules"])
        assert any(r["normalized_digits"] == "15551234567" for r in data["rules"])

    def test_list_rule_sets(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        # Create two sets
        admin_client = _build_app(db, principal)
        admin_client.post(
            "/inbound-rules/sets",
            json={"name": "Set A", "rules": [{"phone_number_pattern": "5551112222"}]},
        )
        admin_client.post(
            "/inbound-rules/sets",
            json={"name": "Set B", "rules": []},
        )

        resp = client.get("/inbound-rules/sets")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 2
        set_a = next(s for s in items if s["name"] == "Set A")
        assert set_a["rules_count"] == 1

    def test_get_single_rule_set(self, db, workspace):
        principal = _principal(workspace.id)
        admin_client = _build_app(db, principal)
        create_resp = admin_client.post(
            "/inbound-rules/sets",
            json={"name": "Set Detail", "description": "Desc", "rules": [{"phone_number_pattern": "5553334444", "label": "Tag"}]},
        )
        set_id = create_resp.json()["id"]

        readonly_client = _build_readonly_app(db, principal)
        resp = readonly_client.get(f"/inbound-rules/sets/{set_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == set_id
        assert body["name"] == "Set Detail"
        assert len(body["rules"]) == 1
        assert body["rules"][0]["label"] == "Tag"

    def test_update_rule_set_and_replace_rules(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        create_resp = client.post(
            "/inbound-rules/sets",
            json={"name": "Initial Set", "rules": [{"phone_number_pattern": "5551111111"}]},
        )
        set_id = create_resp.json()["id"]

        update_resp = client.put(
            f"/inbound-rules/sets/{set_id}",
            json={
                "name": "Updated Set Name",
                "rules": [
                    {"phone_number_pattern": "5552222222", "label": "New Rule 1"},
                    {"phone_number_pattern": "5553333333", "label": "New Rule 2"},
                ],
            },
        )
        assert update_resp.status_code == 200
        body = update_resp.json()
        assert body["name"] == "Updated Set Name"
        assert body["rules_count"] == 2

    def test_delete_rule_set_detaches_flow(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        create_resp = client.post(
            "/inbound-rules/sets",
            json={"name": "Set to Delete", "rules": [{"phone_number_pattern": "5559999999"}]},
        )
        set_id = create_resp.json()["id"]

        # Attach to flow
        client.put(f"/flows/{flow.id}/inbound-rules", json={"inbound_rule_set_id": set_id})

        # Delete set
        del_resp = client.delete(f"/inbound-rules/sets/{set_id}")
        assert del_resp.status_code == 204

        # Flow should be detached
        flow_rules_resp = client.get(f"/flows/{flow.id}/inbound-rules")
        assert flow_rules_resp.status_code == 200
        assert flow_rules_resp.json()["inbound_rule_set_id"] is None

        # Getting deleted set should return 404
        get_resp = client.get(f"/inbound-rules/sets/{set_id}")
        assert get_resp.status_code == 404

    def test_tenant_isolation_rule_sets(self, db, other_workspace, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)
        create_resp = client.post("/inbound-rules/sets", json={"name": "Tenant A Set"})
        set_id = create_resp.json()["id"]

        foreign_principal = _principal(other_workspace.id)
        foreign_client = _build_app(db, foreign_principal)

        assert foreign_client.get(f"/inbound-rules/sets/{set_id}").status_code == 404
        assert foreign_client.put(f"/inbound-rules/sets/{set_id}", json={"name": "Hacked"}).status_code == 404
        assert foreign_client.delete(f"/inbound-rules/sets/{set_id}").status_code == 404


class TestInboundRulesBulkImport:
    def test_bulk_import_new_set(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        raw_csv = """phone_number,label
+1 (555) 100-2000,Spammer A
+1 (555) 300-4000,Spammer B
5555006000
+1 (555) 100-2000,Duplicate Should Skip
"""

        resp = client.post(
            "/inbound-rules/sets/import",
            json={
                "raw_text": raw_csv,
                "new_rule_set_name": "CSV Imported Blocklist",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["imported_count"] == 3
        assert body["skipped_count"] == 1
        assert body["total_rules_count"] == 3
        assert body["rule_set"]["name"] == "CSV Imported Blocklist"

    def test_bulk_import_append_to_existing_set(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        create_resp = client.post(
            "/inbound-rules/sets",
            json={"name": "Existing Set", "rules": [{"phone_number_pattern": "5551110000"}]},
        )
        set_id = create_resp.json()["id"]

        raw_text = """5551110000,Existing Number Skip
5552220000,New Number
5553330000,New Number 2
"""
        resp = client.post(
            "/inbound-rules/sets/import",
            json={
                "rule_set_id": set_id,
                "raw_text": raw_text,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["imported_count"] == 2
        assert body["skipped_count"] == 1
        assert body["total_rules_count"] == 3


class TestFlowInboundRulesAssignment:
    def test_get_flow_inbound_rules_default(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_readonly_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/inbound-rules")
        assert resp.status_code == 200
        body = resp.json()
        assert body["inbound_rule_set_id"] is None
        assert body["inbound_rule_set_name"] is None
        assert body["active_rules_count"] == 0

    def test_assign_and_detach_rule_set_on_flow(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        create_resp = client.post(
            "/inbound-rules/sets",
            json={
                "name": "Flow Blocklist",
                "rules": [
                    {"phone_number_pattern": "5550001111"},
                    {"phone_number_pattern": "5550002222"},
                ],
            },
        )
        set_id = create_resp.json()["id"]

        # Assign
        put_resp = client.put(
            f"/flows/{flow.id}/inbound-rules",
            json={"inbound_rule_set_id": set_id},
        )
        assert put_resp.status_code == 200
        body = put_resp.json()
        assert body["inbound_rule_set_id"] == set_id
        assert body["inbound_rule_set_name"] == "Flow Blocklist"
        assert body["active_rules_count"] == 2

        # Detach
        detach_resp = client.put(
            f"/flows/{flow.id}/inbound-rules",
            json={"inbound_rule_set_id": None},
        )
        assert detach_resp.status_code == 200
        assert detach_resp.json()["inbound_rule_set_id"] is None
        assert detach_resp.json()["active_rules_count"] == 0

    def test_assign_invalid_rule_set_returns_404(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/inbound-rules",
            json={"inbound_rule_set_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

    def test_assign_foreign_rule_set_returns_404(
        self, db, workspace, other_workspace, flow
    ):
        foreign_principal = _principal(other_workspace.id)
        foreign_client = _build_app(db, foreign_principal)
        create_resp = foreign_client.post(
            "/inbound-rules/sets",
            json={"name": "Foreign Blocklist"},
        )
        foreign_set_id = create_resp.json()["id"]

        # Attempt to assign foreign rule set to our flow
        client = _build_app(db, _principal(workspace.id))
        resp = client.put(
            f"/flows/{flow.id}/inbound-rules",
            json={"inbound_rule_set_id": foreign_set_id},
        )
        assert resp.status_code == 404

    def test_foreign_tenant_cannot_access_flow_inbound_rules(
        self, db, other_workspace, flow
    ):
        foreign_principal = _principal(other_workspace.id)
        foreign_client = _build_app(db, foreign_principal)

        assert (
            foreign_client.get(f"/flows/{flow.id}/inbound-rules").status_code
            == 404
        )
        assert (
            foreign_client.put(
                f"/flows/{flow.id}/inbound-rules",
                json={"inbound_rule_set_id": None},
            ).status_code
            == 404
        )


class TestInboundRulesRBAC:
    def test_readonly_user_forbidden_on_mutations(self, db, workspace, flow):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        # POST /inbound-rules/sets -> 403
        resp = forbidden_client.post(
            "/inbound-rules/sets", json={"name": "Test Set"}
        )
        assert resp.status_code == 403

        # PUT /inbound-rules/sets/{id} -> 403
        resp = forbidden_client.put(
            f"/inbound-rules/sets/{uuid.uuid4()}", json={"name": "New Name"}
        )
        assert resp.status_code == 403

        # DELETE /inbound-rules/sets/{id} -> 403
        resp = forbidden_client.delete(f"/inbound-rules/sets/{uuid.uuid4()}")
        assert resp.status_code == 403

        # POST /inbound-rules/sets/import -> 403
        resp = forbidden_client.post(
            "/inbound-rules/sets/import",
            json={"raw_text": "5551234567"},
        )
        assert resp.status_code == 403

        # PUT /flows/{flow_id}/inbound-rules -> 403
        resp = forbidden_client.put(
            f"/flows/{flow.id}/inbound-rules",
            json={"inbound_rule_set_id": None},
        )
        assert resp.status_code == 403

    def test_readonly_user_allowed_on_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        # Create a set as admin first
        admin_client = _build_app(db, principal)
        create_resp = admin_client.post(
            "/inbound-rules/sets",
            json={"name": "Read-Only Check Set"},
        )
        set_id = create_resp.json()["id"]

        readonly_client = _build_readonly_app(db, principal)

        # GET /inbound-rules/sets -> 200
        resp = readonly_client.get("/inbound-rules/sets")
        assert resp.status_code == 200

        # GET /inbound-rules/sets/{id} -> 200
        resp = readonly_client.get(f"/inbound-rules/sets/{set_id}")
        assert resp.status_code == 200

        # GET /flows/{id}/inbound-rules -> 200
        resp = readonly_client.get(f"/flows/{flow.id}/inbound-rules")
        assert resp.status_code == 200


class TestInboundRulesAuditLogging:
    def test_audit_logs_recorded_for_all_mutations(self, db, workspace, flow):
        from app.models.audit_log import AuditLog

        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        # 1. Create rule set
        create_resp = client.post(
            "/inbound-rules/sets",
            json={
                "name": "Audit Test Set",
                "rules": [{"phone_number_pattern": "5551112222", "label": "Tag"}],
            },
        )
        assert create_resp.status_code == 201
        set_id = create_resp.json()["id"]

        log_created = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "inbound_rule_set.created",
                AuditLog.tenant_id == workspace.id,
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert log_created is not None
        assert str(log_created.resource_id) == set_id
        assert log_created.resource_type == "inbound_rule_set"

        # 2. Update rule set
        update_resp = client.put(
            f"/inbound-rules/sets/{set_id}",
            json={"name": "Audit Test Set Updated"},
        )
        assert update_resp.status_code == 200

        log_updated = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "inbound_rule_set.updated",
                AuditLog.tenant_id == workspace.id,
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert log_updated is not None
        assert str(log_updated.resource_id) == set_id

        # 3. Import rules
        import_resp = client.post(
            "/inbound-rules/sets/import",
            json={
                "rule_set_id": set_id,
                "raw_text": "5553334444,Imported Tag\n5555556666",
            },
        )
        assert import_resp.status_code == 200

        log_imported = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "inbound_rule_set.imported",
                AuditLog.tenant_id == workspace.id,
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert log_imported is not None
        assert str(log_imported.resource_id) == set_id

        # 4. Assign flow inbound rules
        flow_resp = client.put(
            f"/flows/{flow.id}/inbound-rules",
            json={"inbound_rule_set_id": set_id},
        )
        assert flow_resp.status_code == 200

        log_flow = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "flow_inbound_rules.updated",
                AuditLog.tenant_id == workspace.id,
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert log_flow is not None
        assert str(log_flow.resource_id) == str(flow.id)
        assert log_flow.resource_type == "call_flow"

        # 5. Delete rule set
        del_resp = client.delete(f"/inbound-rules/sets/{set_id}")
        assert del_resp.status_code == 204

        log_deleted = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "inbound_rule_set.deleted",
                AuditLog.tenant_id == workspace.id,
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert log_deleted is not None
        assert str(log_deleted.resource_id) == set_id


class TestSchemaValidationAndEdgeCases:
    def test_empty_or_whitespace_name_rejected(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.post("/inbound-rules/sets", json={"name": "   "})
        assert resp.status_code in (400, 422)

    def test_invalid_pattern_no_digits_rejected(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.post(
            "/inbound-rules/sets",
            json={
                "name": "Invalid Number Set",
                "rules": [{"phone_number_pattern": "no-digits-here"}],
            },
        )
        assert resp.status_code in (400, 422)

    def test_extra_forbidden_fields_rejected(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.post(
            "/inbound-rules/sets",
            json={"name": "Valid Name", "extra_bad_field": "disallowed"},
        )
        assert resp.status_code in (400, 422)

    def test_import_empty_text_rejected(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.post(
            "/inbound-rules/sets/import",
            json={"raw_text": "   "},
        )
        assert resp.status_code in (400, 422)

    def test_import_various_csv_headers_and_quotes(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        csv_content = """\"Phone Number\",\"Label\"
\"+1 (555) 777-8888\",\"Important Spammer\"
\"+1 (555) 999-0000\",
5551239999,\"Another Label\"
"""
        resp = client.post(
            "/inbound-rules/sets/import",
            json={
                "raw_text": csv_content,
                "new_rule_set_name": "Header Variations Set",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["imported_count"] == 3
        assert body["skipped_count"] == 0

