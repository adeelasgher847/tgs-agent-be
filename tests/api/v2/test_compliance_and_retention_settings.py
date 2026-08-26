"""Unit & integration tests for Compliance & Detection and Data Retention settings on Call Flows.

Covers:
- GET /api/v2/flows/{flow_id}/compliance-detection-settings
- PUT /api/v2/flows/{flow_id}/compliance-detection-settings
- GET /api/v2/flows/{flow_id}/data-retention-settings
- PUT /api/v2/flows/{flow_id}/data-retention-settings
- POST /api/v2/flows/{flow_id}/data-retention/purge
- Extra fields validation (extra='forbid')
- Day boundary validation (1 <= days <= 365)
- RBAC permissions (admin for PUT/POST, readonly for GET)
- Tenant isolation (404 for foreign or non-existent flows)
- Audit log emission
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.api.v2.routers.flows import router as flows_router
from app.core.exception_handlers import register_exception_handlers
from app.models.audit_log import AuditLog
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User
from app.services.call_flow_service import CallFlowService


def _build_app(db_override, principal, *, forbidden=False):
    from app.api.deps import (
        get_db,
        require_admin_or_api_key,
        require_config_or_api_key,
        require_readonly_or_api_key,
    )

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(flows_router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin rank required",
            )

        mini.dependency_overrides[require_admin_or_api_key] = _raise_forbidden
        mini.dependency_overrides[require_config_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal
        mini.dependency_overrides[require_config_or_api_key] = lambda: principal

    mini.dependency_overrides[require_readonly_or_api_key] = lambda: principal
    mini.dependency_overrides[get_db] = lambda: db_override
    return TestClient(mini, raise_server_exceptions=False)


def _principal(tenant_id: uuid.UUID) -> User:
    u = MagicMock(spec=User)
    u.id = uuid.uuid4()
    u.current_tenant_id = tenant_id
    u.role = "admin"
    return u


@pytest.fixture
def workspace(db):
    tenant = Tenant(
        name=f"CompWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"test_comp_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def flow(db, workspace):
    f = CallFlow(
        tenant_id=workspace.id,
        agent_id=uuid.uuid4(),
        name="Compliance and Retention Flow",
        direction="inbound",
        status="active",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestComplianceDetectionSettingsAPI:
    def test_get_compliance_detection_settings_defaults(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/compliance-detection-settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["compliance_monitoring_enabled"] is False
        assert data["anti_bot_detection_enabled"] is False
        assert data["terminate_on_fake_voice"] is False

    def test_update_compliance_detection_settings_success(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "compliance_monitoring_enabled": True,
            "anti_bot_detection_enabled": True,
            "terminate_on_fake_voice": True,
        }
        resp = client.put(
            f"/flows/{flow.id}/compliance-detection-settings", json=payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compliance_monitoring_enabled"] is True
        assert data["anti_bot_detection_enabled"] is True
        assert data["terminate_on_fake_voice"] is True

        # Verify persisted
        db.refresh(flow)
        assert flow.compliance_monitoring_enabled is True
        assert flow.anti_bot_detection_enabled is True
        assert flow.terminate_on_fake_voice is True

    def test_extra_fields_forbidden(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/compliance-detection-settings",
            json={
                "compliance_monitoring_enabled": True,
                "unknown_setting": "malicious",
            },
        )
        assert resp.status_code in (400, 422)

    def test_tenant_isolation_returns_404(self, db, workspace, flow):
        other_tenant_id = uuid.uuid4()
        foreign_principal = _principal(other_tenant_id)
        foreign_client = _build_app(db, foreign_principal)

        assert (
            foreign_client.get(
                f"/flows/{flow.id}/compliance-detection-settings"
            ).status_code
            == 404
        )
        assert (
            foreign_client.put(
                f"/flows/{flow.id}/compliance-detection-settings",
                json={"compliance_monitoring_enabled": True},
            ).status_code
            == 404
        )

    def test_null_column_safety(self):
        mock_flow = MagicMock(spec=CallFlow)
        mock_flow.compliance_monitoring_enabled = None
        mock_flow.anti_bot_detection_enabled = None
        mock_flow.terminate_on_fake_voice = None

        res = CallFlowService._to_compliance_detection_response(mock_flow)
        assert res.compliance_monitoring_enabled is False
        assert res.anti_bot_detection_enabled is False
        assert res.terminate_on_fake_voice is False

    def test_rbac_readonly_forbidden_on_put(self, db, workspace, flow):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        resp = forbidden_client.put(
            f"/flows/{flow.id}/compliance-detection-settings",
            json={"compliance_monitoring_enabled": True},
        )
        assert resp.status_code == 403

    def test_rbac_readonly_allowed_on_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        resp = forbidden_client.get(
            f"/flows/{flow.id}/compliance-detection-settings"
        )
        assert resp.status_code == 200

    def test_audit_log_recorded_on_update(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/compliance-detection-settings",
            json={"anti_bot_detection_enabled": True},
        )
        assert resp.status_code == 200

        log = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "compliance_detection_settings.updated",
                AuditLog.resource_id == flow.id,
            )
            .first()
        )
        assert log is not None
        assert log.new_value["anti_bot_detection_enabled"] is True


class TestDataRetentionSettingsAPI:
    def test_get_data_retention_settings_defaults(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/data-retention-settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["retention_policy_enabled"] is False
        assert data["retention_transcript_enabled"] is False
        assert data["retention_transcript_days"] == 30
        assert data["retention_summary_enabled"] is False
        assert data["retention_summary_days"] == 30
        assert data["retention_recording_enabled"] is False
        assert data["retention_recording_days"] == 30

    def test_update_data_retention_settings_success(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "retention_policy_enabled": True,
            "retention_transcript_enabled": True,
            "retention_transcript_days": 60,
            "retention_summary_enabled": True,
            "retention_summary_days": 90,
            "retention_recording_enabled": True,
            "retention_recording_days": 180,
        }
        resp = client.put(
            f"/flows/{flow.id}/data-retention-settings", json=payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retention_policy_enabled"] is True
        assert data["retention_transcript_enabled"] is True
        assert data["retention_transcript_days"] == 60
        assert data["retention_summary_enabled"] is True
        assert data["retention_summary_days"] == 90
        assert data["retention_recording_enabled"] is True
        assert data["retention_recording_days"] == 180

        # Verify persisted
        db.refresh(flow)
        assert flow.retention_policy_enabled is True
        assert flow.retention_transcript_days == 60
        assert flow.retention_summary_days == 90
        assert flow.retention_recording_days == 180

    def test_day_boundaries_validation(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        # Days < 1 -> 400/422
        assert (
            client.put(
                f"/flows/{flow.id}/data-retention-settings",
                json={"retention_transcript_days": 0},
            ).status_code
            in (400, 422)
        )

        # Days > 365 -> 400/422
        assert (
            client.put(
                f"/flows/{flow.id}/data-retention-settings",
                json={"retention_recording_days": 366},
            ).status_code
            in (400, 422)
        )

        # Days = 1 -> 200
        assert (
            client.put(
                f"/flows/{flow.id}/data-retention-settings",
                json={"retention_transcript_days": 1},
            ).status_code
            == 200
        )

        # Days = 365 -> 200
        assert (
            client.put(
                f"/flows/{flow.id}/data-retention-settings",
                json={"retention_recording_days": 365},
            ).status_code
            == 200
        )

    def test_extra_fields_forbidden(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/data-retention-settings",
            json={
                "retention_policy_enabled": True,
                "unauthorized_field": 123,
            },
        )
        assert resp.status_code in (400, 422)

    def test_tenant_isolation_returns_404(self, db, workspace, flow):
        other_tenant_id = uuid.uuid4()
        foreign_principal = _principal(other_tenant_id)
        foreign_client = _build_app(db, foreign_principal)

        assert (
            foreign_client.get(
                f"/flows/{flow.id}/data-retention-settings"
            ).status_code
            == 404
        )
        assert (
            foreign_client.put(
                f"/flows/{flow.id}/data-retention-settings",
                json={"retention_policy_enabled": True},
            ).status_code
            == 404
        )
        assert (
            foreign_client.post(
                f"/flows/{flow.id}/data-retention/purge"
            ).status_code
            == 404
        )

    def test_null_column_safety(self):
        mock_flow = MagicMock(spec=CallFlow)
        mock_flow.retention_policy_enabled = None
        mock_flow.retention_transcript_enabled = None
        mock_flow.retention_transcript_days = None
        mock_flow.retention_summary_enabled = None
        mock_flow.retention_summary_days = None
        mock_flow.retention_recording_enabled = None
        mock_flow.retention_recording_days = None

        res = CallFlowService._to_data_retention_response(mock_flow)
        assert res.retention_policy_enabled is False
        assert res.retention_transcript_enabled is False
        assert res.retention_transcript_days == 30
        assert res.retention_summary_enabled is False
        assert res.retention_summary_days == 30
        assert res.retention_recording_enabled is False
        assert res.retention_recording_days == 30

    def test_audit_log_recorded_on_retention_update(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/data-retention-settings",
            json={"retention_policy_enabled": True},
        )
        assert resp.status_code == 200

        log = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "data_retention_settings.updated",
                AuditLog.resource_id == flow.id,
            )
            .first()
        )
        assert log is not None
        assert log.new_value["retention_policy_enabled"] is True

    def test_post_data_retention_purge_endpoint(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        flow.retention_policy_enabled = True
        flow.retention_transcript_enabled = True
        flow.retention_transcript_days = 30
        db.commit()

        # Create expired session
        old_session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            call_transcript=[{"role": "user", "content": "hello"}],
        )
        db.add(old_session)
        db.commit()

        resp = client.post(f"/flows/{flow.id}/data-retention/purge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["purged_transcripts_count"] == 1
        assert data["purged_sessions_count"] == 1

        db.refresh(old_session)
        assert old_session.call_transcript is None

        # Verify audit log
        log = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "data_retention.purged",
                AuditLog.resource_id == flow.id,
            )
            .first()
        )
        assert log is not None
        assert log.new_value["purged_transcripts_count"] == 1

    def test_data_retention_rbac_readonly_forbidden_on_put_and_purge(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        put_resp = forbidden_client.put(
            f"/flows/{flow.id}/data-retention-settings",
            json={"retention_policy_enabled": True},
        )
        assert put_resp.status_code == 403

        purge_resp = forbidden_client.post(
            f"/flows/{flow.id}/data-retention/purge"
        )
        assert purge_resp.status_code == 403

    def test_data_retention_rbac_readonly_allowed_on_get(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        resp = forbidden_client.get(
            f"/flows/{flow.id}/data-retention-settings"
        )
        assert resp.status_code == 200

    def test_non_existent_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)
        non_existent_id = uuid.uuid4()

        assert (
            client.get(
                f"/flows/{non_existent_id}/compliance-detection-settings"
            ).status_code
            == 404
        )
        assert (
            client.put(
                f"/flows/{non_existent_id}/compliance-detection-settings",
                json={"compliance_monitoring_enabled": True},
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/flows/{non_existent_id}/data-retention-settings"
            ).status_code
            == 404
        )
        assert (
            client.put(
                f"/flows/{non_existent_id}/data-retention-settings",
                json={"retention_policy_enabled": True},
            ).status_code
            == 404
        )
        assert (
            client.post(
                f"/flows/{non_existent_id}/data-retention/purge"
            ).status_code
            == 404
        )

    def test_soft_deleted_flow_returns_404(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        flow.is_deleted = True
        db.commit()

        assert (
            client.get(
                f"/flows/{flow.id}/compliance-detection-settings"
            ).status_code
            == 404
        )
        assert (
            client.put(
                f"/flows/{flow.id}/compliance-detection-settings",
                json={"compliance_monitoring_enabled": True},
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/flows/{flow.id}/data-retention-settings"
            ).status_code
            == 404
        )
        assert (
            client.put(
                f"/flows/{flow.id}/data-retention-settings",
                json={"retention_policy_enabled": True},
            ).status_code
            == 404
        )
        assert (
            client.post(
                f"/flows/{flow.id}/data-retention/purge"
            ).status_code
            == 404
        )

    def test_post_data_retention_purge_when_disabled_returns_zero(
        self, db, workspace, flow
    ):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        flow.retention_policy_enabled = False
        db.commit()

        resp = client.post(f"/flows/{flow.id}/data-retention/purge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["purged_transcripts_count"] == 0
        assert data["purged_sessions_count"] == 0
