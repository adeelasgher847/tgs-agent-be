"""Unit and API integration tests for Call Flow Recording Settings.

Coverage:
  - PUT /api/v2/flows/{flow_id}/recording-settings (configure recording switches)
  - GET /api/v2/flows/{flow_id}/recording-settings (fetch configured recording settings)
  - Validation: extra fields rejected, schema boundaries
  - RBAC: Admin required for PUT, Read-only allowed for GET, 403 on unauthorized mutation
  - Tenant isolation: 404 for other tenant's flow
  - Audit logging: 'recording_settings.updated' event recorded
  - Public recording access: GET /api/v1/recordings/public/{call_id}
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers
from app.core.request_auth import ApiKeyPrincipal
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
        require_readonly_or_api_key,
    )
    from app.api.v2.routers.flows import router as flows_router

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
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal

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
    t = Tenant(
        name=f"RecWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"rec_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def other_workspace(db):
    t = Tenant(
        name=f"OtherWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def flow(db, workspace):
    f = CallFlow(
        tenant_id=workspace.id,
        agent_id=uuid.uuid4(),
        name="Recording Flow",
        direction="inbound",
        status="active",
        recording_enabled=True,
        public_recording_enabled=False,
        faster_inbound_pickup=False,
        stop_recording_on_transfer=False,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


class TestRecordingSettingsAPI:
    def test_get_recording_settings_defaults(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.get(f"/flows/{flow.id}/recording-settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["recording_enabled"] is True
        assert body["public_recording_enabled"] is False
        assert body["faster_inbound_pickup"] is False
        assert body["stop_recording_on_transfer"] is False

    def test_update_recording_settings_success(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        payload = {
            "recording_enabled": False,
            "public_recording_enabled": True,
            "faster_inbound_pickup": True,
            "stop_recording_on_transfer": True,
        }
        resp = client.put(f"/flows/{flow.id}/recording-settings", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["recording_enabled"] is False
        assert body["public_recording_enabled"] is True
        assert body["faster_inbound_pickup"] is True
        assert body["stop_recording_on_transfer"] is True

        # Verify persisted in DB
        db.refresh(flow)
        assert flow.recording_enabled is False
        assert flow.public_recording_enabled is True
        assert flow.faster_inbound_pickup is True
        assert flow.stop_recording_on_transfer is True

    def test_extra_fields_forbidden(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/recording-settings",
            json={"recording_enabled": True, "unexpected_key": "not_allowed"},
        )
        assert resp.status_code in (400, 422)

    def test_tenant_isolation_returns_404(self, db, other_workspace, flow):
        foreign_principal = _principal(other_workspace.id)
        foreign_client = _build_app(db, foreign_principal)

        assert (
            foreign_client.get(f"/flows/{flow.id}/recording-settings").status_code
            == 404
        )
        assert (
            foreign_client.put(
                f"/flows/{flow.id}/recording-settings",
                json={"recording_enabled": False},
            ).status_code
            == 404
        )

    def test_non_existent_flow_returns_404(self, db, workspace):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)
        missing_id = uuid.uuid4()

        assert (
            client.get(f"/flows/{missing_id}/recording-settings").status_code
            == 404
        )
        assert (
            client.put(
                f"/flows/{missing_id}/recording-settings",
                json={"recording_enabled": False},
            ).status_code
            == 404
        )

    def test_null_column_safety(self):
        mock_flow = MagicMock(spec=CallFlow)
        mock_flow.recording_enabled = None
        mock_flow.public_recording_enabled = None
        mock_flow.faster_inbound_pickup = None
        mock_flow.stop_recording_on_transfer = None

        res = CallFlowService._to_recording_response(mock_flow)
        assert res.recording_enabled is True
        assert res.public_recording_enabled is False
        assert res.faster_inbound_pickup is False
        assert res.stop_recording_on_transfer is False

    def test_rbac_readonly_forbidden_on_put(self, db, workspace, flow):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        resp = forbidden_client.put(
            f"/flows/{flow.id}/recording-settings",
            json={"recording_enabled": False},
        )
        assert resp.status_code == 403

    def test_rbac_readonly_allowed_on_get(self, db, workspace, flow):
        principal = _principal(workspace.id)
        forbidden_client = _build_app(db, principal, forbidden=True)

        resp = forbidden_client.get(f"/flows/{flow.id}/recording-settings")
        assert resp.status_code == 200

    def test_audit_log_recorded_on_update(self, db, workspace, flow):
        principal = _principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/recording-settings",
            json={"public_recording_enabled": True},
        )
        assert resp.status_code == 200

        log = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == "recording_settings.updated",
                AuditLog.tenant_id == workspace.id,
            )
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert log is not None
        assert str(log.resource_id) == str(flow.id)
        assert log.resource_type == "call_flow"


class TestPublicRecordingAccessEndpoint:
    @pytest.fixture
    def recording_app(self, db):
        from app.api.deps import get_db
        from app.routers.recordings import router as recordings_router

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(recordings_router, prefix="/api/v1/recordings")

        def _get_db():
            yield db

        app.dependency_overrides[get_db] = _get_db
        return TestClient(app)

    def test_public_recording_access_success_when_enabled(
        self, db, workspace, flow, recording_app
    ):
        flow.public_recording_enabled = True
        flow.recording_enabled = True
        db.commit()

        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path="recordings/test/session_1/audio.opus",
            duration=120,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        with (
            patch(
                "app.services.s3_recording_service.generate_signed_url",
                return_value="https://s3.example.com/signed-public-url",
            ),
            patch(
                "app.services.s3_recording_service.get_object_size",
                return_value=102400,
            ),
        ):
            # Test /public/{call_id}
            resp = recording_app.get(f"/api/v1/recordings/public/{session.id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"]["url"] == "https://s3.example.com/signed-public-url"
            assert data["data"]["duration"] == 120
            assert data["data"]["size"] == 102400

            # Test /{call_id}/public alias
            resp_alias = recording_app.get(f"/api/v1/recordings/{session.id}/public")
            assert resp_alias.status_code == 200
            assert resp_alias.json()["data"]["url"] == "https://s3.example.com/signed-public-url"

    def test_public_recording_access_forbidden_when_disabled(
        self, db, workspace, flow, recording_app
    ):
        flow.public_recording_enabled = False
        db.commit()

        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path="recordings/test/session_2/audio.opus",
        )
        db.add(session)
        db.commit()

        resp = recording_app.get(f"/api/v1/recordings/public/{session.id}")
        assert resp.status_code == 403
        assert "Public recording access is not enabled" in resp.text

    def test_public_recording_access_forbidden_on_hipaa_flow(
        self, db, workspace, flow, recording_app
    ):
        flow.public_recording_enabled = True
        flow.hipaa_compliance = True
        db.commit()

        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path="recordings/test/session_3/audio.opus",
        )
        db.add(session)
        db.commit()

        resp = recording_app.get(f"/api/v1/recordings/public/{session.id}")
        assert resp.status_code == 403
        assert "HIPAA" in resp.text

    def test_public_recording_access_forbidden_when_no_call_flow(
        self, db, workspace, recording_app
    ):
        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            call_flow_id=None,
            call_type="inbound",
            status="completed",
            start_time=datetime.now(timezone.utc),
            recording_s3_path="recordings/test/session_noflow/audio.opus",
        )
        db.add(session)
        db.commit()

        resp = recording_app.get(f"/api/v1/recordings/public/{session.id}")
        assert resp.status_code == 403

    def test_public_recording_access_forbidden_when_flow_deleted(
        self, db, workspace, flow, recording_app
    ):
        flow.public_recording_enabled = True
        flow.is_deleted = True
        db.commit()

        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path="recordings/test/session_del/audio.opus",
        )
        db.add(session)
        db.commit()

        resp = recording_app.get(f"/api/v1/recordings/public/{session.id}")
        assert resp.status_code == 403

    def test_public_recording_access_not_found_when_recording_disabled_on_flow(
        self, db, workspace, flow, recording_app
    ):
        flow.public_recording_enabled = True
        flow.recording_enabled = False
        db.commit()

        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path="recordings/test/session_rec_disabled/audio.opus",
        )
        db.add(session)
        db.commit()

        resp = recording_app.get(f"/api/v1/recordings/public/{session.id}")
        assert resp.status_code == 404
        assert "Recording not enabled" in resp.text

    def test_public_recording_access_not_found_when_session_missing(
        self, recording_app
    ):
        resp = recording_app.get(f"/api/v1/recordings/public/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_public_recording_upload_failed_and_not_ready_errors(
        self, db, workspace, flow, recording_app
    ):
        flow.public_recording_enabled = True
        flow.recording_enabled = True
        db.commit()

        # Session with upload failed
        session_failed = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path=None,
            recording_error=True,
        )
        db.add(session_failed)
        db.commit()

        resp = recording_app.get(f"/api/v1/recordings/public/{session_failed.id}")
        assert resp.status_code == 404
        assert "upload failed" in resp.text

        # Session not available yet
        session_pending = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path=None,
            recording_error=False,
        )
        db.add(session_pending)
        db.commit()

        resp2 = recording_app.get(f"/api/v1/recordings/public/{session_pending.id}")
        assert resp2.status_code == 404
        assert "not available yet" in resp2.text


class TestAuthenticatedRecordingAccessEndpoint:
    @pytest.fixture
    def authed_recording_app(self, db):
        from app.api.deps import get_db, require_tenant
        from app.routers.recordings import router as recordings_router

        def _make_client(principal):
            app = FastAPI()
            register_exception_handlers(app)
            app.include_router(recordings_router, prefix="/api/v1/recordings")

            def _get_db():
                yield db

            app.dependency_overrides[get_db] = _get_db
            app.dependency_overrides[require_tenant] = lambda: principal
            return TestClient(app)

        return _make_client

    def test_hipaa_recording_access_rbac(
        self, db, workspace, flow, authed_recording_app
    ):
        flow.hipaa_compliance = True
        flow.recording_enabled = True
        db.commit()

        session = CallSession(
            tenant_id=workspace.id,
            user_id=uuid.uuid4(),
            agent_id=flow.agent_id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=flow.created_at,
            recording_s3_path="recordings/test/session_hipaa/audio.opus",
            duration=60,
        )
        db.add(session)
        db.commit()

        manager_user = User(
            id=uuid.uuid4(),
            email="manager@example.com",
            current_tenant_id=workspace.id,
            first_name="Manager",
            last_name="User",
            hashed_password="dummy",
        )
        readonly_user = User(
            id=uuid.uuid4(),
            email="readonly@example.com",
            current_tenant_id=workspace.id,
            first_name="Readonly",
            last_name="User",
            hashed_password="dummy",
        )

        with (
            patch(
                "app.services.s3_recording_service.generate_signed_url",
                return_value="https://s3.example.com/signed-hipaa-url",
            ),
            patch(
                "app.services.s3_recording_service.get_object_size",
                return_value=50000,
            ),
        ):
            # Manager role allowed
            with patch("app.services.role_service.get_membership_role_name", return_value="manager"):
                client_manager = authed_recording_app(manager_user)
                resp = client_manager.get(f"/api/v1/recordings/{session.id}")
                assert resp.status_code == 200
                assert resp.json()["data"]["url"] == "https://s3.example.com/signed-hipaa-url"

            # Readonly role forbidden
            with patch("app.services.role_service.get_membership_role_name", return_value="readonly"):
                client_ro = authed_recording_app(readonly_user)
                resp = client_ro.get(f"/api/v1/recordings/{session.id}")
                assert resp.status_code == 403
                assert "HIPAA" in resp.text

            # API Key principal allowed (machine-to-machine)
            api_principal = MagicMock(spec=ApiKeyPrincipal)
            api_principal.current_tenant_id = workspace.id
            client_api = authed_recording_app(api_principal)
            resp_api = client_api.get(f"/api/v1/recordings/{session.id}")
            assert resp_api.status_code == 200
