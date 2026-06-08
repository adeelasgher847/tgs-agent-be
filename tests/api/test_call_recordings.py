"""
Tests for the call recordings API and supporting services.

Covers:
  GET /api/v1/recordings/{call_id}
  PUT /api/v1/phone-numbers/{id}/configuration
  GET /api/v1/phone-numbers/{id}/configuration
  get_recording_enabled_for_call() helper
  call_recording_upload_service error path
"""

from __future__ import annotations

import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.middleware.api_key_middleware import _attach_workspace_context
from app.models.agent import Agent
from app.models.call_session import CallSession
from app.models.phone_number import NumberConfiguration, PhoneNumber
from app.models.tenant import Tenant
from app.services.recording_config_service import get_recording_enabled_for_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_for(tenant: Tenant) -> Dict[str, Any]:
    return {
        "api_key_id": str(uuid.uuid4()),
        "tenant_id": str(tenant.id),
        "key_is_active": True,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": getattr(tenant, "schema_name", "test"),
            "status": "active",
        },
    }


def _headers(tenant: Tenant) -> Dict[str, str]:
    return {"X-API-Key": "test-rec-key", "X-Workspace-ID": str(tenant.id)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rec_tenant(db):
    t = Tenant(name="Recording Tenant", schema_name="rec_test", status="active")
    db.add(t)
    db.commit()
    db.refresh(t)
    yield t
    db.query(CallSession).filter(CallSession.tenant_id == t.id).delete()
    db.query(PhoneNumber).filter(PhoneNumber.tenant_id == t.id).delete()
    db.query(Agent).filter(Agent.tenant_id == t.id).delete()
    db.delete(t)
    db.commit()


@pytest.fixture
def rec_agent(db, rec_tenant):
    agent = Agent(name="Rec Agent", tenant_id=rec_tenant.id, status="ready")
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


@pytest.fixture
def rec_phone(db, rec_tenant, rec_agent):
    pn = PhoneNumber(
        phone_number="+61298765432",
        tenant_id=rec_tenant.id,
        assistant_id=rec_agent.id,
        status="active",
    )
    db.add(pn)
    db.commit()
    db.refresh(pn)
    return pn


@pytest.fixture
def rec_phone_with_recording_enabled(db, rec_phone):
    config = NumberConfiguration(phone_number_id=rec_phone.id, recording_enabled=True)
    db.add(config)
    db.commit()
    db.refresh(config)
    return rec_phone


@pytest.fixture
def rec_phone_recording_disabled(db, rec_phone):
    config = NumberConfiguration(phone_number_id=rec_phone.id, recording_enabled=False)
    db.add(config)
    db.commit()
    return rec_phone


@pytest.fixture
def call_session_with_recording(db, rec_tenant, rec_agent, rec_phone_with_recording_enabled):
    session = CallSession(
        user_id=uuid.uuid4(),
        agent_id=rec_agent.id,
        tenant_id=rec_tenant.id,
        start_time=__import__("datetime").datetime.utcnow(),
        status="completed",
        call_type="outbound",
        duration=120,
        assistant_phone_number=rec_phone_with_recording_enabled.phone_number,
        recording_gcs_path="recordings/tenant1/call1/20260609.opus",
        recording_error=False,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    yield session
    db.delete(session)
    db.commit()


@pytest.fixture
def call_session_no_recording(db, rec_tenant, rec_agent, rec_phone_recording_disabled):
    session = CallSession(
        user_id=uuid.uuid4(),
        agent_id=rec_agent.id,
        tenant_id=rec_tenant.id,
        start_time=__import__("datetime").datetime.utcnow(),
        status="completed",
        call_type="outbound",
        duration=90,
        assistant_phone_number=rec_phone_recording_disabled.phone_number,
        recording_gcs_path=None,
        recording_error=False,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    yield session
    db.delete(session)
    db.commit()


@pytest.fixture
def authed_client(client: TestClient, rec_tenant: Tenant):
    payload = _payload_for(rec_tenant)

    async def _resolve(_key_hash, _workspace_id):
        return payload

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield client


# ---------------------------------------------------------------------------
# get_recording_enabled_for_call
# ---------------------------------------------------------------------------


class TestGetRecordingEnabledForCall:
    def test_recording_enabled_true(self, db, rec_tenant, rec_agent, rec_phone_with_recording_enabled):
        session = CallSession(
            user_id=uuid.uuid4(),
            agent_id=rec_agent.id,
            tenant_id=rec_tenant.id,
            start_time=__import__("datetime").datetime.utcnow(),
            status="completed",
            call_type="outbound",
            assistant_phone_number=rec_phone_with_recording_enabled.phone_number,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        result = get_recording_enabled_for_call(db, session)
        assert result is True

        db.delete(session)
        db.commit()

    def test_recording_enabled_false(self, db, rec_tenant, rec_agent, rec_phone_recording_disabled):
        session = CallSession(
            user_id=uuid.uuid4(),
            agent_id=rec_agent.id,
            tenant_id=rec_tenant.id,
            start_time=__import__("datetime").datetime.utcnow(),
            status="completed",
            call_type="outbound",
            assistant_phone_number=rec_phone_recording_disabled.phone_number,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        result = get_recording_enabled_for_call(db, session)
        assert result is False

        db.delete(session)
        db.commit()

    def test_no_phone_number_returns_false(self, db, rec_tenant, rec_agent):
        session = CallSession(
            user_id=uuid.uuid4(),
            agent_id=rec_agent.id,
            tenant_id=rec_tenant.id,
            start_time=__import__("datetime").datetime.utcnow(),
            status="completed",
            call_type="web",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        result = get_recording_enabled_for_call(db, session)
        assert result is False

        db.delete(session)
        db.commit()


# ---------------------------------------------------------------------------
# GET /api/v1/recordings/{call_id}
# ---------------------------------------------------------------------------


class TestGetRecording:
    def test_returns_signed_url_when_recording_available(
        self, authed_client: TestClient, rec_tenant: Tenant, call_session_with_recording
    ):
        mock_url = "https://storage.googleapis.com/bucket/recordings/xyz?X-Goog-Signature=abc"
        with (
            patch(
                "app.services.gcs_recording_service.generate_signed_url",
                return_value=mock_url,
            ),
            patch(
                "app.services.gcs_recording_service.get_object_size",
                return_value=204800,
            ),
        ):
            resp = authed_client.get(
                f"/api/v1/recordings/{call_session_with_recording.id}",
                headers=_headers(rec_tenant),
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["url"] == mock_url
        assert body["data"]["duration"] == 120
        assert body["data"]["size"] == 204800

    def test_returns_404_when_recording_disabled(
        self, authed_client: TestClient, rec_tenant: Tenant, call_session_no_recording
    ):
        resp = authed_client.get(
            f"/api/v1/recordings/{call_session_no_recording.id}",
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 404

    def test_returns_404_for_wrong_tenant(
        self, authed_client: TestClient, rec_tenant: Tenant, call_session_with_recording
    ):
        # Use a different tenant header — the tenant_id won't match the session
        other_tenant_id = str(uuid.uuid4())
        resp = authed_client.get(
            f"/api/v1/recordings/{call_session_with_recording.id}",
            headers={"X-API-Key": "test-rec-key", "X-Workspace-ID": other_tenant_id},
        )
        assert resp.status_code in (401, 403, 404)

    def test_returns_404_when_call_not_found(
        self, authed_client: TestClient, rec_tenant: Tenant
    ):
        resp = authed_client.get(
            f"/api/v1/recordings/{uuid.uuid4()}",
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 404

    def test_returns_404_when_recording_error_and_no_path(
        self, db, authed_client: TestClient, rec_tenant: Tenant, rec_agent, rec_phone_with_recording_enabled
    ):
        session = CallSession(
            user_id=uuid.uuid4(),
            agent_id=rec_agent.id,
            tenant_id=rec_tenant.id,
            start_time=__import__("datetime").datetime.utcnow(),
            status="completed",
            call_type="outbound",
            assistant_phone_number=rec_phone_with_recording_enabled.phone_number,
            recording_gcs_path=None,
            recording_error=True,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        resp = authed_client.get(
            f"/api/v1/recordings/{session.id}",
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 404

        db.delete(session)
        db.commit()


# ---------------------------------------------------------------------------
# PUT /GET /api/v1/phone-numbers/{id}/configuration
# ---------------------------------------------------------------------------


class TestPhoneNumberConfiguration:
    def test_put_enables_recording(
        self, authed_client: TestClient, rec_tenant: Tenant, rec_phone
    ):
        resp = authed_client.put(
            f"/api/v1/phone-numbers/{rec_phone.id}/configuration",
            json={"recording_enabled": True, "max_duration_seconds": 1800},
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["recording_enabled"] is True
        assert body["data"]["max_duration_seconds"] == 1800

    def test_put_disables_recording(
        self, authed_client: TestClient, rec_tenant: Tenant, rec_phone
    ):
        resp = authed_client.put(
            f"/api/v1/phone-numbers/{rec_phone.id}/configuration",
            json={"recording_enabled": False, "max_duration_seconds": 3600},
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["recording_enabled"] is False

    def test_get_configuration(
        self, authed_client: TestClient, rec_tenant: Tenant, rec_phone_with_recording_enabled
    ):
        resp = authed_client.get(
            f"/api/v1/phone-numbers/{rec_phone_with_recording_enabled.id}/configuration",
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["recording_enabled"] is True

    def test_get_configuration_not_found(
        self, authed_client: TestClient, rec_tenant: Tenant, rec_phone
    ):
        # rec_phone has no configuration row yet
        resp = authed_client.get(
            f"/api/v1/phone-numbers/{rec_phone.id}/configuration",
            headers=_headers(rec_tenant),
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Upload service — error path
# ---------------------------------------------------------------------------


class TestCallRecordingUploadService:
    def test_upload_failure_sets_recording_error(self, db, rec_tenant, rec_agent, rec_phone_with_recording_enabled):
        """When LiveKit egress fails, recording_error must be set to True."""
        from app.services.call_recording_upload_service import _check_and_finalize

        session = CallSession(
            user_id=uuid.uuid4(),
            agent_id=rec_agent.id,
            tenant_id=rec_tenant.id,
            start_time=__import__("datetime").datetime.utcnow(),
            status="completed",
            call_type="outbound",
            assistant_phone_number=rec_phone_with_recording_enabled.phone_number,
            recording_gcs_path=None,
            recording_error=False,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # EGRESS_FAILED = 4 (livekit.api integer constant — avoid importing livekit in
        # tests because conftest mocks 'google' which breaks protobuf)
        mock_egress_info = MagicMock()
        mock_egress_info.status = 4  # EGRESS_FAILED

        with patch(
            "app.services.call_recording_upload_service._fetch_egress_info_async",
            return_value=mock_egress_info,
        ):
            _check_and_finalize(db, session, "egress-id-123", "recordings/ws/call/20260609.opus")

        db.refresh(session)
        assert session.recording_error is True
        assert session.recording_gcs_path is None

        db.delete(session)
        db.commit()

    def test_upload_success_sets_gcs_path(self, db, rec_tenant, rec_agent, rec_phone_with_recording_enabled):
        """When LiveKit egress completes, recording_gcs_path must be set."""
        from app.services.call_recording_upload_service import _check_and_finalize

        session = CallSession(
            user_id=uuid.uuid4(),
            agent_id=rec_agent.id,
            tenant_id=rec_tenant.id,
            start_time=__import__("datetime").datetime.utcnow(),
            status="completed",
            call_type="outbound",
            assistant_phone_number=rec_phone_with_recording_enabled.phone_number,
            recording_gcs_path=None,
            recording_error=False,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # EGRESS_COMPLETE = 3
        mock_egress_info = MagicMock()
        mock_egress_info.status = 3  # EGRESS_COMPLETE

        gcs_path = f"recordings/{rec_tenant.id}/{session.id}/20260609.opus"

        with (
            patch(
                "app.services.call_recording_upload_service._stop_egress_async",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.call_recording_upload_service._fetch_egress_info_async",
                return_value=mock_egress_info,
            ),
            patch("app.services.call_recording_upload_service.asyncio.sleep", new_callable=AsyncMock),
            patch("app.services.gcs_recording_service.update_object_metadata"),
        ):
            _check_and_finalize(db, session, "egress-id-456", gcs_path)

        db.refresh(session)
        assert session.recording_gcs_path == gcs_path
        assert session.recording_error is False

        db.delete(session)
        db.commit()
