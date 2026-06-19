"""
Tests for GDPR data-portability endpoints.

Coverage:
  - POST /workspace/data-export: 202 + job_id, creates job, enqueues ARQ, audits
  - GET /workspace/data-export/{job_id}: processing / ready (+download_url) / error
  - GET /workspace/data-export/{job_id}: 404 for unknown job
  - Admin RBAC required for both endpoints
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.api.deps import get_db, require_admin
from app.core.exception_handlers import register_exception_handlers

WORKSPACE_ID = uuid.uuid4()
USER_ID = uuid.uuid4()
JOB_ID = uuid.uuid4()


def _make_admin_user() -> MagicMock:
    user = MagicMock()
    user.id = USER_ID
    user.current_tenant_id = WORKSPACE_ID
    return user


def _build_app(db_override, *, admin_override=None) -> TestClient:
    from app.api.v2.routers.workspace import router as workspace_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(workspace_router)

    mini.dependency_overrides[require_admin] = admin_override or (lambda: _make_admin_user())
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


class TestTriggerDataExport:
    def test_returns_202_with_job_id(self):
        db = MagicMock()
        fake_job = MagicMock(id=JOB_ID)

        with (
            patch("app.api.v2.routers.workspace.create_export_job", return_value=fake_job) as mock_create,
            patch("app.api.v2.routers.workspace._enqueue_data_export_job", new_callable=AsyncMock) as mock_enqueue,
            patch("app.api.v2.routers.workspace.log_audit_event") as mock_audit,
        ):
            client = _build_app(db)
            resp = client.post("/workspace/data-export")

        assert resp.status_code == status.HTTP_202_ACCEPTED, resp.text
        assert resp.json()["job_id"] == str(JOB_ID)
        mock_create.assert_called_once_with(db, WORKSPACE_ID, USER_ID)
        mock_enqueue.assert_awaited_once_with(str(JOB_ID))
        mock_audit.assert_called_once()
        assert mock_audit.call_args.kwargs["action"] == "workspace.data_export_requested"

    def test_requires_admin(self):
        db = MagicMock()

        def _forbidden():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

        client = _build_app(db, admin_override=_forbidden)
        resp = client.post("/workspace/data-export")

        assert resp.status_code == status.HTTP_403_FORBIDDEN


class TestGetDataExportStatus:
    def test_processing_status_has_no_download_url(self):
        db = MagicMock()
        job = MagicMock(id=JOB_ID, status="processing", gcs_path=None)

        with patch("app.api.v2.routers.workspace.get_export_job", return_value=job):
            client = _build_app(db)
            resp = client.get(f"/workspace/data-export/{JOB_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processing"
        assert body["download_url"] is None

    def test_ready_status_returns_signed_url(self):
        db = MagicMock()
        job = MagicMock(id=JOB_ID, status="ready", gcs_path=f"data-exports/{WORKSPACE_ID}/{JOB_ID}.zip")

        with (
            patch("app.api.v2.routers.workspace.get_export_job", return_value=job),
            patch(
                "app.services.gcs_recording_service.generate_signed_url",
                return_value="https://signed.example.com/export.zip",
            ) as mock_signed,
        ):
            client = _build_app(db)
            resp = client.get(f"/workspace/data-export/{JOB_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["download_url"] == "https://signed.example.com/export.zip"
        mock_signed.assert_called_once()
        assert mock_signed.call_args.args[0] == job.gcs_path

    def test_error_status_has_no_download_url(self):
        db = MagicMock()
        job = MagicMock(id=JOB_ID, status="error", gcs_path=None)

        with patch("app.api.v2.routers.workspace.get_export_job", return_value=job):
            client = _build_app(db)
            resp = client.get(f"/workspace/data-export/{JOB_ID}")

        assert resp.status_code == 200
        assert resp.json() == {"status": "error", "download_url": None}

    def test_unknown_job_returns_404(self):
        db = MagicMock()

        with patch("app.api.v2.routers.workspace.get_export_job", return_value=None):
            client = _build_app(db)
            resp = client.get(f"/workspace/data-export/{uuid.uuid4()}")

        assert resp.status_code == 404
