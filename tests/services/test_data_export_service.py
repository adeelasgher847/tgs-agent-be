"""
Tests for the GDPR export orchestration (app/services/data_export_service.py)
and its ARQ task wrapper (app/workers/batch_call_worker.run_data_export_job).

Coverage:
  - run_export_job: builds a real ZIP (SQLite db), uploads it to GCS, emails
    the workspace admin a signed URL, marks the job 'ready'
  - run_export_job: no admin user found -> still 'ready', email skipped
  - run_export_job: GCS upload failure -> job marked 'error', email skipped
  - run_export_job: the local temp ZIP file is always cleaned up
  - run_data_export_job (ARQ task): loads the job and delegates; no-ops
    gracefully when the job row is missing
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock, patch

from app.models.data_export_job import DataExportJob
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.services.data_export_service import run_export_job


def _make_admin(db, tenant_id):
    admin_role = db.query(Role).filter(Role.name == "admin").first()
    if admin_role is None:
        admin_role = Role(name="admin")
        db.add(admin_role)
        db.commit()
        db.refresh(admin_role)

    user = User(
        email=f"admin-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="hashed",
        first_name="Admin",
        last_name="User",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id, tenant_id=tenant_id, role_id=admin_role.id
        )
    )
    db.commit()
    return user


def _make_job(db, tenant_id):
    job = DataExportJob(workspace_id=tenant_id, status="processing")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


class TestRunExportJob:
    def test_success_uploads_and_emails_admin(self, db):
        tenant = Tenant(name=f"ExportSvc-{uuid.uuid4().hex[:8]}", schema_name="export_svc")
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        admin = _make_admin(db, tenant.id)
        job = _make_job(db, tenant.id)

        captured_paths = []

        def _fake_upload(local_path, key):
            captured_paths.append(local_path)
            assert os.path.exists(local_path)  # must be uploaded before cleanup
            return key

        with (
            patch("app.services.s3_data_export_service.upload_export_zip", side_effect=_fake_upload) as mock_upload,
            patch("app.services.s3_recording_service.generate_signed_url", return_value="https://signed.example/export.zip") as mock_signed,
            patch("app.services.email_service.email_service.send_data_export_ready_email") as mock_email,
        ):
            run_export_job(db, job)

        db.refresh(job)
        assert job.status == "ready"
        assert job.s3_path == f"data-exports/{tenant.id}/{job.id}.zip"
        assert job.completed_at is not None

        mock_upload.assert_called_once()
        assert mock_upload.call_args.args[1] == job.s3_path

        mock_signed.assert_called_once_with(job.s3_path, expiry_seconds=24 * 60 * 60)

        mock_email.assert_called_once_with(admin.email, "https://signed.example/export.zip", tenant.name)

        # Temp file must be removed after upload, regardless of outcome.
        assert not os.path.exists(captured_paths[0])

    def test_no_admin_user_still_marks_ready_but_skips_email(self, db):
        tenant = Tenant(name=f"NoAdmin-{uuid.uuid4().hex[:8]}", schema_name="no_admin")
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        job = _make_job(db, tenant.id)

        with (
            patch("app.services.s3_data_export_service.upload_export_zip", return_value="key") as mock_upload,
            patch("app.services.s3_recording_service.generate_signed_url", return_value="https://signed.example/export.zip"),
            patch("app.services.email_service.email_service.send_data_export_ready_email") as mock_email,
        ):
            run_export_job(db, job)

        db.refresh(job)
        assert job.status == "ready"
        mock_upload.assert_called_once()
        mock_email.assert_not_called()

    def test_gcs_failure_marks_job_error_and_skips_email(self, db):
        tenant = Tenant(name=f"FailExport-{uuid.uuid4().hex[:8]}", schema_name="fail_export")
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        job = _make_job(db, tenant.id)

        with (
            patch("app.services.s3_data_export_service.upload_export_zip", side_effect=RuntimeError("GCS down")),
            patch("app.services.email_service.email_service.send_data_export_ready_email") as mock_email,
        ):
            run_export_job(db, job)

        db.refresh(job)
        assert job.status == "error"
        assert "GCS down" in job.error_message
        mock_email.assert_not_called()


class TestRunDataExportJobTask:
    def test_loads_job_and_delegates(self):
        from app.workers.batch_call_worker import run_data_export_job

        job_id = uuid.uuid4()
        fake_job = MagicMock(id=job_id)
        db = MagicMock()
        db.get.return_value = fake_job

        with (
            patch("app.db.session.SessionLocal") as mock_sl,
            patch("app.services.data_export_service.run_export_job") as mock_run,
        ):
            mock_sl.return_value = db
            import asyncio

            asyncio.run(run_data_export_job({}, str(job_id)))

        mock_run.assert_called_once_with(db, fake_job)
        db.close.assert_called_once()

    def test_missing_job_is_a_noop(self):
        from app.workers.batch_call_worker import run_data_export_job

        db = MagicMock()
        db.get.return_value = None

        with (
            patch("app.db.session.SessionLocal") as mock_sl,
            patch("app.services.data_export_service.run_export_job") as mock_run,
        ):
            mock_sl.return_value = db
            import asyncio

            asyncio.run(run_data_export_job({}, str(uuid.uuid4())))

        mock_run.assert_not_called()
        db.close.assert_called_once()
