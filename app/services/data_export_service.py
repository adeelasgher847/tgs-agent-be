"""
GDPR data-export orchestration: job lifecycle (create / lookup / status
transitions) plus the actual export run invoked by the ARQ worker.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.data_export_job import DataExportJob
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association


def create_export_job(
    db: Session, workspace_id: uuid.UUID, requested_by_user_id: Optional[uuid.UUID]
) -> DataExportJob:
    job = DataExportJob(
        workspace_id=workspace_id,
        requested_by_user_id=requested_by_user_id,
        status="processing",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_export_job(
    db: Session, workspace_id: uuid.UUID, job_id: uuid.UUID
) -> Optional[DataExportJob]:
    return db.execute(
        select(DataExportJob).where(
            DataExportJob.id == job_id,
            DataExportJob.workspace_id == workspace_id,
        )
    ).scalar_one_or_none()


def _get_workspace_admin_email(db: Session, workspace_id: uuid.UUID) -> Optional[str]:
    row = db.execute(
        select(User.email)
        .join(user_tenant_association, User.id == user_tenant_association.c.user_id)
        .join(Role, Role.id == user_tenant_association.c.role_id)
        .where(
            user_tenant_association.c.tenant_id == workspace_id,
            Role.name == "admin",
            User.deleted_at.is_(None),
        )
        .limit(1)
    ).first()
    return row[0] if row else None


def run_export_job(db: Session, job: DataExportJob) -> None:
    """
    Build the export ZIP, upload it to GCS, email the signed URL to the
    workspace admin, and update the job's status accordingly.

    Never raises — failures are captured on the job row (status='error')
    so the ARQ worker doesn't need its own try/except around this call.
    """
    from app.services import gcs_data_export_service
    from app.services.data_export_zip_builder import build_export_zip
    from app.services.email_service import email_service

    zip_path: Optional[str] = None
    try:
        zip_path = build_export_zip(db, job.workspace_id)

        key = gcs_data_export_service.build_export_gcs_key(job.workspace_id, job.id)
        gcs_data_export_service.upload_export_zip(zip_path, key)

        from app.services.gcs_recording_service import generate_signed_url

        signed_url = generate_signed_url(
            key, expiry_seconds=gcs_data_export_service.DATA_EXPORT_SIGNED_URL_EXPIRY_SECONDS
        )

        job.status = "ready"
        job.gcs_path = key
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        admin_email = _get_workspace_admin_email(db, job.workspace_id)
        tenant = db.get(Tenant, job.workspace_id)
        if admin_email:
            email_service.send_data_export_ready_email(
                admin_email, signed_url, tenant.name if tenant else str(job.workspace_id)
            )
        else:
            logger.warning(
                "data_export_job %s: no admin user found for workspace %s — export ready but no email sent",
                job.id, job.workspace_id,
            )

    except Exception as exc:
        logger.error(
            "data_export_job %s failed: %s", job.id, exc, exc_info=True
        )
        db.rollback()
        job = db.get(DataExportJob, job.id)
        if job is not None:
            job.status = "error"
            job.error_message = str(exc)[:1000]
            db.commit()
    finally:
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
