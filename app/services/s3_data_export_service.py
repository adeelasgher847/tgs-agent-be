"""
S3 upload for GDPR workspace data-export ZIPs.

The `gcs_path` column on the export job row is unchanged — it now stores
an S3 object key.

S3 path layout: data-exports/{workspace_id}/{job_id}.zip

Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
"""
from __future__ import annotations

import uuid

from app.core.config import settings
from app.core.logger import logger
from app.services.s3_service import get_s3_client

DATA_EXPORT_SIGNED_URL_EXPIRY_SECONDS = 24 * 60 * 60  # 24 hours, per ticket


def build_export_gcs_key(workspace_id: uuid.UUID, job_id: uuid.UUID) -> str:
    return f"data-exports/{workspace_id}/{job_id}.zip"


def upload_export_zip(local_zip_path: str, key: str) -> str:
    """
    Upload a locally-built ZIP file to S3 by streaming it from disk.

    Returns the S3 key on success; raises on any S3 error.
    """
    client = get_s3_client()
    client.upload_file(
        local_zip_path,
        settings.S3_RECORDINGS_BUCKET,
        key,
        ExtraArgs={"ContentType": "application/zip"},
    )

    logger.info("S3 data-export upload complete: s3://%s/%s", settings.S3_RECORDINGS_BUCKET, key)
    return key
