"""
GCS upload for GDPR workspace data-export ZIPs.

Mirrors gcs_recording_service.py / batch_call_gcs_service.py: ADC via
GOOGLE_APPLICATION_CREDENTIALS, lazy client init.

GCS path layout: data-exports/{workspace_id}/{job_id}.zip

The ZIP itself is built on local disk by data_export_zip_builder (streaming
writes, never holding the full export in memory). upload_export_zip then
streams that file to GCS via upload_from_filename, which reads it in chunks
rather than loading it into memory.
"""
from __future__ import annotations

import uuid

from app.core.config import settings
from app.core.logger import logger

DATA_EXPORT_SIGNED_URL_EXPIRY_SECONDS = 24 * 60 * 60  # 24 hours, per ticket


def build_export_gcs_key(workspace_id: uuid.UUID, job_id: uuid.UUID) -> str:
    return f"data-exports/{workspace_id}/{job_id}.zip"


def _get_gcs_client():
    from google.cloud import storage  # type: ignore

    return storage.Client()


def upload_export_zip(local_zip_path: str, key: str) -> str:
    """
    Upload a locally-built ZIP file to GCS by streaming it from disk.

    Returns the GCS key on success; raises on any GCS error.
    """
    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.blob(key)
    blob.upload_from_filename(local_zip_path, content_type="application/zip")

    logger.info("GCS data-export upload complete: gs://%s/%s", settings.GCS_RECORDINGS_BUCKET, key)
    return key
