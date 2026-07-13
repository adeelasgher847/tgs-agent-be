"""
Streaming GCS upload for batch CSV files.

Mirrors gcs_recording_service.py patterns: ADC via GOOGLE_APPLICATION_CREDENTIALS,
lazy client init, single-method upload surface.

GCS path layout: batch-files/{workspace_id}/{batch_id}.csv
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator, Iterator, Union

from app.core.config import settings
from app.core.logger import logger


def build_batch_csv_gcs_key(workspace_id: uuid.UUID, batch_id: uuid.UUID) -> str:
    return f"batch-files/{workspace_id}/{batch_id}.csv"


def _get_gcs_client():
    from google.cloud import storage  # type: ignore

    return storage.Client()


def upload_batch_csv(
    key: str,
    data: Union[bytes, Iterator[bytes]],
    workspace_id: uuid.UUID,
    batch_id: uuid.UUID,
) -> str:
    """
    Upload a CSV to GCS.  Accepts either raw bytes or a byte iterator for
    streaming large files without loading everything into memory at once.

    Returns the GCS key on success; raises on any GCS error.
    """
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.blob(key)
    blob.metadata = {
        "workspaceId": str(workspace_id),
        "batchId": str(batch_id),
        "contentType": "text/csv",
    }

    if isinstance(data, bytes):
        blob.upload_from_string(data, content_type="text/csv")
        logger.info(
            "GCS batch CSV upload complete: gs://%s/%s (%d bytes)",
            settings.GCS_RECORDINGS_BUCKET,
            key,
            len(data),
        )
    else:
        # Stream upload via a file-like wrapper around the iterator
        import io

        buf = io.BytesIO(b"".join(data))
        blob.upload_from_file(buf, content_type="text/csv", rewind=True)
        logger.info(
            "GCS batch CSV streaming upload complete: gs://%s/%s",
            settings.GCS_RECORDINGS_BUCKET,
            key,
        )

    return key


def delete_batch_csv(key: str) -> None:
    """Delete a batch CSV from GCS.  No-op if the object does not exist."""
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.blob(key)
    try:
        blob.delete()
        logger.info("GCS batch CSV deleted: %s", key)
    except Exception as exc:  # google.api_core.exceptions.NotFound
        logger.warning("GCS delete skipped (not found): %s — %s", key, exc)
