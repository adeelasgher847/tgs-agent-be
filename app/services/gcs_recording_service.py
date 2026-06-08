"""
GCS Recording Service — upload call recordings and generate signed URLs.

Credentials are loaded via ADC (GOOGLE_APPLICATION_CREDENTIALS) using the
same pattern as GoogleSttService and VertexGeminiService.

# Infra: GCS lifecycle rule deletes recordings/ prefix objects after 90 days.
# Apply the following lifecycle condition to the bucket via Terraform or gcloud:
#   condition: { age: 90, matchesPrefix: ["recordings/"] }
#   action: { type: "Delete" }

# Sprint 5: use CMEK encryption key for HIPAA tenants (customer-managed encryption).
# Wire the key ARN via GCS_CMEK_KEY_NAME env var and pass kms_key_name to blob.rewrite().
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

from app.core.config import settings
from app.core.logger import logger


def _get_gcs_client():
    """Return an authenticated GCS client using ADC."""
    from google.cloud import storage  # type: ignore

    return storage.Client()


def build_gcs_key(workspace_id: uuid.UUID, call_id: uuid.UUID, end_time: Optional[datetime.datetime] = None) -> str:
    """Return the canonical GCS object key for a call recording."""
    date_str = (end_time or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y%m%d")
    return f"{settings.GCS_RECORDINGS_PREFIX}/{workspace_id}/{call_id}/{date_str}.opus"


def upload_recording(
    key: str,
    file_bytes: bytes,
    metadata: dict,
    content_type: str = "audio/ogg; codecs=opus",
) -> str:
    """
    Upload an Opus recording to GCS and set custom metadata.

    Returns the full GCS path (key) after a successful upload.
    Raises on any GCS error — caller is responsible for error handling.
    """
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.blob(key)

    # Custom metadata: callId, workspaceId, agentId, duration
    blob.metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
    blob.upload_from_string(file_bytes, content_type=content_type)

    logger.info("GCS upload complete: gs://%s/%s (%d bytes)", settings.GCS_RECORDINGS_BUCKET, key, len(file_bytes))
    return key


def update_object_metadata(key: str, metadata: dict) -> None:
    """Patch custom metadata on an already-uploaded GCS object."""
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.get_blob(key)
    if blob is None:
        logger.warning("GCS patch_metadata: object not found: %s", key)
        return
    blob.metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
    blob.patch()
    logger.debug("GCS metadata updated: %s", key)


def get_object_size(key: str) -> Optional[int]:
    """Return the size in bytes of a GCS object, or None if not found."""
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.get_blob(key)
    return blob.size if blob else None


def generate_signed_url(
    gcs_path: str,
    expiry_seconds: int = settings.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
) -> str:
    """
    Generate a short-lived V4 signed URL for direct client download.

    Requires a service account with roles/storage.objectViewer.
    Uses service account credentials from GOOGLE_APPLICATION_CREDENTIALS (ADC).

    # Sprint 5: for HIPAA tenants, ensure the bucket uses CMEK encryption.
    """
    import google.auth  # type: ignore
    from google.auth.transport import requests as google_requests  # type: ignore
    from google.cloud import storage  # type: ignore

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    # Refresh credentials so we have a valid access token for signing.
    credentials.refresh(google_requests.Request())

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.blob(gcs_path)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(seconds=expiry_seconds),
        method="GET",
        credentials=credentials,
    )
    return url
