"""
GCS Recording Service — upload call recordings and generate signed URLs.

Credentials are loaded via ADC (GOOGLE_APPLICATION_CREDENTIALS) using the
same pattern as GoogleSttService and VertexGeminiService.

# Infra: GCS lifecycle rule deletes recordings/ prefix objects after 90 days.
# Apply the following lifecycle condition to the bucket via Terraform or gcloud:
#   condition: { age: 90, matchesPrefix: ["recordings/"] }
#   action: { type: "Delete" }
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
    *,
    kms_key_name: Optional[str] = None,
) -> str:
    """
    Upload an Opus recording to GCS and set custom metadata.

    When *kms_key_name* is provided (HIPAA flows) the object is encrypted with
    the tenant's Customer-Managed Encryption Key (CMEK) via Cloud KMS.

    Returns the full GCS path (key) after a successful upload.
    Raises on any GCS error — caller is responsible for error handling.
    """
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    blob = bucket.blob(key, kms_key_name=kms_key_name)

    # Custom metadata: callId, workspaceId, agentId, duration
    blob.metadata = {str(k): str(v) for k, v in metadata.items() if v is not None}
    blob.upload_from_string(file_bytes, content_type=content_type)

    if kms_key_name:
        logger.info(
            "GCS CMEK upload complete: gs://%s/%s (%d bytes) kms=%s",
            settings.GCS_RECORDINGS_BUCKET, key, len(file_bytes), kms_key_name,
        )
    else:
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


def set_bucket_default_kms_key(kms_key_name: str) -> None:
    """
    Set the GCS recordings bucket's default Cloud KMS key for CMEK encryption.

    Once set, every new object written to the bucket — including recordings
    uploaded directly by LiveKit — is encrypted with this key automatically,
    without any application-level changes to the upload path.

    Requires the GCS service account to have the Cloud KMS CryptoKey Encrypter/
    Decrypter role on the specified key.

    Raises on any GCS or KMS API error; caller is responsible for error handling.
    """
    from google.cloud import storage  # type: ignore

    client = _get_gcs_client()
    bucket = client.get_bucket(settings.GCS_RECORDINGS_BUCKET)
    bucket.default_kms_key_name = kms_key_name
    bucket.patch()
    logger.info(
        "GCS bucket %s default KMS key set to %s",
        settings.GCS_RECORDINGS_BUCKET,
        kms_key_name,
    )


def delete_workspace_recordings(workspace_id: uuid.UUID) -> int:
    """
    Delete every recording object under recordings/{workspace_id}/ in GCS.

    Used by the GDPR account-deletion flow. Best-effort: caller decides
    whether a failure here should block the (otherwise irreversible) erasure
    response. Returns the number of objects deleted.
    """
    client = _get_gcs_client()
    bucket = client.bucket(settings.GCS_RECORDINGS_BUCKET)
    prefix = f"{settings.GCS_RECORDINGS_PREFIX}/{workspace_id}/"

    deleted = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        blob.delete()
        deleted += 1

    logger.info(
        "GCS workspace recordings deleted: workspace=%s prefix=%s count=%d",
        workspace_id, prefix, deleted,
    )
    return deleted


def generate_signed_url(
    gcs_path: str,
    expiry_seconds: int = settings.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
) -> str:
    """
    Generate a short-lived V4 signed URL for direct client download.

    Requires a service account with roles/storage.objectViewer.
    Uses service account credentials from GOOGLE_APPLICATION_CREDENTIALS (ADC).
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
