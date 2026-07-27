"""
S3 Recording Service — upload call recordings and generate signed URLs.

The `recording_s3_path` / `s3_path` DB columns store S3 object keys.

Docs:
- https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logger import logger
from app.services.s3_service import get_s3_client


def build_s3_key(workspace_id: uuid.UUID, call_id: uuid.UUID, end_time: Optional[datetime.datetime] = None) -> str:
    """Return the canonical S3 object key for a call recording."""
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
    Upload an Opus recording to S3 and set custom metadata.

    When *kms_key_name* is provided (HIPAA flows) the object is encrypted with
    the tenant's Customer-Managed Key (CMK) via SSE-KMS.

    Returns the object key after a successful upload.
    Raises on any S3 error — caller is responsible for error handling.
    """
    client = get_s3_client()

    put_kwargs = {
        "Bucket": settings.S3_RECORDINGS_BUCKET,
        "Key": key,
        "Body": file_bytes,
        "ContentType": content_type,
        "Metadata": {str(k): str(v) for k, v in metadata.items() if v is not None},
    }
    if kms_key_name:
        put_kwargs["ServerSideEncryption"] = "aws:kms"
        put_kwargs["SSEKMSKeyId"] = kms_key_name

    client.put_object(**put_kwargs)

    if kms_key_name:
        logger.info(
            "S3 CMK upload complete: s3://%s/%s (%d bytes) kms=%s",
            settings.S3_RECORDINGS_BUCKET, key, len(file_bytes), kms_key_name,
        )
    else:
        logger.info("S3 upload complete: s3://%s/%s (%d bytes)", settings.S3_RECORDINGS_BUCKET, key, len(file_bytes))
    return key


def update_object_metadata(key: str, metadata: dict) -> None:
    """Replace custom metadata on an already-uploaded S3 object via a self-copy."""
    client = get_s3_client()
    bucket = settings.S3_RECORDINGS_BUCKET

    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            logger.warning("S3 update_object_metadata: object not found: %s", key)
            return
        raise

    client.copy_object(
        Bucket=bucket,
        Key=key,
        CopySource={"Bucket": bucket, "Key": key},
        Metadata={str(k): str(v) for k, v in metadata.items() if v is not None},
        MetadataDirective="REPLACE",
        ContentType=head.get("ContentType", "application/octet-stream"),
    )
    logger.debug("S3 metadata updated: %s", key)


def get_object_size(key: str) -> Optional[int]:
    """Return the size in bytes of an S3 object, or None if not found."""
    client = get_s3_client()
    try:
        head = client.head_object(Bucket=settings.S3_RECORDINGS_BUCKET, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise
    return head.get("ContentLength")


def set_bucket_default_kms_key(kms_key_name: str) -> None:
    """
    Set the S3 recordings bucket's default SSE-KMS key.

    Once set, every new object written to the bucket is encrypted with this
    key by default, without any application-level changes to the upload path.

    Requires the calling principal to have s3:PutEncryptionConfiguration on
    the bucket and kms:GenerateDataKey / kms:Decrypt on the key.

    Raises on any S3 or KMS API error; caller is responsible for error handling.
    """
    client = get_s3_client()
    client.put_bucket_encryption(
        Bucket=settings.S3_RECORDINGS_BUCKET,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "aws:kms",
                        "KMSMasterKeyID": kms_key_name,
                    },
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )
    logger.info(
        "S3 bucket %s default KMS key set to %s",
        settings.S3_RECORDINGS_BUCKET,
        kms_key_name,
    )


def delete_workspace_recordings(workspace_id: uuid.UUID) -> int:
    """
    Delete every recording object under recordings/{workspace_id}/ in S3.

    Used by the GDPR account-deletion flow. Best-effort: caller decides
    whether a failure here should block the (otherwise irreversible) erasure
    response. Returns the number of objects deleted.
    """
    client = get_s3_client()
    bucket = settings.S3_RECORDINGS_BUCKET
    prefix = f"{settings.GCS_RECORDINGS_PREFIX}/{workspace_id}/"

    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        keys = [{"Key": obj["Key"]} for obj in contents]
        client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        deleted += len(keys)

    logger.info(
        "S3 workspace recordings deleted: workspace=%s prefix=%s count=%d",
        workspace_id, prefix, deleted,
    )
    return deleted


def generate_signed_url(
    gcs_path: str,
    expiry_seconds: int = settings.GCS_RECORDINGS_SIGNED_URL_EXPIRY_SECONDS,
) -> str:
    """
    Generate a short-lived presigned URL for direct client download.

    Parameter name kept as `gcs_path` for signature compatibility with the
    GCS implementation it replaces; it holds the S3 object key.
    """
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_RECORDINGS_BUCKET, "Key": gcs_path},
        ExpiresIn=expiry_seconds,
    )
