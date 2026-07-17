"""
Streaming S3 upload for batch CSV files.

S3 path layout: batch-files/{workspace_id}/{batch_id}.csv

Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
"""
from __future__ import annotations

import io
import uuid
from typing import Iterator, Union

from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logger import logger
from app.services.s3_service import get_s3_client


def build_batch_csv_gcs_key(workspace_id: uuid.UUID, batch_id: uuid.UUID) -> str:
    return f"batch-files/{workspace_id}/{batch_id}.csv"


def upload_batch_csv(
    key: str,
    data: Union[bytes, Iterator[bytes]],
    workspace_id: uuid.UUID,
    batch_id: uuid.UUID,
) -> str:
    """
    Upload a CSV to S3. Accepts either raw bytes or a byte iterator.

    Returns the S3 key on success; raises on any S3 error.
    """
    client = get_s3_client()
    bucket = settings.S3_RECORDINGS_BUCKET
    metadata = {
        "workspaceId": str(workspace_id),
        "batchId": str(batch_id),
        "contentType": "text/csv",
    }

    if isinstance(data, bytes):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="text/csv",
            Metadata=metadata,
        )
        logger.info(
            "S3 batch CSV upload complete: s3://%s/%s (%d bytes)",
            bucket,
            key,
            len(data),
        )
    else:
        buf = io.BytesIO(b"".join(data))
        client.upload_fileobj(
            buf,
            bucket,
            key,
            ExtraArgs={"ContentType": "text/csv", "Metadata": metadata},
        )
        logger.info(
            "S3 batch CSV streaming upload complete: s3://%s/%s",
            bucket,
            key,
        )

    return key


def delete_batch_csv(key: str) -> None:
    """Delete a batch CSV from S3. No-op if the object does not exist."""
    client = get_s3_client()
    bucket = settings.S3_RECORDINGS_BUCKET
    try:
        client.delete_object(Bucket=bucket, Key=key)
        logger.info("S3 batch CSV deleted: %s", key)
    except ClientError as exc:
        logger.warning("S3 delete skipped (not found): %s — %s", key, exc)
