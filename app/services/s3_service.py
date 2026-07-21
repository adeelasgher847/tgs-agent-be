"""
Shared AWS client factory.

Every service that talks to an AWS API must obtain its client via one of
the get_*_client() helpers below rather than instantiating boto3 directly,
so credentials and region configuration stay centralized in one place.

Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
      https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ses.html
"""
from __future__ import annotations

import boto3

from app.core.config import settings


def _client_kwargs() -> dict:
    kwargs = {"region_name": settings.AWS_REGION_NAME}
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    return kwargs


def get_s3_client():
    """Return a boto3 S3 client configured from application settings."""
    return boto3.client("s3", **_client_kwargs())


def get_kms_client():
    """Return a boto3 KMS client configured from application settings."""
    return boto3.client("kms", **_client_kwargs())


def get_ses_client():
    """Return a boto3 SES client configured from application settings."""
    return boto3.client("ses", **_client_kwargs())
