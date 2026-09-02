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
from app.core.logger import logger


def _client_kwargs() -> dict:
    """Build kwargs for a boto3 client, preferring explicit static credentials
    but falling back to boto3's own Default Credential Provider Chain (ECS
    task role, EC2 instance profile, IRSA, ~/.aws, etc.) when none are set.
    """
    kwargs: dict = {}

    region_name = (settings.AWS_REGION_NAME or "").strip()
    if region_name:
        kwargs["region_name"] = region_name

    access_key = (settings.AWS_ACCESS_KEY_ID or "").strip()
    secret_key = (settings.AWS_SECRET_ACCESS_KEY or "").strip()
    session_token = (settings.AWS_SESSION_TOKEN or "").strip()

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        # A session token without base static keys is meaningless — only
        # attach it when the static credential pair is also present.
        if session_token:
            kwargs["aws_session_token"] = session_token
    elif access_key or secret_key:
        # Only one of the pair is set — likely a config typo. Silently
        # falling through to the ambient credential chain (task role / IRSA
        # / ~/.aws) could resolve a different AWS identity than intended, so
        # this is worth a log line even though it's not fatal.
        logger.warning(
            "AWS static credentials are only partially configured "
            "(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must both be set) "
            "— falling back to the default credential provider chain."
        )

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


def verify_aws_caller_identity() -> dict | None:
    """Best-effort STS identity check for opportunistic startup logging.

    Never raises — any failure (missing credentials, network error, etc.) is
    logged as a warning and the function returns None. This is purely
    diagnostic and must never gate application startup or request handling.
    """
    try:
        sts_client = boto3.client("sts", **_client_kwargs())
        identity = sts_client.get_caller_identity()
        logger.info(
            "AWS caller identity resolved: arn=%s account=%s",
            identity.get("Arn"),
            identity.get("Account"),
        )
        return identity
    except Exception as exc:  # noqa: BLE001 - intentionally broad, fail-open
        logger.warning("Failed to verify AWS caller identity: %s", exc)
        return None
