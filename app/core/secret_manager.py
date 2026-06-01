"""
Secret Manager — GCP Secret Manager client with local-dev fallback.

Usage:
    from app.core.secret_manager import get_twilio_credentials
    account_sid, auth_token = get_twilio_credentials()

In staging (ENVIRONMENT="staging") the test credentials are returned automatically
so no real Twilio numbers are ever purchased in a non-production environment.

In production (ENVIRONMENT="production") credentials are fetched from GCP Secret Manager.
Secret names expected:
    TWILIO_ACCOUNT_SID   →  projects/{project}/secrets/TWILIO_ACCOUNT_SID/versions/latest
    TWILIO_AUTH_TOKEN    →  projects/{project}/secrets/TWILIO_AUTH_TOKEN/versions/latest

In development the env-var / .env values are returned as-is (no Secret Manager call).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from app.core.config import settings
from app.core.logger import logger


def _fetch_from_secret_manager(secret_id: str) -> Optional[str]:
    """Fetch a single secret value from GCP Secret Manager."""
    if not settings.GCP_PROJECT_ID:
        return None
    try:
        from google.api_core.exceptions import NotFound, PermissionDenied
        from google.cloud import secretmanager  # type: ignore

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{settings.GCP_PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8").strip()
    except (PermissionDenied, NotFound) as exc:
        logger.error("Secret Manager permanent failure for %s: %s", secret_id, exc)
        return None
    except Exception as exc:
        logger.warning("Secret Manager transient failure for %s: %s", secret_id, exc)
        return None


@lru_cache(maxsize=1)
def _load_production_credentials() -> Tuple[str, str]:
    """Fetch real Twilio credentials from Secret Manager (cached after first call)."""
    sid = _fetch_from_secret_manager("TWILIO_ACCOUNT_SID") or settings.TWILIO_ACCOUNT_SID
    token = _fetch_from_secret_manager("TWILIO_AUTH_TOKEN") or settings.TWILIO_AUTH_TOKEN
    if not sid or not token:
        raise RuntimeError(
            "Twilio credentials unavailable. Set GCP_PROJECT_ID and Secret Manager secrets "
            "or TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN env vars for local development."
        )
    return sid, token


def get_twilio_credentials() -> Tuple[str, str]:
    """
    Return (account_sid, auth_token) appropriate for the current environment.

    - staging   → TWILIO_TEST_ACCOUNT_SID / TWILIO_TEST_AUTH_TOKEN (no real purchases)
    - production → GCP Secret Manager (with env-var fallback)
    - development → env-var / .env values
    """
    env = settings.ENVIRONMENT.lower()

    if env == "staging":
        sid = settings.TWILIO_TEST_ACCOUNT_SID
        token = settings.TWILIO_TEST_AUTH_TOKEN
        if not sid or not token:
            raise RuntimeError(
                "ENVIRONMENT=staging but TWILIO_TEST_ACCOUNT_SID / TWILIO_TEST_AUTH_TOKEN "
                "are not set. Configure them via Secret Manager or .env.staging."
            )
        return sid, token

    if env == "production":
        return _load_production_credentials()

    # development — plain env vars
    sid = settings.TWILIO_ACCOUNT_SID
    token = settings.TWILIO_AUTH_TOKEN
    if not sid or not token:
        raise RuntimeError(
            "Twilio credentials not found. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN "
            "in your .env file for local development."
        )
    return sid, token


def is_staging() -> bool:
    return settings.ENVIRONMENT.lower() == "staging"
