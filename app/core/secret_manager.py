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


@lru_cache(maxsize=1)
def _load_livekit_production_credentials() -> tuple[str, str, str]:
    """Fetch LiveKit credentials from Secret Manager (cached after first call)."""
    url = _fetch_from_secret_manager("LIVEKIT_URL") or settings.LIVEKIT_URL
    api_key = _fetch_from_secret_manager("LIVEKIT_API_KEY") or settings.LIVEKIT_API_KEY
    api_secret = _fetch_from_secret_manager("LIVEKIT_API_SECRET") or settings.LIVEKIT_API_SECRET
    if not url or not api_key or not api_secret:
        raise RuntimeError(
            "LiveKit credentials unavailable. Set GCP_PROJECT_ID and Secret Manager secrets "
            "(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) or the equivalent env vars."
        )
    return url, api_key, api_secret


def get_livekit_credentials() -> tuple[str, str, str]:
    """
    Return (url, api_key, api_secret) appropriate for the current environment.

    - production/staging → GCP Secret Manager with env-var fallback
      (secrets injected as env vars in both LiveKit GKE Deployment and API server Deployment)
    - development        → env-var / .env values

    Raises RuntimeError in staging/production when credentials are absent.
    Raises ValueError in development when credentials are absent.
    """
    env = settings.ENVIRONMENT.lower()

    if env in ("production", "staging"):
        url, api_key, api_secret = _load_livekit_production_credentials()
        if not url or not api_key or not api_secret:
            raise RuntimeError(
                f"LiveKit credentials unavailable in {env}. "
                "Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET via Secret Manager or env vars."
            )
        return url, api_key, api_secret

    # development — plain env vars / .env
    url = settings.LIVEKIT_URL
    api_key = settings.LIVEKIT_API_KEY
    api_secret = settings.LIVEKIT_API_SECRET
    if not url or not api_key or not api_secret:
        raise ValueError(
            "LiveKit credentials not configured. Add LIVEKIT_URL, LIVEKIT_API_KEY, "
            "and LIVEKIT_API_SECRET to your .env file for local development."
        )
    return url, api_key, api_secret


@lru_cache(maxsize=1)
def get_rime_api_key() -> str:
    """
    Return the Rime TTS API key appropriate for the current environment.

    - production/staging → GCP Secret Manager (secret name: RIME_API_KEY), with
      env-var fallback so deployments that set the var directly still work.
    - development → plain env-var / .env value.

    Raises ValueError (development) or RuntimeError (staging/production) if no
    key is available so callers fail at startup rather than sending unauthenticated
    requests mid-call.
    """
    env = settings.ENVIRONMENT.lower()

    if env in ("production", "staging"):
        secret_key = _fetch_from_secret_manager("RIME_API_KEY")
        if secret_key:
            return secret_key
        # Fall back to env var (covers deployments that inject the var via CI/CD).
        env_key = getattr(settings, "RIME_API_KEY", "") or ""
        if env_key.strip():
            return env_key.strip()
        raise RuntimeError(
            f"RIME_API_KEY unavailable in {env}. "
            "Set it in GCP Secret Manager (secret: RIME_API_KEY) or as an env var."
        )

    # development — plain env var / .env
    env_key = getattr(settings, "RIME_API_KEY", "") or ""
    if not env_key.strip():
        raise ValueError(
            "RIME_API_KEY is not set. Add it to your .env file for local development."
        )
    return env_key.strip()
