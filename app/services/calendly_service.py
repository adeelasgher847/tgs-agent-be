"""
Calendly calendar integration — OAuth 2.0, availability, and booking.

Docs:
  API v2 reference:  https://developer.calendly.com/api-docs
  OAuth 2.0 flow:    https://developer.calendly.com/api-docs/ZG9jOjM2MzE2MDM2-oauth-with-calendly
  Available times:   https://developer.calendly.com/api-docs/ZG9jOjQxMzE3NDY2-get-event-type-available-times
  Create invitee:    https://developer.calendly.com/api-docs/ZG9jOjM4MTEyOTk1-create-invitee
  Get current user:  https://developer.calendly.com/api-docs/ZG9jOjE2MTAzNTY-get-current-user

All slot conflict-checking, timezone handling, and calendar sync are owned by
Calendly — this module only exchanges/refreshes OAuth tokens and proxies the
availability + booking API calls.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db_encryption import decrypt_calendly_token, encrypt_calendly_token
from app.core.logger import logger
from app.models.calendly_integration import CalendlyIntegration

AUTHORIZE_URL = "https://auth.calendly.com/oauth/authorize"
TOKEN_URL = "https://auth.calendly.com/oauth/token"
API_BASE_URL = "https://api.calendly.com"

_STATE_PURPOSE = "calendly_oauth_state"
_STATE_TTL_MINUTES = 10

# Refresh the access token proactively when it expires within this window.
_TOKEN_REFRESH_MARGIN_SECONDS = 300  # 5 minutes, per spec

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0


async def _request_with_backoff(method: str, url: str, **kwargs) -> httpx.Response:
    """Call Calendly with exponential backoff on 429 (1s, 2s, 4s, 8s, 16s).

    Mirrors app/services/hubspot_service.py::_request_with_backoff for the
    same class of OAuth-integration HTTP call.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        attempt = 0
        while True:
            response = await client.request(method, url, **kwargs)
            if response.status_code != 429 or attempt >= _MAX_RETRIES:
                return response

            retry_after_header = response.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after_header)
                if retry_after_header
                else _BASE_BACKOFF_SECONDS * (2 ** attempt)
            )
            logger.warning(
                "Calendly 429 on %s %s; retrying in %.1fs (attempt %d/%d)",
                method,
                url,
                wait_seconds,
                attempt + 1,
                _MAX_RETRIES,
            )
            await asyncio.sleep(wait_seconds)
            attempt += 1


# ─── OAuth state (signed, carries workspace_id through the redirect round-trip) ──


def build_oauth_state(workspace_id: uuid.UUID) -> str:
    payload = {
        "workspace_id": str(workspace_id),
        "purpose": _STATE_PURPOSE,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=_STATE_TTL_MINUTES),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_oauth_state(state: str) -> uuid.UUID:
    """Raises ValueError if state is missing, expired, tampered, or malformed."""
    try:
        payload = jwt.decode(state, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        raise ValueError("Invalid or expired OAuth state") from exc

    if payload.get("purpose") != _STATE_PURPOSE:
        raise ValueError("Invalid OAuth state purpose")

    workspace_id_str = payload.get("workspace_id")
    if not workspace_id_str:
        raise ValueError("OAuth state missing workspace_id")

    try:
        return uuid.UUID(workspace_id_str)
    except ValueError as exc:
        raise ValueError("OAuth state has malformed workspace_id") from exc


def get_redirect_uri() -> str:
    return settings.CALENDLY_REDIRECT_URI or (
        f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/api/v2/integrations/calendly/callback"
    )


def build_authorization_url(state: str) -> str:
    params = {
        "client_id": settings.CALENDLY_CLIENT_ID,
        "redirect_uri": get_redirect_uri(),
        "response_type": "code",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ─── Token exchange / refresh ─────────────────────────────────────────────────


async def exchange_code_for_tokens(code: str) -> dict:
    response = await _request_with_backoff(
        "POST",
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": settings.CALENDLY_CLIENT_ID,
            "client_secret": settings.CALENDLY_CLIENT_SECRET,
            "redirect_uri": get_redirect_uri(),
            "code": code,
        },
    )
    response.raise_for_status()
    return response.json()


async def refresh_access_token(refresh_token: str) -> dict:
    response = await _request_with_backoff(
        "POST",
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": settings.CALENDLY_CLIENT_ID,
            "client_secret": settings.CALENDLY_CLIENT_SECRET,
            "refresh_token": refresh_token,
        },
    )
    response.raise_for_status()
    return response.json()


async def get_current_user(access_token: str) -> dict:
    """GET /users/me — used right after token exchange to resolve calendly_user_uri."""
    response = await _request_with_backoff(
        "GET",
        f"{API_BASE_URL}/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    response.raise_for_status()
    return response.json()


# ─── CalendlyIntegration CRUD ──────────────────────────────────────────────────


def get_integration(db: Session, workspace_id: uuid.UUID) -> Optional[CalendlyIntegration]:
    return (
        db.query(CalendlyIntegration)
        .filter(CalendlyIntegration.workspace_id == workspace_id)
        .first()
    )


def tenant_has_calendly_connected(db: Session, workspace_id: uuid.UUID) -> bool:
    return get_integration(db, workspace_id) is not None


def upsert_tokens(
    db: Session,
    workspace_id: uuid.UUID,
    token_response: dict,
    *,
    calendly_user_uri: Optional[str] = None,
    connected_by_user_id: Optional[uuid.UUID] = None,
    calendly_event_type_uri: Optional[str] = None,
) -> CalendlyIntegration:
    access_token = token_response["access_token"]
    refresh_token = token_response.get("refresh_token")
    expires_in = int(token_response.get("expires_in", 7200))  # Calendly default: 2 hours
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    row = get_integration(db, workspace_id)
    if row is None:
        row = CalendlyIntegration(workspace_id=workspace_id)

    row.access_token = encrypt_calendly_token(access_token, db)
    if refresh_token:
        row.refresh_token = encrypt_calendly_token(refresh_token, db)
    row.token_expires_at = expires_at
    if calendly_user_uri is not None:
        row.calendly_user_uri = calendly_user_uri
    if connected_by_user_id is not None:
        row.connected_by_user_id = connected_by_user_id
    if calendly_event_type_uri is not None:
        row.calendly_event_type_uri = calendly_event_type_uri

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def get_valid_access_token(db: Session, workspace_id: uuid.UUID) -> Optional[str]:
    """
    Return a usable access token, refreshing it first if it expires within
    _TOKEN_REFRESH_MARGIN_SECONDS (5 minutes) — checked before every Calendly API call.
    """
    row = get_integration(db, workspace_id)
    if row is None or not row.access_token:
        return None

    expires_at = row.token_expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if expires_at is not None and expires_at > now + timedelta(seconds=_TOKEN_REFRESH_MARGIN_SECONDS):
        return decrypt_calendly_token(row.access_token, db)

    if not row.refresh_token:
        logger.warning(
            "Calendly access token expiring for workspace=%s but no refresh_token stored",
            workspace_id,
        )
        return decrypt_calendly_token(row.access_token, db) if expires_at and expires_at > now else None

    try:
        refresh_token_plain = decrypt_calendly_token(row.refresh_token, db)
        token_response = await refresh_access_token(refresh_token_plain)
    except Exception:
        logger.warning(
            "Calendly token refresh failed for workspace=%s", workspace_id, exc_info=True
        )
        return None

    row = upsert_tokens(db, workspace_id, token_response)
    return decrypt_calendly_token(row.access_token, db)


async def disconnect(db: Session, workspace_id: uuid.UUID) -> bool:
    """Revoke the access token at Calendly (best-effort) and delete the local row."""
    row = get_integration(db, workspace_id)
    if row is None:
        return False

    if row.access_token:
        try:
            access_token_plain = decrypt_calendly_token(row.access_token, db)
            await _request_with_backoff(
                "POST",
                f"{TOKEN_URL}/revoke",
                data={
                    "client_id": settings.CALENDLY_CLIENT_ID,
                    "client_secret": settings.CALENDLY_CLIENT_SECRET,
                    "token": access_token_plain,
                },
            )
        except Exception:
            logger.warning(
                "Calendly token revoke failed (continuing with local disconnect) workspace=%s",
                workspace_id,
                exc_info=True,
            )

    db.delete(row)
    db.commit()
    return True


# ─── Availability ──────────────────────────────────────────────────────────────


async def get_available_slots(
    db: Session,
    workspace_id: uuid.UUID,
    date_from: datetime,
    date_to: datetime,
) -> list[dict]:
    """
    Fetch bookable slots from Calendly for the connected event type.

    Returns a normalized list: [{slot_start, slot_end, available}].
    Raises ValueError if Calendly is not connected or no event type is selected.
    """
    row = get_integration(db, workspace_id)
    if row is None:
        raise ValueError("Calendly is not connected for this workspace")
    if not row.calendly_event_type_uri:
        raise ValueError("No Calendly event type configured for this workspace")

    access_token = await get_valid_access_token(db, workspace_id)
    if not access_token:
        raise ValueError("Calendly access token unavailable — reconnect required")

    def _iso(dt_val: datetime) -> str:
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    response = await _request_with_backoff(
        "GET",
        f"{API_BASE_URL}/event_type_available_times",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "event_type": row.calendly_event_type_uri,
            "start_time": _iso(date_from),
            "end_time": _iso(date_to),
        },
    )
    response.raise_for_status()
    payload = response.json()

    slots: list[dict] = []
    for item in payload.get("collection", []):
        start_time = item.get("start_time")
        if not start_time:
            continue
        try:
            slot_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        status = (item.get("status") or "available").lower()
        slots.append(
            {
                "slot_start": slot_start,
                "slot_end": item.get("end_time"),
                "available": status == "available",
            }
        )
    return slots


# ─── Booking ───────────────────────────────────────────────────────────────────


async def book_appointment(
    db: Session,
    workspace_id: uuid.UUID,
    *,
    start_time: datetime,
    attendee_email: str,
    attendee_name: str,
    description: Optional[str] = None,
) -> dict:
    """
    Schedule an appointment on Calendly via POST /invitees. Injects the voice
    call summary (if any) into the invitee's comments/description field.
    """
    row = get_integration(db, workspace_id)
    if row is None:
        raise ValueError("Calendly is not connected for this workspace")
    if not row.calendly_event_type_uri:
        raise ValueError("No Calendly event type configured for this workspace")

    access_token = await get_valid_access_token(db, workspace_id)
    if not access_token:
        raise ValueError("Calendly access token unavailable — reconnect required")

    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    start_time_iso_utc = start_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    payload = {
        "event_type": row.calendly_event_type_uri,
        "start_time": start_time_iso_utc,
        "invitee": {
            "email": attendee_email,
            "name": attendee_name,
            "timezone": "UTC",
        },
    }
    if description:
        payload["invitee"]["comments"] = description

    response = await _request_with_backoff(
        "POST",
        f"{API_BASE_URL}/invitees",
        headers={"Authorization": f"Bearer {access_token}"},
        json=payload,
    )
    response.raise_for_status()
    return response.json()
