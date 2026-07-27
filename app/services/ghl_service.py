"""
GoHighLevel (GHL) CRM integration — OAuth 2.0, contact lookup, and post-call
note write-back.

External HTTP calls: GHL OAuth (``https://services.leadconnectorhq.com/oauth/token``)
and the API v2 Contacts/Notes API (``https://services.leadconnectorhq.com/contacts/*``).
Every API v2 request must carry the ``Version: 2021-07-28`` header.

Rate limit: GHL allows 100 requests/10s per authorization. ``check_rate_limit``
enforces this with a Redis fixed-window counter, keyed per tenant, before every
outbound API v2 call (not the OAuth token endpoint).

Every public entrypoint used at call time (contact lookup, CRM context injection,
post-call write-back) fails open: GHL being down or rate-limited logs a warning
and returns None/"" rather than raising, so a call is never blocked.

Unlike HubSpot/Salesforce (which fire-and-forget via a thread/asyncio task),
post-call write-back here is enqueued as an ARQ background job (see
app/workers/batch_call_worker.py::ghl_post_call_writeback) per the GHL
acceptance criteria, and retries at most once before recording the failure on
workspace_integration.metadata.last_ghl_error.
"""
from __future__ import annotations

import asyncio
import json
import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.db_encryption import decrypt_ghl_token, encrypt_ghl_token
from app.core.logger import logger
from app.core.secret_manager import get_ghl_oauth_credentials
from app.db.session import SessionLocal
from app.models.call_session import CallSession
from app.models.workspace_integration import WorkspaceIntegration
from app.utils.arq_pool import get_arq_pool
from app.utils.redis_client import get_redis

PROVIDER = "gohighlevel"

AUTHORIZE_URL = "https://marketplace.gohighlevel.com/oauth/chooselocation"

SCOPES = "contacts.readonly contacts.write"

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0

_STATE_PURPOSE = "ghl_oauth_state"
_STATE_TTL_MINUTES = 10

_CONTACT_CACHE_TTL_SECONDS = 300  # 5 minutes, per acceptance criteria
_CONTACT_NOT_FOUND_SENTINEL = "__not_found__"

_DEFAULT_WRITE_BACK_ENABLED = True

_WRITE_BACK_RETRY_DELAY_SECONDS = 300  # 5 minutes
_WRITE_BACK_ERROR_WINDOW_SECONDS = 24 * 60 * 60  # rolling 24h window
_WRITE_BACK_ALERT_THRESHOLD = 5

# Redis-backed fixed-window rate limiter (100 requests / 10 seconds).
_RATE_LIMIT_MAX_REQUESTS = 100
_RATE_LIMIT_WINDOW_SECONDS = 10

# Calling codes with a well-known "local"/trunk-prefixed national format,
# used as the fallback lookup when the E.164 search returns no match.
# AU: +61412345678 -> 0412345678 (per acceptance criteria example).
_TRUNK_ZERO_COUNTRY_CODES = ("61", "44", "64")


def _api_base_url() -> str:
    return settings.GHL_API_BASE_URL.rstrip("/")


def _api_version() -> str:
    return settings.GHL_API_VERSION


def _token_url() -> str:
    return f"{_api_base_url()}/oauth/token"


# ─── Redis-backed rate limiter (100 req / 10s) ─────────────────────────────────


async def check_rate_limit(tenant_id: uuid.UUID) -> None:
    """
    Enforce GHL's 100 requests/10s limit with a Redis fixed-window counter.

    No-ops (does not block) if Redis is unavailable — a missing rate limiter
    must never itself become a reason to fail a CRM call.
    """
    redis_client = get_redis()
    if redis_client is None:
        return

    key = f"ghl:rate_limit:{tenant_id}"
    try:
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, _RATE_LIMIT_WINDOW_SECONDS)
        if count > _RATE_LIMIT_MAX_REQUESTS:
            ttl = await redis_client.ttl(key)
            wait_seconds = ttl if ttl and ttl > 0 else _RATE_LIMIT_WINDOW_SECONDS
            logger.warning(
                "GHL rate limit exceeded for tenant=%s; waiting %.1fs", tenant_id, wait_seconds
            )
            await asyncio.sleep(wait_seconds)
    except Exception:
        logger.warning("GHL rate limiter check failed (continuing)", exc_info=True)


# ─── HTTP with backoff ────────────────────────────────────────────────────────


async def _request_with_backoff(method: str, url: str, **kwargs) -> httpx.Response:
    """Call GHL with exponential backoff on 429 (1s, 2s, 4s, 8s, 16s)."""
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
                "GHL 429 on %s %s; retrying in %.1fs (attempt %d/%d)",
                method,
                url,
                wait_seconds,
                attempt + 1,
                _MAX_RETRIES,
            )
            await asyncio.sleep(wait_seconds)
            attempt += 1


def _api_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Version": _api_version(),
    }


# ─── OAuth state (signed, carries tenant_id through the redirect round-trip) ──


def build_oauth_state(tenant_id: uuid.UUID) -> str:
    payload = {
        "tenant_id": str(tenant_id),
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

    tenant_id_str = payload.get("tenant_id")
    if not tenant_id_str:
        raise ValueError("OAuth state missing tenant_id")

    try:
        return uuid.UUID(tenant_id_str)
    except ValueError as exc:
        raise ValueError("OAuth state has malformed tenant_id") from exc


def get_redirect_uri() -> str:
    return settings.GHL_REDIRECT_URI or (
        f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/api/v1/integrations/leadconnector/callback"
    )


def build_authorization_url(state: str) -> str:
    client_id, _ = get_ghl_oauth_credentials()
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": get_redirect_uri(),
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ─── Token exchange / refresh ─────────────────────────────────────────────────


async def exchange_code_for_tokens(code: str) -> dict:
    client_id, client_secret = get_ghl_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        _token_url(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": get_redirect_uri(),
        },
    )
    response.raise_for_status()
    return response.json()


async def refresh_access_token(refresh_token: str) -> dict:
    client_id, client_secret = get_ghl_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        _token_url(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )
    response.raise_for_status()
    return response.json()


# ─── WorkspaceIntegration CRUD ────────────────────────────────────────────────


def get_integration(db: Session, tenant_id: uuid.UUID) -> Optional[WorkspaceIntegration]:
    return (
        db.query(WorkspaceIntegration)
        .filter(
            WorkspaceIntegration.workspace_id == tenant_id,
            WorkspaceIntegration.provider == PROVIDER,
        )
        .first()
    )


def tenant_has_ghl_connected(db: Session, tenant_id: uuid.UUID) -> bool:
    return get_integration(db, tenant_id) is not None


def get_connection_status(
    db: Session, tenant_id: uuid.UUID
) -> Tuple[bool, Optional[datetime]]:
    row = get_integration(db, tenant_id)
    if row is None:
        return False, None
    return True, row.created_at


def upsert_tokens(
    db: Session, tenant_id: uuid.UUID, token_response: dict
) -> WorkspaceIntegration:
    access_token = token_response["access_token"]
    refresh_token = token_response.get("refresh_token")
    expires_in = int(token_response.get("expires_in", 3600))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    location_id = token_response.get("locationId")

    row = get_integration(db, tenant_id)
    if row is None:
        row = WorkspaceIntegration(workspace_id=tenant_id, provider=PROVIDER)

    row.access_token = encrypt_ghl_token(access_token, db)
    if refresh_token:
        row.refresh_token = encrypt_ghl_token(refresh_token, db)
    row.token_expires_at = expires_at

    if location_id:
        updated_metadata = dict(row.extra_metadata or {})
        updated_metadata["location_id"] = location_id
        row.extra_metadata = updated_metadata
        flag_modified(row, "extra_metadata")

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def get_valid_access_token(
    db: Session, tenant_id: uuid.UUID
) -> Optional[Tuple[str, str]]:
    """Return (access_token, location_id), refreshing the token first if within 5 min of expiry."""
    row = get_integration(db, tenant_id)
    if row is None or not row.access_token:
        return None

    location_id = (row.extra_metadata or {}).get("location_id")
    if not location_id:
        logger.warning(
            "GHL integration for tenant=%s has no stored location_id", tenant_id
        )
        return None

    expires_at = row.token_expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if expires_at is not None and expires_at > now + timedelta(minutes=5):
        return decrypt_ghl_token(row.access_token, db), location_id

    if not row.refresh_token:
        logger.warning(
            "GHL access token expired for tenant=%s but no refresh_token stored", tenant_id
        )
        return None

    try:
        refresh_token_plain = decrypt_ghl_token(row.refresh_token, db)
        token_response = await refresh_access_token(refresh_token_plain)
    except Exception:
        logger.warning("GHL token refresh failed for tenant=%s", tenant_id, exc_info=True)
        return None

    row = upsert_tokens(db, tenant_id, token_response)
    location_id = (row.extra_metadata or {}).get("location_id") or location_id
    return decrypt_ghl_token(row.access_token, db), location_id


async def _force_refresh_access_token(
    db: Session, tenant_id: uuid.UUID
) -> Optional[Tuple[str, str]]:
    """
    Unconditionally refresh the GHL access token, ignoring token_expires_at.

    A call can run longer than our assumed token TTL; always called immediately
    before the write-back API call, mirroring HubSpot/Salesforce's forced
    pre-writeback refresh.
    """
    row = get_integration(db, tenant_id)
    if row is None or not row.refresh_token:
        return await get_valid_access_token(db, tenant_id)

    try:
        refresh_token_plain = decrypt_ghl_token(row.refresh_token, db)
        token_response = await refresh_access_token(refresh_token_plain)
    except Exception:
        logger.warning(
            "GHL forced pre-writeback token refresh failed for tenant=%s",
            tenant_id,
            exc_info=True,
        )
        return None

    row = upsert_tokens(db, tenant_id, token_response)
    location_id = (row.extra_metadata or {}).get("location_id")
    if not location_id:
        return None
    return decrypt_ghl_token(row.access_token, db), location_id


def get_integration_settings(db: Session, tenant_id: uuid.UUID) -> dict:
    """Return the tenant's GHL connection status and write-back toggle, with defaults applied."""
    row = get_integration(db, tenant_id)
    if row is None:
        return {
            "connected": False,
            "connected_at": None,
            "last_sync_at": None,
            "write_back_enabled": _DEFAULT_WRITE_BACK_ENABLED,
        }

    metadata = row.extra_metadata or {}
    last_sync_at = metadata.get("last_write_back_at") or metadata.get("last_lookup_at")
    return {
        "connected": True,
        "connected_at": row.created_at,
        "last_sync_at": last_sync_at,
        "write_back_enabled": metadata.get("write_back_enabled", _DEFAULT_WRITE_BACK_ENABLED),
    }


def update_integration_settings(
    db: Session, tenant_id: uuid.UUID, *, write_back_enabled: bool
) -> WorkspaceIntegration:
    """Persist the write-back toggle. Raises ValueError if not connected."""
    row = get_integration(db, tenant_id)
    if row is None:
        raise ValueError("GoHighLevel is not connected for this workspace")

    updated_metadata = dict(row.extra_metadata or {})
    updated_metadata["write_back_enabled"] = write_back_enabled
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None


def set_last_ghl_error(db: Session, tenant_id: uuid.UUID, error: Optional[str]) -> None:
    """
    Record (or clear, when error is None) the last post-call write-back failure
    on ``workspace_integration.metadata.last_ghl_error``.

    Also stamps last_write_back_status/last_write_back_at for the sync-status
    endpoint. error=None means the write-back just succeeded.
    """
    row = get_integration(db, tenant_id)
    if row is None:
        return

    updated_metadata = dict(row.extra_metadata or {})
    if error:
        updated_metadata["last_ghl_error"] = error
        updated_metadata["last_write_back_status"] = "failed"
    else:
        updated_metadata.pop("last_ghl_error", None)
        updated_metadata["last_write_back_status"] = "success"
        updated_metadata["last_write_back_at"] = _utc_now_iso()
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()


def record_write_back_failure(db: Session, tenant_id: uuid.UUID, error_msg: str) -> None:
    """
    Persist the structured last-failure error (after the single retry is
    exhausted), bump the rolling 24h failure counter, and alert the workspace
    admin once failures cross _WRITE_BACK_ALERT_THRESHOLD within the window.
    """
    row = get_integration(db, tenant_id)
    if row is None:
        return

    now = datetime.now(timezone.utc)
    now_iso = _utc_now_iso()
    window_start = now - timedelta(seconds=_WRITE_BACK_ERROR_WINDOW_SECONDS)

    updated_metadata = dict(row.extra_metadata or {})
    updated_metadata["last_ghl_error"] = error_msg
    updated_metadata["last_write_back_status"] = "failed"
    updated_metadata["last_write_back_at"] = now_iso

    failure_timestamps = [
        ts
        for ts in updated_metadata.get("write_back_failure_timestamps", [])
        if _parse_iso(ts) is not None and _parse_iso(ts) > window_start
    ]
    failure_timestamps.append(now_iso)
    updated_metadata["write_back_failure_timestamps"] = failure_timestamps

    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()

    if len(failure_timestamps) == _WRITE_BACK_ALERT_THRESHOLD:
        _send_write_back_failure_alert(db, tenant_id, error_msg, now_iso, len(failure_timestamps))


def _send_write_back_failure_alert(
    db: Session, tenant_id: uuid.UUID, error_msg: str, timestamp_iso: str, failure_count: int
) -> None:
    """Best-effort admin email alert. Never raises — a notification failure must not affect write-back."""
    try:
        from app.models.tenant import Tenant
        from app.services.data_export_service import _get_workspace_admin_email
        from app.services.email_service import email_service

        admin_email = _get_workspace_admin_email(db, tenant_id)
        tenant = db.get(Tenant, tenant_id)
        if not admin_email:
            admin_email = getattr(tenant, "contact_email", None) if tenant else None
        if not admin_email:
            logger.warning(
                "No admin or contact email found for tenant=%s; cannot send "
                "GoHighLevel write-back failure alert",
                tenant_id,
            )
            return

        workspace_name = tenant.name if tenant else str(tenant_id)
        frontend_base = (settings.FRONTEND_URL or "").rstrip("/")
        reconnect_url = f"{frontend_base}/settings/integrations" if frontend_base else "/settings/integrations"

        email_service.send_generic_email(
            to_email=admin_email,
            subject=f"Action Required: GoHighLevel Integration Write-Back Failures for {workspace_name}",
            html_body=(
                f"<p>GoHighLevel post-call write-back has failed {failure_count} times in the "
                f"last 24 hours for workspace <strong>{workspace_name}</strong>.</p>"
                f"<p>Most recent failure at {timestamp_iso}: {error_msg}</p>"
                f"<p>Please reconnect GoHighLevel to restore syncing: "
                f'<a href="{reconnect_url}">{reconnect_url}</a></p>'
            ),
        )
    except Exception:
        logger.warning(
            "Failed to send GoHighLevel write-back failure alert for tenant=%s",
            tenant_id,
            exc_info=True,
        )


def get_sync_status(db: Session, tenant_id: uuid.UUID) -> dict:
    """Sync visibility for the dashboard: last lookup/write-back times, status, 24h error count."""
    row = get_integration(db, tenant_id)
    if row is None:
        return {
            "last_lookup_at": None,
            "last_write_back_at": None,
            "last_write_back_status": None,
            "last_ghl_error": None,
            "error_count_24h": 0,
        }

    metadata = row.extra_metadata or {}
    window_start = datetime.now(timezone.utc) - timedelta(seconds=_WRITE_BACK_ERROR_WINDOW_SECONDS)
    error_count_24h = sum(
        1
        for ts in metadata.get("write_back_failure_timestamps", [])
        if _parse_iso(ts) is not None and _parse_iso(ts) > window_start
    )

    return {
        "last_lookup_at": metadata.get("last_lookup_at"),
        "last_write_back_at": metadata.get("last_write_back_at"),
        "last_write_back_status": metadata.get("last_write_back_status"),
        "last_ghl_error": metadata.get("last_ghl_error"),
        "error_count_24h": error_count_24h,
    }


def _touch_last_lookup_at(db: Session, tenant_id: uuid.UUID, *, commit: bool = True) -> None:
    """
    Best-effort timestamp of the last contact lookup, for the sync-status endpoint.

    commit=False must be used when `db` is the shared, long-lived session tied to
    an in-progress call (see get_crm_context_block_for_call), which never calls
    db.commit() directly to avoid prematurely committing the caller's still-open
    call transaction.
    """
    try:
        row = get_integration(db, tenant_id)
        if row is None:
            return
        updated_metadata = dict(row.extra_metadata or {})
        updated_metadata["last_lookup_at"] = _utc_now_iso()
        row.extra_metadata = updated_metadata
        flag_modified(row, "extra_metadata")
        db.add(row)
        if commit:
            db.commit()
        else:
            with db.begin_nested():
                db.flush()
    except Exception:
        logger.warning(
            "Failed to record GHL last_lookup_at (non-critical) tenant=%s",
            tenant_id,
            exc_info=True,
        )


async def disconnect(db: Session, tenant_id: uuid.UUID) -> bool:
    """
    Revoke local GHL OAuth credentials and delete the workspace_integration row.

    GHL's API v2 does not document a token-revocation endpoint, so — unlike
    HubSpot/Salesforce disconnect — there is no external revoke call here;
    deleting the encrypted tokens locally is the full revocation surface we
    control.
    """
    row = get_integration(db, tenant_id)
    if row is None:
        return False

    db.delete(row)
    db.commit()
    return True


# ─── Contact lookup (Contacts API, Redis-cached) ───────────────────────────────


def normalize_to_e164(phone: str) -> str:
    """Normalize phone number to E.164 format if possible.
    E.164 format: +[country_code][subscriber_number] up to 15 digits total.
    """
    if not phone:
        return ""
    phone = phone.strip()
    has_plus = phone.startswith("+")
    digits = "".join(c for c in phone if c.isdigit())

    if not digits:
        return phone

    if has_plus:
        return f"+{digits}"

    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    else:
        return f"+{digits}"


def local_format_fallback(e164_phone: str) -> Optional[str]:
    """
    Best-effort "local" format fallback used when the E.164 lookup finds no
    contact: strip the country calling code and prepend a trunk 0, mirroring
    AU (+61412345678 -> 0412345678) and similar 0-trunk numbering plans.

    Returns None when the number doesn't start with a known trunk-0 calling
    code (e.g. NANP +1 numbers have no equivalent local form to try).
    """
    if not e164_phone or not e164_phone.startswith("+"):
        return None
    digits = e164_phone[1:]
    for code in sorted(_TRUNK_ZERO_COUNTRY_CODES, key=len, reverse=True):
        if digits.startswith(code):
            subscriber = digits[len(code):]
            if subscriber:
                return f"0{subscriber}"
    return None


def _contact_cache_key(tenant_id: uuid.UUID, phone: str) -> str:
    # Hash the phone number to avoid storing PII in plaintext in Redis keys.
    phone_hash = hashlib.sha256(phone.encode("utf-8")).hexdigest()
    return f"ghl:contact:{tenant_id}:{phone_hash}"


def _contact_dict_from_ghl(raw: dict) -> dict:
    first = (raw.get("firstName") or "").strip()
    last = (raw.get("lastName") or "").strip()
    name = raw.get("contactName") or f"{first} {last}".strip() or None
    return {
        "id": raw.get("id"),
        "name": name,
        "email": raw.get("email"),
        "tags": raw.get("tags") or [],
        "pipeline_stage": raw.get("pipelineStage") or raw.get("pipeline_stage"),
        "last_activity_date": raw.get("lastActivity") or raw.get("dateUpdated"),
    }


async def search_contact_by_phone(
    access_token: str, location_id: str, phone: str, tenant_id: uuid.UUID
) -> Optional[dict]:
    """
    Search GHL contacts by phone. Tries E.164 first; if no match, falls back
    to the local/trunk-0 format (see local_format_fallback) before giving up.
    """
    url = f"{_api_base_url()}/contacts/"
    headers = _api_headers(access_token)

    e164_phone = normalize_to_e164(phone)
    candidates = [e164_phone] if e164_phone else []
    local_phone = local_format_fallback(e164_phone) if e164_phone else None
    if local_phone:
        candidates.append(local_phone)

    for candidate in candidates:
        await check_rate_limit(tenant_id)
        response = await _request_with_backoff(
            "GET",
            url,
            headers=headers,
            params={"locationId": location_id, "phone": candidate},
        )
        response.raise_for_status()
        contacts = response.json().get("contacts") or []
        if contacts:
            return contacts[0]

    return None


async def get_contact_for_phone(
    db: Session, tenant_id: uuid.UUID, phone: str, *, commit_lookup_timestamp: bool = True
) -> Optional[dict]:
    """Look up a contact by phone, Redis-cached for 5 minutes. Fails open on any error.

    commit_lookup_timestamp=False must be passed by live in-call callers sharing a
    long-lived db session (see _touch_last_lookup_at).
    """
    _touch_last_lookup_at(db, tenant_id, commit=commit_lookup_timestamp)

    normalized_phone = normalize_to_e164(phone)
    redis_client = get_redis()
    cache_key = _contact_cache_key(tenant_id, normalized_phone)

    if redis_client is not None:
        try:
            cached = await redis_client.get(cache_key)
            if cached is not None:
                return None if cached == _CONTACT_NOT_FOUND_SENTINEL else json.loads(cached)
        except Exception:
            logger.warning("GHL contact cache read failed (continuing)", exc_info=True)

    try:
        token_info = await get_valid_access_token(db, tenant_id)
        if not token_info:
            return None
        access_token, location_id = token_info
        raw_contact = await search_contact_by_phone(
            access_token, location_id, normalized_phone, tenant_id
        )
    except Exception:
        logger.warning(
            "GHL contact search failed for tenant=%s (failing open)", tenant_id, exc_info=True
        )
        return None

    contact = _contact_dict_from_ghl(raw_contact) if raw_contact else None

    if redis_client is not None:
        try:
            payload = _CONTACT_NOT_FOUND_SENTINEL if contact is None else json.dumps(contact)
            await redis_client.set(cache_key, payload, ex=_CONTACT_CACHE_TTL_SECONDS)
        except Exception:
            logger.warning("GHL contact cache write failed (non-critical)", exc_info=True)

    return contact


# ─── CRM context injection (conversation_orchestrator) ────────────────────────


async def get_crm_context_block_for_call(db: Session, call_session: CallSession) -> str:
    """
    Fetch the CRM context block for a call's system prompt, once per call.

    Cached in call_session.call_metadata["ghl_crm_context"] so subsequent turns
    (the prompt is rebuilt every turn) reuse the cached string instead of
    re-querying GHL/Redis. Fails open — returns "" on any error so a CRM
    outage never blocks the call.
    """
    metadata = call_session.call_metadata or {}
    if "ghl_crm_context" in metadata:
        return metadata.get("ghl_crm_context") or ""

    context_block = ""
    try:
        if call_session.customer_phone_number and tenant_has_ghl_connected(
            db, call_session.tenant_id
        ):
            contact = await get_contact_for_phone(
                db,
                call_session.tenant_id,
                call_session.customer_phone_number,
                commit_lookup_timestamp=False,
            )
            if contact:
                tags = contact.get("tags") or []
                context_block = (
                    f"CRM CONTEXT (GoHighLevel): Name: {contact.get('name') or 'Unknown'}, "
                    f"Tags: {', '.join(tags)}, "
                    f"Pipeline: {contact.get('pipeline_stage') or 'Unknown'}"
                )
    except Exception:
        logger.warning(
            "GHL CRM context lookup failed (continuing without CRM context)", exc_info=True
        )
        context_block = ""

    try:
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["ghl_crm_context"] = context_block
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache GHL CRM context on call_session (non-critical)", exc_info=True
        )

    return context_block


# ─── Note creation (post-call write-back + direct endpoint) ───────────────────


def _transcript_text(db: Session, call_session: CallSession) -> str:
    from app.services.transcript_service import transcript_service

    messages = transcript_service.get_messages_by_session(db, call_session.id)
    lines = [f"{m.role}: {m.message}" for m in messages if m.message]
    return "\n".join(lines)


def generate_transcript_summary(db: Session, call_session: CallSession) -> str:
    """2-sentence Gemini summary of the call transcript. Fails open — returns ''."""
    transcript_text = _transcript_text(db, call_session)
    if not transcript_text.strip():
        return ""

    from app.services.gemini_service import gemini_service

    try:
        result = gemini_service.generate_text(
            prompt=(
                "Summarize this phone call transcript in exactly 2 sentences, "
                "focused on the caller's request and the outcome:\n\n" + transcript_text
            ),
            model_name="gemini-2.0-flash",
            temperature=0.2,
            max_tokens=120,
        )
        return (result.get("content") or "").strip()
    except Exception:
        logger.warning(
            "GHL post-call summary generation failed (continuing without summary)", exc_info=True
        )
        return ""


def get_cached_transcript_summary(db: Session, call_session: CallSession) -> str:
    """
    2-sentence Gemini summary of the call transcript, cached on
    call_session.call_metadata["ghl_call_summary"] so a retried write-back
    never calls Gemini twice for the same call.
    """
    metadata = call_session.call_metadata or {}
    if "ghl_call_summary" in metadata:
        return metadata.get("ghl_call_summary") or ""

    summary = generate_transcript_summary(db, call_session)

    try:
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["ghl_call_summary"] = summary
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache GHL call summary on call_session (non-critical)", exc_info=True
        )

    return summary


def build_note_content(
    *, duration_seconds: int, outcome: Optional[str], summary: str
) -> str:
    """Note body: call duration, outcome, and the 2-sentence Gemini summary."""
    summary_text = summary or "No summary available."
    return (
        f"Call duration: {max(int(duration_seconds), 0)}s. "
        f"Outcome: {outcome or 'completed'}. "
        f"Summary: {summary_text}"
    )


async def create_note(
    access_token: str, contact_id: str, content: str, tenant_id: uuid.UUID
) -> dict:
    await check_rate_limit(tenant_id)
    url = f"{_api_base_url()}/contacts/{contact_id}/notes"
    response = await _request_with_backoff(
        "POST",
        url,
        headers={**_api_headers(access_token), "Content-Type": "application/json"},
        json={"body": content},
    )
    response.raise_for_status()
    return response.json()


_MAX_ERROR_LEN = 500


def _safe_error_msg(exc: Exception) -> str:
    """Extract a brief, safe error summary for tenant-visible storage."""
    msg = str(exc)
    # Redact anything that looks like a bearer token
    msg = re.sub(r'Bearer [A-Za-z0-9\-._~+/]+=*', 'Bearer [redacted]', msg)
    return msg[:_MAX_ERROR_LEN]


async def _run_post_call_writeback_async(db: Session, call_session: CallSession) -> None:
    tenant_id = call_session.tenant_id

    integration_settings = get_integration_settings(db, tenant_id)
    if not integration_settings["write_back_enabled"]:
        logger.info(
            "GHL write-back skipped — disabled in settings for tenant=%s", tenant_id
        )
        return

    # A call can outlast our assumed access-token TTL — always force a fresh
    # token right before the write-back call rather than trusting the DB's
    # cached token_expires_at.
    token_info = await _force_refresh_access_token(db, tenant_id)
    if not token_info:
        return
    access_token, _location_id = token_info

    contact = await get_contact_for_phone(db, tenant_id, call_session.customer_phone_number)
    if not contact or not contact.get("id"):
        logger.info(
            "GHL write-back skipped — no matching contact for session=%s", call_session.id
        )
        return

    summary = get_cached_transcript_summary(db, call_session)
    note_content = build_note_content(
        duration_seconds=call_session.duration or 0,
        outcome=call_session.status,
        summary=summary,
    )

    async def _write() -> None:
        await create_note(access_token, contact["id"], note_content, tenant_id)

    try:
        await _write()
    except Exception as exc:
        logger.warning(
            "GHL note creation failed for session=%s; retrying in %ds: %s",
            call_session.id,
            _WRITE_BACK_RETRY_DELAY_SECONDS,
            exc,
            exc_info=True,
        )
        try:
            db.rollback()
        except Exception:
            logger.warning(
                "Failed to release DB connection before GHL write-back retry (non-critical)",
                exc_info=True,
            )
        await asyncio.sleep(_WRITE_BACK_RETRY_DELAY_SECONDS)
        try:
            await _write()
        except Exception as retry_exc:
            logger.error(
                "GHL note creation failed after retry for session=%s: %s",
                call_session.id,
                retry_exc,
                exc_info=True,
            )
            record_write_back_failure(db, tenant_id, _safe_error_msg(retry_exc))
            return

    set_last_ghl_error(db, tenant_id, None)
    logger.info(
        "GHL note created for session=%s contact=%s", call_session.id, contact["id"]
    )


async def _post_call_writeback_arq_task(ctx: dict, call_session_id: str) -> None:
    """
    ARQ job entrypoint — registered as ``ghl_post_call_writeback`` in
    app/workers/batch_call_worker.py::WorkerSettings.functions.
    """
    db: Session = SessionLocal()
    try:
        session_uuid = uuid.UUID(call_session_id)
        call_session = db.query(CallSession).filter(CallSession.id == session_uuid).first()
        if not call_session or not call_session.customer_phone_number:
            return
        if not tenant_has_ghl_connected(db, call_session.tenant_id):
            return

        await _run_post_call_writeback_async(db, call_session)
    except Exception:
        logger.warning(
            "GHL post-call write-back failed (non-critical) session=%s",
            call_session_id,
            exc_info=True,
        )
    finally:
        db.close()


def schedule_ghl_writeback(call_session_id: uuid.UUID) -> None:
    """
    Enqueue the post-call write-back as an ARQ background job. Fire-and-forget
    — never blocks the caller. Fails open if the ARQ pool isn't ready: GHL
    sync is best-effort and must never affect call completion.
    """
    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; GHL write-back skipped for session=%s", call_session_id
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job("ghl_post_call_writeback", str(call_session_id))
        except Exception as exc:
            logger.warning("Failed to enqueue GHL write-back job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())
