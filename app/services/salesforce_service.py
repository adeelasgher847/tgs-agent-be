"""
Salesforce CRM integration — OAuth 2.0 Web Server Flow, contact lookup, and
post-call write-back to the Task (Activity) object.

External HTTP calls: Salesforce OAuth (``{SALESFORCE_LOGIN_URL}/services/oauth2/*``,
always the fixed login host, never the org's instance) and the org's REST Data API
(``{instance_url}/services/data/v{version}/*`` — instance_url varies per org and is
only known after the OAuth token exchange, unlike HubSpot's fixed api.hubapi.com host).

Every public entrypoint used at call time (contact lookup, CRM context injection,
post-call write-back) fails open: Salesforce being down or rate-limited logs a
warning and returns None/"" rather than raising, so a call is never blocked.
"""
from __future__ import annotations

import asyncio
import json
import hashlib
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.db_encryption import decrypt_salesforce_token, encrypt_salesforce_token
from app.core.logger import logger
from app.core.secret_manager import get_salesforce_oauth_credentials
from app.db.session import SessionLocal
from app.models.call_session import CallSession
from app.models.workspace_integration import WorkspaceIntegration
from app.utils.redis_client import get_redis

PROVIDER = "salesforce"

SCOPES = "api refresh_token"

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0

_STATE_PURPOSE = "salesforce_oauth_state"
_STATE_TTL_MINUTES = 10

_CONTACT_CACHE_TTL_SECONDS = 300  # 5 minutes, per acceptance criteria
_CONTACT_NOT_FOUND_SENTINEL = "__not_found__"

_DEFAULT_WRITE_BACK_ENABLED = True

_WRITE_BACK_RETRY_DELAY_SECONDS = 300  # 5 minutes
_WRITE_BACK_ERROR_WINDOW_SECONDS = 24 * 60 * 60  # rolling 24h window for admin alerting
_WRITE_BACK_ALERT_THRESHOLD = 5


def _login_url() -> str:
    return settings.SALESFORCE_LOGIN_URL.rstrip("/")


def _api_version() -> str:
    return settings.SALESFORCE_API_VERSION


# ─── HTTP with backoff ────────────────────────────────────────────────────────


async def _request_with_backoff(method: str, url: str, **kwargs) -> httpx.Response:
    """Call Salesforce with exponential backoff on 429 (1s, 2s, 4s, 8s, 16s)."""
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
                "Salesforce 429 on %s %s; retrying in %.1fs (attempt %d/%d)",
                method,
                url,
                wait_seconds,
                attempt + 1,
                _MAX_RETRIES,
            )
            await asyncio.sleep(wait_seconds)
            attempt += 1


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
    return settings.SALESFORCE_REDIRECT_URI or (
        f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/api/v1/integrations/salesforce/callback"
    )


def build_authorization_url(state: str) -> str:
    client_id, _ = get_salesforce_oauth_credentials()
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": get_redirect_uri(),
        "scope": SCOPES,
        "state": state,
    }
    return f"{_login_url()}/services/oauth2/authorize?{urlencode(params)}"


# ─── Token exchange / refresh ─────────────────────────────────────────────────


async def exchange_code_for_tokens(code: str) -> dict:
    client_id, client_secret = get_salesforce_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        f"{_login_url()}/services/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": get_redirect_uri(),
            "code": code,
        },
    )
    response.raise_for_status()
    return response.json()


async def refresh_access_token(refresh_token: str) -> dict:
    client_id, client_secret = get_salesforce_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        f"{_login_url()}/services/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
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


def tenant_has_salesforce_connected(db: Session, tenant_id: uuid.UUID) -> bool:
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
    instance_url = token_response.get("instance_url")
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=settings.SALESFORCE_ACCESS_TOKEN_TTL_SECONDS
    )

    row = get_integration(db, tenant_id)
    if row is None:
        row = WorkspaceIntegration(workspace_id=tenant_id, provider=PROVIDER)

    row.access_token = encrypt_salesforce_token(access_token, db)
    if refresh_token:
        row.refresh_token = encrypt_salesforce_token(refresh_token, db)
    row.token_expires_at = expires_at

    if instance_url:
        updated_metadata = dict(row.extra_metadata or {})
        updated_metadata["instance_url"] = instance_url
        row.extra_metadata = updated_metadata
        flag_modified(row, "extra_metadata")

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def get_valid_access_token(
    db: Session, tenant_id: uuid.UUID
) -> Optional[Tuple[str, str]]:
    """Return (access_token, instance_url), refreshing the token first if it's near-expiry."""
    row = get_integration(db, tenant_id)
    if row is None or not row.access_token:
        return None

    instance_url = (row.extra_metadata or {}).get("instance_url")
    if not instance_url:
        logger.warning(
            "Salesforce integration for tenant=%s has no stored instance_url", tenant_id
        )
        return None

    expires_at = row.token_expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if expires_at is not None and expires_at > now + timedelta(seconds=60):
        return decrypt_salesforce_token(row.access_token, db), instance_url

    if not row.refresh_token:
        logger.warning(
            "Salesforce access token expired for tenant=%s but no refresh_token stored",
            tenant_id,
        )
        return None

    try:
        refresh_token_plain = decrypt_salesforce_token(row.refresh_token, db)
        token_response = await refresh_access_token(refresh_token_plain)
    except Exception:
        logger.warning(
            "Salesforce token refresh failed for tenant=%s", tenant_id, exc_info=True
        )
        return None

    row = upsert_tokens(db, tenant_id, token_response)
    instance_url = (row.extra_metadata or {}).get("instance_url") or instance_url
    return decrypt_salesforce_token(row.access_token, db), instance_url


async def _force_refresh_access_token(
    db: Session, tenant_id: uuid.UUID
) -> Optional[Tuple[str, str]]:
    """
    Unconditionally refresh the Salesforce access token, ignoring token_expires_at.

    A call can run longer than our assumed token TTL; always called immediately
    before the write-back API call, mirroring HubSpot's forced pre-writeback refresh.
    """
    row = get_integration(db, tenant_id)
    if row is None or not row.refresh_token:
        return await get_valid_access_token(db, tenant_id)

    try:
        refresh_token_plain = decrypt_salesforce_token(row.refresh_token, db)
        token_response = await refresh_access_token(refresh_token_plain)
    except Exception:
        logger.warning(
            "Salesforce forced pre-writeback token refresh failed for tenant=%s",
            tenant_id,
            exc_info=True,
        )
        return None

    row = upsert_tokens(db, tenant_id, token_response)
    instance_url = (row.extra_metadata or {}).get("instance_url")
    if not instance_url:
        return None
    return decrypt_salesforce_token(row.access_token, db), instance_url


def get_integration_settings(db: Session, tenant_id: uuid.UUID) -> dict:
    """Return the tenant's Salesforce connection status and write-back toggle, with defaults applied."""
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
        raise ValueError("Salesforce is not connected for this workspace")

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


def set_last_write_back_error(
    db: Session, tenant_id: uuid.UUID, error: Optional[str]
) -> None:
    """
    Record (or clear, when error is None) the last post-call write-back failure.

    Also stamps last_write_back_status/last_write_back_at for the sync-status
    endpoint. error=None means the write-back just succeeded.
    """
    row = get_integration(db, tenant_id)
    if row is None:
        return

    updated_metadata = dict(row.extra_metadata or {})
    if error:
        updated_metadata["last_write_back_error"] = error
        updated_metadata["last_write_back_status"] = "failed"
    else:
        updated_metadata.pop("last_write_back_error", None)
        updated_metadata["last_write_back_status"] = "success"
        updated_metadata["last_write_back_at"] = _utc_now_iso()
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()


def record_write_back_failure(db: Session, tenant_id: uuid.UUID, error_msg: str) -> None:
    """
    Persist the structured last-failure error (after retry exhaustion), bump the
    rolling 24h failure counter, and alert the workspace admin once failures
    cross _WRITE_BACK_ALERT_THRESHOLD within the window.
    """
    row = get_integration(db, tenant_id)
    if row is None:
        return

    now = datetime.now(timezone.utc)
    now_iso = _utc_now_iso()
    window_start = now - timedelta(seconds=_WRITE_BACK_ERROR_WINDOW_SECONDS)

    updated_metadata = dict(row.extra_metadata or {})
    updated_metadata["last_write_back_error"] = {"timestamp": now_iso, "error": error_msg}
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

    # Alert only on the exact crossing into the threshold, not on every failure
    # thereafter — each call here appends exactly one timestamp, so the count
    # can't skip over the threshold, and this avoids re-alerting the admin on
    # every single failure once a workspace is already over the limit.
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
                "Salesforce write-back failure alert",
                tenant_id,
            )
            return

        workspace_name = tenant.name if tenant else str(tenant_id)
        frontend_base = (settings.FRONTEND_URL or "").rstrip("/")
        reconnect_url = f"{frontend_base}/settings/integrations" if frontend_base else "/settings/integrations"

        email_service.send_generic_email(
            to_email=admin_email,
            subject=f"Action Required: Salesforce Integration Write-Back Failures for {workspace_name}",
            html_body=(
                f"<p>Salesforce post-call write-back has failed {failure_count} times in the "
                f"last 24 hours for workspace <strong>{workspace_name}</strong>.</p>"
                f"<p>Most recent failure at {timestamp_iso}: {error_msg}</p>"
                f"<p>Please reconnect Salesforce to restore syncing: "
                f'<a href="{reconnect_url}">{reconnect_url}</a></p>'
            ),
        )
    except Exception:
        logger.warning(
            "Failed to send Salesforce write-back failure alert for tenant=%s",
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
        "error_count_24h": error_count_24h,
    }


def _touch_last_lookup_at(db: Session, tenant_id: uuid.UUID, *, commit: bool = True) -> None:
    """
    Best-effort timestamp of the last contact lookup, for the sync-status endpoint.

    commit=False must be used when `db` is the shared, long-lived session tied to
    an in-progress call (see get_crm_context_block_for_call), which never calls
    db.commit() directly to avoid prematurely committing the caller's still-open
    call transaction — this mirrors that begin_nested()/flush() pattern instead.
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
            "Failed to record Salesforce last_lookup_at (non-critical) tenant=%s",
            tenant_id,
            exc_info=True,
        )


async def disconnect(db: Session, tenant_id: uuid.UUID) -> bool:
    """Revoke the refresh token at Salesforce (best-effort) and delete the local row."""
    row = get_integration(db, tenant_id)
    if row is None:
        return False

    if row.refresh_token:
        try:
            refresh_token_plain = decrypt_salesforce_token(row.refresh_token, db)
            await _request_with_backoff(
                "POST",
                f"{_login_url()}/services/oauth2/revoke",
                data={"token": refresh_token_plain},
            )
        except Exception:
            logger.warning(
                "Salesforce token revoke failed (continuing with local disconnect) tenant=%s",
                tenant_id,
                exc_info=True,
            )

    db.delete(row)
    db.commit()
    return True


# ─── Contact lookup (SOQL query, Redis-cached) ─────────────────────────────────


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


def _get_phone_search_values(phone: str) -> list[str]:
    """Generate a list of unique exact-match phone number variations for lookup."""
    values = set()
    digits = "".join(c for c in phone if c.isdigit())

    normalized = normalize_to_e164(phone)
    if normalized:
        values.add(normalized)

    stripped = phone.strip()
    if stripped:
        values.add(stripped)

    if digits:
        values.add(digits)
        if len(digits) == 11 and digits.startswith("1"):
            values.add(digits[1:])
        elif len(digits) == 10:
            values.add(f"1{digits}")

    return sorted(list(values))


def _contact_cache_key(tenant_id: uuid.UUID, phone: str) -> str:
    # Hash the phone number to avoid storing PII in plaintext in Redis keys.
    phone_hash = hashlib.sha256(phone.encode("utf-8")).hexdigest()
    return f"salesforce:contact:{tenant_id}:{phone_hash}"


def _soql_escape(value: str) -> str:
    """Escape single quotes and backslashes for safe interpolation into a SOQL string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _contact_dict_from_salesforce(raw: dict) -> dict:
    account = raw.get("Account") or {}
    return {
        "id": raw.get("Id"),
        "name": raw.get("Name"),
        "email": raw.get("Email"),
        "account": account.get("Name") if isinstance(account, dict) else None,
    }


async def search_contact_by_phone(
    access_token: str, instance_url: str, phone: str
) -> Optional[dict]:
    search_vals = _get_phone_search_values(phone)
    if not search_vals:
        return None

    phone_clause = " OR ".join(f"Phone='{_soql_escape(v)}'" for v in search_vals)
    soql = f"SELECT Id, Name, Email, Account.Name FROM Contact WHERE {phone_clause} LIMIT 1"

    url = f"{instance_url.rstrip('/')}/services/data/{_api_version()}/query"
    response = await _request_with_backoff(
        "GET",
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"q": soql},
    )
    response.raise_for_status()
    records = response.json().get("records") or []
    return records[0] if records else None


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
            logger.warning("Salesforce contact cache read failed (continuing)", exc_info=True)

    try:
        token_info = await get_valid_access_token(db, tenant_id)
        if not token_info:
            return None
        access_token, instance_url = token_info
        raw_contact = await search_contact_by_phone(access_token, instance_url, normalized_phone)
    except Exception:
        logger.warning(
            "Salesforce contact search failed for tenant=%s (failing open)",
            tenant_id,
            exc_info=True,
        )
        return None

    contact = _contact_dict_from_salesforce(raw_contact) if raw_contact else None

    if redis_client is not None:
        try:
            payload = (
                _CONTACT_NOT_FOUND_SENTINEL if contact is None else json.dumps(contact)
            )
            await redis_client.set(cache_key, payload, ex=_CONTACT_CACHE_TTL_SECONDS)
        except Exception:
            logger.warning("Salesforce contact cache write failed (non-critical)", exc_info=True)

    return contact


# ─── CRM context injection (conversation_orchestrator) ────────────────────────


async def get_crm_context_block_for_call(db: Session, call_session: CallSession) -> str:
    """
    Fetch the CRM context block for a call's system prompt, once per call.

    Cached in call_session.call_metadata["salesforce_crm_context"] so subsequent
    turns (the prompt is rebuilt every turn) reuse the cached string instead of
    re-querying Salesforce/Redis. Fails open — returns "" on any error so a CRM
    outage never blocks the call.
    """
    metadata = call_session.call_metadata or {}
    if "salesforce_crm_context" in metadata:
        return metadata.get("salesforce_crm_context") or ""

    context_block = ""
    try:
        if call_session.customer_phone_number and tenant_has_salesforce_connected(
            db, call_session.tenant_id
        ):
            contact = await get_contact_for_phone(
                db,
                call_session.tenant_id,
                call_session.customer_phone_number,
                commit_lookup_timestamp=False,
            )
            if contact:
                context_block = (
                    f"CRM CONTEXT (Salesforce): Name: {contact.get('name') or 'Unknown'}, "
                    f"Account: {contact.get('account') or 'Unknown'}, "
                    f"Email: {contact.get('email') or 'Unknown'}"
                )
    except Exception:
        logger.warning(
            "Salesforce CRM context lookup failed (continuing without CRM context)",
            exc_info=True,
        )
        context_block = ""

    try:
        # Use a nested transaction savepoint to isolate database flushing.
        # This keeps the helper fail-open and avoids committing the main transaction
        # during prompt generation.
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["salesforce_crm_context"] = context_block
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache Salesforce CRM context on call_session (non-critical)",
            exc_info=True,
        )

    return context_block


# ─── Post-call write-back (Task/Activity + Gemini summary) ────────────────────


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
            "Salesforce post-call summary generation failed (continuing without summary)",
            exc_info=True,
        )
        return ""


def get_cached_transcript_summary(db: Session, call_session: CallSession) -> str:
    """
    2-sentence Gemini summary of the call transcript, cached on
    call_session.call_metadata["salesforce_call_summary"] so a retried write-back
    never calls Gemini twice for the same call.
    """
    metadata = call_session.call_metadata or {}
    if "salesforce_call_summary" in metadata:
        return metadata.get("salesforce_call_summary") or ""

    summary = generate_transcript_summary(db, call_session)

    try:
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["salesforce_call_summary"] = summary
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache Salesforce call summary on call_session (non-critical)",
            exc_info=True,
        )

    return summary


def _sf_call_type(call_type: Optional[str]) -> str:
    return "Inbound" if (call_type or "").lower() == "inbound" else "Outbound"


async def create_call_task(
    access_token: str,
    instance_url: str,
    contact_id: str,
    *,
    occurred_at: datetime,
    duration_seconds: int,
    description: str,
    call_type: Optional[str] = None,
) -> dict:
    payload = {
        "WhoId": contact_id,
        "Subject": "AI Voice Call",
        "Status": "Completed",
        "Description": description,
        "ActivityDate": occurred_at.date().isoformat(),
        "TaskSubtype": "Call",
        "CallDurationInSeconds": max(int(duration_seconds), 0),
        "CallType": _sf_call_type(call_type),
    }
    url = f"{instance_url.rstrip('/')}/services/data/{_api_version()}/sobjects/Task"
    response = await _request_with_backoff(
        "POST",
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        json=payload,
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
            "Salesforce write-back skipped — disabled in settings for tenant=%s",
            tenant_id,
        )
        return

    # A call can outlast our assumed access-token TTL — always force a fresh
    # token right before the write-back call rather than trusting the DB's
    # cached token_expires_at.
    token_info = await _force_refresh_access_token(db, tenant_id)
    if not token_info:
        return
    access_token, instance_url = token_info

    contact = await get_contact_for_phone(
        db, tenant_id, call_session.customer_phone_number
    )
    if not contact or not contact.get("id"):
        logger.info(
            "Salesforce write-back skipped — no matching contact for session=%s",
            call_session.id,
        )
        return

    summary = get_cached_transcript_summary(db, call_session)
    body_text = summary or "Call completed — no transcript summary available."

    async def _write() -> None:
        await create_call_task(
            access_token,
            instance_url,
            contact["id"],
            occurred_at=call_session.start_time or datetime.now(timezone.utc),
            duration_seconds=call_session.duration or 0,
            description=body_text,
            call_type=call_session.call_type,
        )

    try:
        await _write()
    except Exception as exc:
        logger.warning(
            "Salesforce Task creation failed for session=%s; retrying in %ds: %s",
            call_session.id,
            _WRITE_BACK_RETRY_DELAY_SECONDS,
            exc,
            exc_info=True,
        )
        # Release the checked-out DB connection for the duration of the retry
        # delay — see hubspot_service._run_post_call_writeback_async for the
        # rationale (avoids exhausting the pool during a CRM outage).
        try:
            db.rollback()
        except Exception:
            logger.warning(
                "Failed to release DB connection before Salesforce write-back retry (non-critical)",
                exc_info=True,
            )
        await asyncio.sleep(_WRITE_BACK_RETRY_DELAY_SECONDS)
        try:
            await _write()
        except Exception as retry_exc:
            logger.error(
                "Salesforce Task creation failed after retry for session=%s: %s",
                call_session.id,
                retry_exc,
                exc_info=True,
            )
            record_write_back_failure(db, tenant_id, _safe_error_msg(retry_exc))
            return

    set_last_write_back_error(db, tenant_id, None)
    logger.info(
        "Salesforce Task created for session=%s contact=%s",
        call_session.id,
        contact["id"],
    )


def run_post_call_writeback(call_session_id: uuid.UUID) -> None:
    """Sync entrypoint — opens its own session, mirrors hubspot_service.run_post_call_writeback."""
    db: Session = SessionLocal()
    try:
        call_session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
        if not call_session or not call_session.customer_phone_number:
            return
        if not tenant_has_salesforce_connected(db, call_session.tenant_id):
            return

        asyncio.run(_run_post_call_writeback_async(db, call_session))
    except Exception:
        logger.warning(
            "Salesforce post-call write-back failed (non-critical) session=%s",
            call_session_id,
            exc_info=True,
        )
    finally:
        db.close()


async def _run_post_call_writeback_in_thread(call_session_id: uuid.UUID) -> None:
    await asyncio.to_thread(run_post_call_writeback, call_session_id)


def schedule_salesforce_writeback(call_session_id: uuid.UUID) -> None:
    """
    Fire-and-forget post-call write-back. Never blocks the caller.

    Write-back can retry once after a 5-minute delay on failure (see
    _run_post_call_writeback_async), so the no-running-loop branch must not
    run it inline — callers here include the synchronous
    CallSessionService.update_call_session_status, which must not be held
    open for minutes on a Salesforce outage.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(
            target=run_post_call_writeback, args=(call_session_id,), daemon=True
        ).start()
        return
    asyncio.create_task(_run_post_call_writeback_in_thread(call_session_id))
