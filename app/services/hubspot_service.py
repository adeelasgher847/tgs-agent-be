"""
HubSpot CRM integration — OAuth 2.0, contact lookup, and post-call write-back.

External HTTP calls: HubSpot OAuth (api.hubapi.com/oauth/v1/token), CRM Search API
(crm/v3/objects/contacts/search), and Engagements/Calls API (crm/v3/objects/calls).

Rate limit: HubSpot allows 110 requests/10s per account — _request_with_backoff
retries on 429 with exponential backoff (honoring Retry-After when present).

Every public entrypoint used at call time (contact lookup, CRM context injection,
post-call write-back) fails open: HubSpot being down or rate-limited logs a
warning and returns None/"" rather than raising, so a call is never blocked.
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
from app.core.db_encryption import decrypt_hubspot_token, encrypt_hubspot_token
from app.core.logger import logger
from app.core.secret_manager import get_hubspot_oauth_credentials
from app.db.session import SessionLocal
from app.models.call_session import CallSession
from app.models.workspace_integration import WorkspaceIntegration
from app.utils.redis_client import get_redis

AUTHORIZE_URL = "https://app.hubspot.com/oauth/authorize"
TOKEN_URL = "https://api.hubapi.com/oauth/v1/token"
REVOKE_URL_TEMPLATE = "https://api.hubapi.com/oauth/v1/refresh-tokens/{token}"
SEARCH_CONTACTS_URL = "https://api.hubapi.com/crm/v3/objects/contacts/search"
CREATE_CALL_URL = "https://api.hubapi.com/crm/v3/objects/calls"

SCOPES = "crm.objects.contacts.read crm.objects.contacts.write"

# HubSpot-documented default association type ID for "call to contact"
# (HUBSPOT_DEFINED associations table). Stable across portals.
HUBSPOT_CALL_TO_CONTACT_ASSOCIATION_TYPE_ID = 194

PROVIDER = "hubspot"

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0

_STATE_PURPOSE = "hubspot_oauth_state"
_STATE_TTL_MINUTES = 10

_CONTACT_CACHE_TTL_SECONDS = 300  # 5 minutes, per acceptance criteria
_CONTACT_NOT_FOUND_SENTINEL = "__not_found__"

_DEFAULT_CONTACT_LOOKUP_ENABLED = True
_DEFAULT_WRITE_BACK_ENABLED = True

_SUMMARY_CACHE_TTL_SECONDS = 3600  # 1 hour — long enough to cover writeback retries

_HS_CALL_STATUS_MAP = {
    "completed": "COMPLETED",
    "failed": "FAILED",
    "busy": "BUSY",
    "no_answer": "NO_ANSWER",
}


# ─── HTTP with backoff ────────────────────────────────────────────────────────


async def _request_with_backoff(method: str, url: str, **kwargs) -> httpx.Response:
    """Call HubSpot with exponential backoff on 429 (1s, 2s, 4s, 8s, 16s)."""
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
                "HubSpot 429 on %s %s; retrying in %.1fs (attempt %d/%d)",
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
    return settings.HUBSPOT_REDIRECT_URI or (
        f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/api/v1/integrations/hubspot/callback"
    )


def build_authorization_url(state: str) -> str:
    client_id, _ = get_hubspot_oauth_credentials()
    params = {
        "client_id": client_id,
        "redirect_uri": get_redirect_uri(),
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ─── Token exchange / refresh ─────────────────────────────────────────────────


async def exchange_code_for_tokens(code: str) -> dict:
    client_id, client_secret = get_hubspot_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        TOKEN_URL,
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
    client_id, client_secret = get_hubspot_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        TOKEN_URL,
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


def tenant_has_hubspot_connected(db: Session, tenant_id: uuid.UUID) -> bool:
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
    expires_in = int(token_response.get("expires_in", 1800))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    row = get_integration(db, tenant_id)
    if row is None:
        row = WorkspaceIntegration(workspace_id=tenant_id, provider=PROVIDER)

    row.access_token = encrypt_hubspot_token(access_token, db)
    if refresh_token:
        row.refresh_token = encrypt_hubspot_token(refresh_token, db)
    row.token_expires_at = expires_at

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def get_valid_access_token(db: Session, tenant_id: uuid.UUID) -> Optional[str]:
    """Return a usable access token, refreshing it first if it's expired (or near-expiry)."""
    row = get_integration(db, tenant_id)
    if row is None or not row.access_token:
        return None

    expires_at = row.token_expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if expires_at is not None and expires_at > now + timedelta(seconds=60):
        return decrypt_hubspot_token(row.access_token, db)

    if not row.refresh_token:
        logger.warning(
            "HubSpot access token expired for tenant=%s but no refresh_token stored",
            tenant_id,
        )
        return None

    try:
        refresh_token_plain = decrypt_hubspot_token(row.refresh_token, db)
        token_response = await refresh_access_token(refresh_token_plain)
    except Exception:
        logger.warning(
            "HubSpot token refresh failed for tenant=%s", tenant_id, exc_info=True
        )
        return None

    row = upsert_tokens(db, tenant_id, token_response)
    return decrypt_hubspot_token(row.access_token, db)


def get_field_mappings(db: Session, tenant_id: uuid.UUID) -> list[dict]:
    row = get_integration(db, tenant_id)
    if row is None or not row.extra_metadata:
        return []
    return row.extra_metadata.get("field_mappings") or []


def save_field_mappings(
    db: Session, tenant_id: uuid.UUID, mappings: list[dict]
) -> WorkspaceIntegration:
    """Persist the tenant's HubSpot field -> prompt variable mappings. Raises ValueError if not connected."""
    row = get_integration(db, tenant_id)
    if row is None:
        raise ValueError("HubSpot is not connected for this workspace")

    updated_metadata = dict(row.extra_metadata or {})
    updated_metadata["field_mappings"] = mappings
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_integration_settings(db: Session, tenant_id: uuid.UUID) -> dict:
    """Return the tenant's HubSpot connection status, toggles, and field mappings, with defaults applied."""
    row = get_integration(db, tenant_id)
    if row is None:
        return {
            "connected": False,
            "connected_at": None,
            "contact_lookup_enabled": _DEFAULT_CONTACT_LOOKUP_ENABLED,
            "write_back_enabled": _DEFAULT_WRITE_BACK_ENABLED,
            "field_mappings": [],
        }

    metadata = row.extra_metadata or {}
    return {
        "connected": True,
        "connected_at": row.created_at,
        "contact_lookup_enabled": metadata.get(
            "contact_lookup_enabled", _DEFAULT_CONTACT_LOOKUP_ENABLED
        ),
        "write_back_enabled": metadata.get(
            "write_back_enabled", _DEFAULT_WRITE_BACK_ENABLED
        ),
        "field_mappings": metadata.get("field_mappings") or [],
    }


def update_integration_settings(
    db: Session,
    tenant_id: uuid.UUID,
    *,
    contact_lookup_enabled: bool,
    write_back_enabled: bool,
) -> WorkspaceIntegration:
    """Persist the contact-lookup / write-back toggles. Raises ValueError if not connected."""
    row = get_integration(db, tenant_id)
    if row is None:
        raise ValueError("HubSpot is not connected for this workspace")

    updated_metadata = dict(row.extra_metadata or {})
    updated_metadata["contact_lookup_enabled"] = contact_lookup_enabled
    updated_metadata["write_back_enabled"] = write_back_enabled
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def set_last_write_back_error(
    db: Session, tenant_id: uuid.UUID, error: Optional[str]
) -> None:
    """Record (or clear, when error is None) the last post-call write-back failure."""
    row = get_integration(db, tenant_id)
    if row is None:
        return

    updated_metadata = dict(row.extra_metadata or {})
    if error:
        updated_metadata["last_write_back_error"] = error
    else:
        updated_metadata.pop("last_write_back_error", None)
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()


async def disconnect(db: Session, tenant_id: uuid.UUID) -> bool:
    """Revoke the refresh token at HubSpot (best-effort) and delete the local row."""
    row = get_integration(db, tenant_id)
    if row is None:
        return False

    if row.refresh_token:
        try:
            refresh_token_plain = decrypt_hubspot_token(row.refresh_token, db)
            await _request_with_backoff(
                "DELETE", REVOKE_URL_TEMPLATE.format(token=refresh_token_plain)
            )
        except Exception:
            logger.warning(
                "HubSpot token revoke failed (continuing with local disconnect) tenant=%s",
                tenant_id,
                exc_info=True,
            )

    db.delete(row)
    db.commit()
    return True


# ─── Contact lookup (CRM Search API, Redis-cached) ────────────────────────────


def normalize_to_e164(phone: str) -> str:
    """Normalize phone number to E.164 format if possible.
    E.164 format: +[country_code][subscriber_number] up to 15 digits total.
    """
    if not phone:
        return ""
    # Strip any leading/trailing whitespace
    phone = phone.strip()
    
    # Check if it has a leading '+'
    has_plus = phone.startswith("+")
    
    # Extract only digits
    digits = "".join(c for c in phone if c.isdigit())
    
    if not digits:
        return phone
        
    if has_plus:
        return f"+{digits}"
        
    # If no leading '+', handle common cases:
    # 10 digits: assume US number, prepend +1
    if len(digits) == 10:
        return f"+1{digits}"
    # 11 digits starting with 1: prepend +
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    # Otherwise, just prepend '+'
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
        # If it's a US number with country code, add the 10-digit national number
        if len(digits) == 11 and digits.startswith("1"):
            values.add(digits[1:])
        # If it's a 10-digit number, also add with country code '1' (without '+')
        elif len(digits) == 10:
            values.add(f"1{digits}")
            
    return sorted(list(values))


def _contact_cache_key(tenant_id: uuid.UUID, phone: str) -> str:
    # Hash the phone number to avoid storing PII in plaintext in Redis keys.
    phone_hash = hashlib.sha256(phone.encode("utf-8")).hexdigest()
    return f"hubspot:contact:{tenant_id}:{phone_hash}"


def _contact_dict_from_hubspot(raw: dict) -> dict:
    props = raw.get("properties", {}) or {}
    first = (props.get("firstname") or "").strip()
    last = (props.get("lastname") or "").strip()
    name = f"{first} {last}".strip() or None
    last_interaction = props.get("notes_last_contacted") or props.get("lastmodifieddate")
    return {
        "id": raw.get("id"),
        "name": name,
        "email": props.get("email"),
        "company": props.get("company"),
        "last_interaction_date": last_interaction,
    }


async def search_contact_by_phone(
    access_token: str, phone: str, extra_properties: Optional[list[str]] = None
) -> Optional[dict]:
    search_vals = _get_phone_search_values(phone)
    if not search_vals:
        return None

    properties = [
        "firstname",
        "lastname",
        "email",
        "company",
        "phone",
        "mobilephone",
        "notes_last_contacted",
        "lastmodifieddate",
    ]
    for prop in extra_properties or []:
        if prop and prop not in properties:
            properties.append(prop)

    # Construct filter groups to search both `phone` and `mobilephone` fields
    # against all exact-match variations (joined with OR logic).
    filter_groups = []
    for val in search_vals:
        filter_groups.append({
            "filters": [
                {
                    "propertyName": "phone",
                    "operator": "EQ",
                    "value": val,
                }
            ]
        })
        filter_groups.append({
            "filters": [
                {
                    "propertyName": "mobilephone",
                    "operator": "EQ",
                    "value": val,
                }
            ]
        })

    response = await _request_with_backoff(
        "POST",
        SEARCH_CONTACTS_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "filterGroups": filter_groups,
            "properties": properties,
            "limit": 1,
        },
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    return results[0] if results else None


async def get_contact_for_phone(
    db: Session, tenant_id: uuid.UUID, phone: str
) -> Optional[dict]:
    """Look up a contact by phone, Redis-cached for 5 minutes. Fails open on any error."""
    # Normalize phone numbers to E.164 format if possible before lookup/caching.
    normalized_phone = normalize_to_e164(phone)
    redis_client = get_redis()
    cache_key = _contact_cache_key(tenant_id, normalized_phone)

    if redis_client is not None:
        try:
            cached = await redis_client.get(cache_key)
            if cached is not None:
                return None if cached == _CONTACT_NOT_FOUND_SENTINEL else json.loads(cached)
        except Exception:
            logger.warning("HubSpot contact cache read failed (continuing)", exc_info=True)

    try:
        field_mappings = get_field_mappings(db, tenant_id)
        extra_properties = [
            m.get("hubspot_field")
            for m in field_mappings
            if isinstance(m, dict) and m.get("hubspot_field")
        ]
    except Exception:
        extra_properties = []

    try:
        access_token = await get_valid_access_token(db, tenant_id)
        if not access_token:
            return None
        raw_contact = await search_contact_by_phone(
            access_token, normalized_phone, extra_properties=extra_properties
        )
    except Exception:
        logger.warning(
            "HubSpot contact search failed for tenant=%s (failing open)",
            tenant_id,
            exc_info=True,
        )
        return None

    contact = _contact_dict_from_hubspot(raw_contact) if raw_contact else None
    if contact is not None:
        contact["raw_properties"] = raw_contact.get("properties") or {}

    if redis_client is not None:
        try:
            payload = (
                _CONTACT_NOT_FOUND_SENTINEL if contact is None else json.dumps(contact)
            )
            await redis_client.set(cache_key, payload, ex=_CONTACT_CACHE_TTL_SECONDS)
        except Exception:
            logger.warning("HubSpot contact cache write failed (non-critical)", exc_info=True)

    return contact


# ─── CRM context injection (conversation_orchestrator) ────────────────────────


async def get_crm_context_block_for_call(db: Session, call_session: CallSession) -> str:
    """
    Fetch the CRM context block for a call's system prompt, once per call.

    Cached in call_session.call_metadata["hubspot_crm_context"] so subsequent
    turns (the prompt is rebuilt every turn) reuse the cached string instead of
    re-querying HubSpot/Redis. Fails open — returns "" on any error so a CRM
    outage never blocks the call.
    """
    metadata = call_session.call_metadata or {}
    if "hubspot_crm_context" in metadata:
        return metadata.get("hubspot_crm_context") or ""

    context_block = ""
    try:
        if call_session.customer_phone_number and tenant_has_hubspot_connected(
            db, call_session.tenant_id
        ):
            contact = await get_contact_for_phone(
                db, call_session.tenant_id, call_session.customer_phone_number
            )
            if contact:
                context_block = (
                    "# CRM CONTEXT\n"
                    f"CRM CONTEXT: Caller name: {contact.get('name') or 'Unknown'}, "
                    f"Company: {contact.get('company') or 'Unknown'}, "
                    f"Last interaction: {contact.get('last_interaction_date') or 'Unknown'}."
                )
    except Exception:
        logger.warning(
            "HubSpot CRM context lookup failed (continuing without CRM context)",
            exc_info=True,
        )
        context_block = ""

    try:
        # Use a nested transaction savepoint to isolate database flushing.
        # This keeps the helper fail-open and avoids committing the main transaction
        # during prompt generation.
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["hubspot_crm_context"] = context_block
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache HubSpot CRM context on call_session (non-critical)",
            exc_info=True,
        )

    return context_block


# ─── Custom field mapping (prompt-variable injection) ─────────────────────────


def resolve_field_mapping_values(
    contact: Optional[dict], field_mappings: list[dict]
) -> dict[str, str]:
    """Map configured HubSpot fields to prompt-variable values from a contact record."""
    if not contact or not field_mappings:
        return {}

    raw_properties = contact.get("raw_properties") or {}
    values: dict[str, str] = {}
    for mapping in field_mappings:
        hubspot_field = mapping.get("hubspot_field")
        prompt_variable = mapping.get("prompt_variable")
        if not hubspot_field or not prompt_variable:
            continue
        value = raw_properties.get(hubspot_field)
        if value is None:
            value = contact.get(hubspot_field)
        if value is not None:
            values[prompt_variable] = str(value)
    return values


async def get_field_mapping_values_for_call(
    db: Session, call_session: CallSession
) -> dict[str, str]:
    """
    Resolve configured HubSpot field mappings to prompt-variable values for a call.

    Cached in call_session.call_metadata["hubspot_field_mapping_values"] so it's
    computed once per call (mirrors get_crm_context_block_for_call). Fails open —
    returns {} on any error so a CRM outage never blocks the call.
    """
    metadata = call_session.call_metadata or {}
    if "hubspot_field_mapping_values" in metadata:
        return metadata.get("hubspot_field_mapping_values") or {}

    values: dict[str, str] = {}
    try:
        if call_session.customer_phone_number and tenant_has_hubspot_connected(
            db, call_session.tenant_id
        ):
            integration_settings = get_integration_settings(db, call_session.tenant_id)
            field_mappings = integration_settings["field_mappings"]
            if integration_settings["contact_lookup_enabled"] and field_mappings:
                contact = await get_contact_for_phone(
                    db, call_session.tenant_id, call_session.customer_phone_number
                )
                values = resolve_field_mapping_values(contact, field_mappings)
    except Exception:
        logger.warning(
            "HubSpot field mapping resolution failed (continuing without field mappings)",
            exc_info=True,
        )
        values = {}

    try:
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["hubspot_field_mapping_values"] = values
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache HubSpot field mapping values on call_session (non-critical)",
            exc_info=True,
        )

    return values


def apply_field_mapping_values(prompt: str, values: dict[str, str]) -> str:
    """Replace `{prompt_variable}` placeholders in the prompt text with resolved values."""
    if not values:
        return prompt
    for prompt_variable, value in values.items():
        prompt = prompt.replace("{" + prompt_variable + "}", value)
    return prompt


# ─── Post-call write-back (Engagements/Calls API + Gemini summary) ────────────


def _hs_call_status(status: Optional[str]) -> str:
    return _HS_CALL_STATUS_MAP.get((status or "").lower(), "COMPLETED")


def _hs_call_direction(call_type: Optional[str]) -> str:
    return "INBOUND" if (call_type or "").lower() == "inbound" else "OUTBOUND"


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
            "HubSpot post-call summary generation failed (continuing without summary)",
            exc_info=True,
        )
        return ""


def get_cached_transcript_summary(db: Session, call_session: CallSession) -> str:
    """
    2-sentence Gemini summary of the call transcript, cached on
    call_session.call_metadata["hubspot_call_summary"] so a retried write-back
    never calls Gemini twice for the same call.
    """
    metadata = call_session.call_metadata or {}
    if "hubspot_call_summary" in metadata:
        return metadata.get("hubspot_call_summary") or ""

    summary = generate_transcript_summary(db, call_session)

    try:
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata["hubspot_call_summary"] = summary
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache HubSpot call summary on call_session (non-critical)",
            exc_info=True,
        )

    return summary


async def create_call_engagement(
    access_token: str,
    contact_id: str,
    *,
    occurred_at: datetime,
    duration_seconds: int,
    direction: str,
    hs_status: str,
    title: str,
    body_text: str,
) -> dict:
    payload = {
        "properties": {
            "hs_timestamp": str(int(occurred_at.timestamp() * 1000)),
            "hs_call_title": title,
            "hs_call_body": body_text,
            "hs_call_duration": str(int(duration_seconds * 1000)),
            "hs_call_direction": direction,
            "hs_call_status": hs_status,
        },
        "associations": [
            {
                "to": {"id": contact_id},
                "types": [
                    {
                        "associationCategory": "HUBSPOT_DEFINED",
                        "associationTypeId": HUBSPOT_CALL_TO_CONTACT_ASSOCIATION_TYPE_ID,
                    }
                ],
            }
        ],
    }
    response = await _request_with_backoff(
        "POST",
        CREATE_CALL_URL,
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
            "HubSpot write-back skipped — disabled in settings for tenant=%s",
            tenant_id,
        )
        return

    access_token = await get_valid_access_token(db, tenant_id)
    if not access_token:
        return

    contact = await get_contact_for_phone(
        db, tenant_id, call_session.customer_phone_number
    )
    if not contact or not contact.get("id"):
        logger.info(
            "HubSpot write-back skipped — no matching contact for session=%s",
            call_session.id,
        )
        return

    summary = get_cached_transcript_summary(db, call_session)
    body_text = summary or "Call completed — no transcript summary available."

    try:
        await create_call_engagement(
            access_token,
            contact["id"],
            occurred_at=call_session.start_time or datetime.now(timezone.utc),
            duration_seconds=call_session.duration or 0,
            direction=_hs_call_direction(call_session.call_type),
            hs_status=_hs_call_status(call_session.status),
            title=f"Voice agent call — {call_session.status}",
            body_text=body_text,
        )
    except Exception as exc:
        logger.error(
            "HubSpot call engagement creation failed for session=%s: %s",
            call_session.id,
            exc,
            exc_info=True,
        )
        set_last_write_back_error(db, tenant_id, _safe_error_msg(exc))
        return

    set_last_write_back_error(db, tenant_id, None)
    logger.info(
        "HubSpot call engagement created for session=%s contact=%s",
        call_session.id,
        contact["id"],
    )


def run_post_call_writeback(call_session_id: uuid.UUID) -> None:
    """Sync entrypoint — opens its own session, mirrors sync_inbound_call_to_crm."""
    db: Session = SessionLocal()
    try:
        call_session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
        if not call_session or not call_session.customer_phone_number:
            return
        if not tenant_has_hubspot_connected(db, call_session.tenant_id):
            return

        asyncio.run(_run_post_call_writeback_async(db, call_session))
    except Exception:
        logger.warning(
            "HubSpot post-call write-back failed (non-critical) session=%s",
            call_session_id,
            exc_info=True,
        )
    finally:
        db.close()


async def _run_post_call_writeback_in_thread(call_session_id: uuid.UUID) -> None:
    await asyncio.to_thread(run_post_call_writeback, call_session_id)


def schedule_hubspot_writeback(call_session_id: uuid.UUID) -> None:
    """Fire-and-forget post-call write-back. Never blocks the caller."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        run_post_call_writeback(call_session_id)
        return
    asyncio.create_task(_run_post_call_writeback_in_thread(call_session_id))
