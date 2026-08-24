"""
Slack integration — OAuth v2, channel listing, and post-call summary posting.

External HTTP calls: Slack OAuth (slack.com/api/oauth.v2.access),
conversations.list (fetch channels), chat.postMessage (post a summary), and
auth.revoke (best-effort token revocation on disconnect).

Rate limiting: Slack's Web API returns 429 with a `Retry-After` header when a
tenant's app exceeds its per-method rate limit — `_request_with_backoff`
retries on 429 with exponential backoff (honoring Retry-After when present),
mirroring `hubspot_service.py`.

`send_call_summary` (the call-time write path) fails open: a Slack outage or
misconfiguration never blocks or affects call completion. `list_channels`
(an admin-facing settings action, not call-time) raises on failure instead —
mirroring how HubSpot's field-mapping property fetch (also an admin action)
raises rather than fails open, since silently returning an empty channel list
would look like "workspace has no channels" rather than "Slack is down".
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.db_encryption import decrypt_slack_token, encrypt_slack_token
from app.core.logger import logger
from app.core.secret_manager import get_slack_oauth_credentials
from app.db.session import SessionLocal
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.workspace_integration import WorkspaceIntegration

AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
TOKEN_URL = "https://slack.com/api/oauth.v2.access"
REVOKE_URL = "https://slack.com/api/auth.revoke"
CONVERSATIONS_LIST_URL = "https://slack.com/api/conversations.list"
POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

# Minimal bot-token scopes needed to list channels and post the summary message.
# chat:write.public is required to post to a public channel the bot hasn't been
# manually invited to — conversations.list surfaces every public channel in the
# workspace regardless of bot membership, so without this scope most selected
# channels would fail to receive messages with a silent (fail-open) not_in_channel
# error. Private channels still require a manual /invite regardless of scope —
# that's a Slack platform restriction with no API workaround.
SCOPES = "channels:read,groups:read,chat:write,chat:write.public"

PROVIDER = "slack"

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0

# Caps the admin channel-picker fetch at 3 pages of 200 (600 channels) so a
# workspace with thousands of channels can't turn GET /channels into a
# multi-second blocking fan-out. The dashboard picker only needs a
# reasonably-sized list to select from, not the full workspace inventory.
_MAX_CHANNEL_LIST_PAGES = 3

_STATE_PURPOSE = "slack_oauth_state"
_STATE_TTL_MINUTES = 10

_MAX_ERROR_LEN = 500


# ─── HTTP with backoff ────────────────────────────────────────────────────────


async def _request_with_backoff(
    method: str,
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    **kwargs,
) -> httpx.Response:
    """Call Slack with exponential backoff on 429 (1s, 2s, 4s, 8s, 16s).

    Pass an existing ``client`` to reuse a connection across multiple calls
    (e.g. paginated ``conversations.list`` requests); otherwise a short-lived
    client is opened and closed for this single call.
    """
    if client is not None:
        return await _request_with_backoff_using_client(client, method, url, **kwargs)
    async with httpx.AsyncClient(timeout=20.0) as owned_client:
        return await _request_with_backoff_using_client(
            owned_client, method, url, **kwargs
        )


async def _request_with_backoff_using_client(
    client: httpx.AsyncClient, method: str, url: str, **kwargs
) -> httpx.Response:
    attempt = 0
    while True:
        response = await client.request(method, url, **kwargs)
        if response.status_code != 429 or attempt >= _MAX_RETRIES:
            return response

        retry_after_header = response.headers.get("Retry-After")
        wait_seconds = (
            float(retry_after_header)
            if retry_after_header
            else _BASE_BACKOFF_SECONDS * (2**attempt)
        )
        logger.warning(
            "Slack 429 on %s %s; retrying in %.1fs (attempt %d/%d)",
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
        payload = jwt.decode(
            state, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
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
    return settings.SLACK_REDIRECT_URI or (
        f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/api/v1/integrations/slack/callback"
    )


def get_authorization_url(tenant_id: uuid.UUID) -> str:
    client_id, _ = get_slack_oauth_credentials()
    state = build_oauth_state(tenant_id)
    params = {
        "client_id": client_id,
        "redirect_uri": get_redirect_uri(),
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ─── Token exchange ────────────────────────────────────────────────────────────


async def exchange_code_for_tokens(code: str) -> dict:
    """
    POST to Slack's oauth.v2.access. Slack always returns HTTP 200 even on
    failure, signalling errors via `ok: false` + `error` in the JSON body —
    unlike HubSpot's token endpoint, which uses HTTP status codes.
    """
    client_id, client_secret = get_slack_oauth_credentials()
    response = await _request_with_backoff(
        "POST",
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": get_redirect_uri(),
            "code": code,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise ValueError(f"Slack OAuth token exchange failed: {payload.get('error')}")
    return payload


# ─── WorkspaceIntegration CRUD ────────────────────────────────────────────────


def get_integration(db: Session, tenant_id: uuid.UUID) -> WorkspaceIntegration | None:
    return (
        db.query(WorkspaceIntegration)
        .filter(
            WorkspaceIntegration.workspace_id == tenant_id,
            WorkspaceIntegration.provider == PROVIDER,
        )
        .first()
    )


def tenant_has_slack_connected(db: Session, tenant_id: uuid.UUID) -> bool:
    return get_integration(db, tenant_id) is not None


def upsert_integration(
    db: Session, tenant_id: uuid.UUID, token_response: dict
) -> WorkspaceIntegration:
    """
    Encrypt+store the bot token, plus team_id/team_name in extra_metadata.

    Slack bot tokens (`xoxb-...`) issued via oauth.v2.access do not expire and
    there is no refresh_token in the response, unlike HubSpot's OAuth flow.
    """
    access_token = token_response["access_token"]
    team = token_response.get("team") or {}

    row = get_integration(db, tenant_id)
    if row is None:
        row = WorkspaceIntegration(workspace_id=tenant_id, provider=PROVIDER)

    row.access_token = encrypt_slack_token(access_token, db)

    updated_metadata = dict(row.extra_metadata or {})
    updated_metadata["team_id"] = team.get("id")
    updated_metadata["team_name"] = team.get("name")
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def get_valid_access_token(db: Session, tenant_id: uuid.UUID) -> str | None:
    """
    Return the decrypted bot token, or None if not connected.

    Slack bot tokens don't expire, so there's no refresh cycle — this helper
    exists purely to centralize the decrypt call so future rotation support
    (e.g. token rotation via Slack's newer OAuth flows) only touches one place.
    """
    row = get_integration(db, tenant_id)
    if row is None or not row.access_token:
        return None
    return decrypt_slack_token(row.access_token, db)


def get_connection_status(
    db: Session, tenant_id: uuid.UUID
) -> tuple[bool, datetime | None]:
    row = get_integration(db, tenant_id)
    if row is None:
        return False, None
    return True, row.created_at


def get_integration_status(db: Session, tenant_id: uuid.UUID) -> dict:
    row = get_integration(db, tenant_id)
    if row is None:
        return {
            "connected": False,
            "connected_at": None,
            "team_name": None,
            "default_channel_id": None,
            "default_channel_name": None,
        }

    metadata = row.extra_metadata or {}
    return {
        "connected": True,
        "connected_at": row.created_at,
        "team_name": metadata.get("team_name"),
        "default_channel_id": metadata.get("default_channel_id"),
        "default_channel_name": metadata.get("default_channel_name"),
    }


def set_default_channel(
    db: Session, tenant_id: uuid.UUID, channel_id: str | None, channel_name: str | None
) -> WorkspaceIntegration:
    """Persist (or clear, when both args are None) the workspace's default Slack channel."""
    row = get_integration(db, tenant_id)
    if row is None:
        raise ValueError("Slack is not connected for this workspace")

    updated_metadata = dict(row.extra_metadata or {})
    if channel_id:
        updated_metadata["default_channel_id"] = channel_id
        updated_metadata["default_channel_name"] = channel_name
    else:
        updated_metadata.pop("default_channel_id", None)
        updated_metadata.pop("default_channel_name", None)
    row.extra_metadata = updated_metadata
    flag_modified(row, "extra_metadata")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def disconnect(db: Session, tenant_id: uuid.UUID) -> bool:
    """Revoke the bot token at Slack (best-effort) and delete the local row."""
    row = get_integration(db, tenant_id)
    if row is None:
        return False

    try:
        access_token = decrypt_slack_token(row.access_token, db)
        await _request_with_backoff(
            "POST",
            REVOKE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    except Exception:
        logger.warning(
            "Slack token revoke failed (continuing with local disconnect) tenant=%s",
            tenant_id,
            exc_info=True,
        )

    db.delete(row)
    db.commit()
    return True


# ─── Channel listing (admin-facing settings action — raises on failure) ───────


async def list_channels(db: Session, tenant_id: uuid.UUID) -> list[dict]:
    """
    Fetch the tenant's available Slack channels via conversations.list.

    Admin-facing settings action, not a call-time path — raises ValueError on
    any failure (Slack not connected, API error) instead of failing open, so
    the dashboard can surface a clear error rather than silently showing an
    empty channel list.
    """
    access_token = await get_valid_access_token(db, tenant_id)
    if not access_token:
        raise ValueError("Slack is not connected for this workspace")

    channels: list[dict] = []
    cursor: str | None = None
    page = 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        while True:
            params = {
                "types": "public_channel,private_channel",
                "exclude_archived": "true",
                "limit": "200",
            }
            if cursor:
                params["cursor"] = cursor

            response = await _request_with_backoff(
                "GET",
                CONVERSATIONS_LIST_URL,
                client=client,
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise ValueError(
                    f"Slack conversations.list failed: {payload.get('error')}"
                )

            for ch in payload.get("channels") or []:
                channels.append({"id": ch.get("id"), "name": ch.get("name")})

            page += 1
            cursor = (payload.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor or page >= _MAX_CHANNEL_LIST_PAGES:
                break

    return channels


# ─── Post-call summary send (chat.postMessage) ─────────────────────────────────


def _resolve_channel(
    call_flow: CallFlow, integration: WorkspaceIntegration
) -> str | None:
    if call_flow.slack_channel_id:
        return call_flow.slack_channel_id
    metadata = integration.extra_metadata or {}
    return metadata.get("default_channel_id")


def _sentiment_and_summary(call_session: CallSession) -> tuple[str, str | None]:
    """
    Pull the summary/sentiment computed by voice_analysis_service.analyze_call_transcript
    (cached on call_session.call_metadata["llm_call_analysis"]). Lazily computes it
    (serialized against the automatic ARQ job via the same pg advisory lock) if it
    isn't cached yet — mirrors post_call_email_service._build_email_content /
    _send_post_call_email_summary_impl.
    """
    analysis_data: dict = {}
    if call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata:
        analysis_data = (
            call_session.call_metadata["llm_call_analysis"].get("analysis") or {}
        )

    summary = (
        call_session.transcript_summary
        or analysis_data.get("summary")
        or "No summary available."
    )
    sentiment = analysis_data.get("sentiment")
    return summary, sentiment


async def _post_message(
    access_token: str, channel: str, text: str, blocks: list[dict]
) -> None:
    response = await _request_with_backoff(
        "POST",
        POST_MESSAGE_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        json={"channel": channel, "text": text, "blocks": blocks},
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise ValueError(f"Slack chat.postMessage failed: {payload.get('error')}")


def _safe_error_msg(exc: Exception) -> str:
    msg = str(exc)
    return msg[:_MAX_ERROR_LEN]


async def _send_call_summary_async(db: Session, call_session_id: uuid.UUID) -> None:
    call_session = (
        db.query(CallSession).filter(CallSession.id == call_session_id).first()
    )
    if not call_session:
        return

    if not call_session.call_flow_id:
        return
    call_flow = (
        db.query(CallFlow).filter(CallFlow.id == call_session.call_flow_id).first()
    )
    if not call_flow or not call_flow.slack_summary_enabled:
        return

    integration = get_integration(db, call_session.tenant_id)
    if integration is None:
        return

    channel = _resolve_channel(call_flow, integration)
    if not channel:
        logger.info(
            "Slack summary skipped — no channel configured for session=%s",
            call_session_id,
        )
        return

    try:
        access_token = decrypt_slack_token(integration.access_token, db)
    except Exception:
        logger.warning(
            "Slack token decryption failed for tenant=%s (non-critical)",
            call_session.tenant_id,
            exc_info=True,
        )
        return

    # Lazily compute the transcript summary/sentiment if the automatic
    # generate_call_summary ARQ job hasn't finished yet — serializes on the
    # same pg advisory lock so it never races/duplicates that (paid) LLM call.
    if not (
        call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata
    ):
        try:
            from app.services.voice_analysis_service import ensure_call_summary_locked

            ensure_call_summary_locked(db, call_session, raise_on_no_transcript=False)
        except Exception:
            logger.warning(
                "Slack summary: analysis generation failed (non-critical) session=%s",
                call_session_id,
                exc_info=True,
            )

    summary, sentiment = _sentiment_and_summary(call_session)

    phone = call_session.customer_phone_number or "Unknown"
    duration = call_session.duration
    duration_text = f"{duration}s" if duration is not None else "Unknown"

    text = f"Call summary — {phone}"
    fields = [
        {"type": "mrkdwn", "text": f"*Caller:*\n{phone}"},
        {"type": "mrkdwn", "text": f"*Status:*\n{call_session.status}"},
        {"type": "mrkdwn", "text": f"*Duration:*\n{duration_text}"},
    ]
    if sentiment:
        fields.append({"type": "mrkdwn", "text": f"*Sentiment:*\n{sentiment}"})

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "Call Summary"}},
        {"type": "section", "fields": fields},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:*\n{summary}"},
        },
    ]

    await _post_message(access_token, channel, text, blocks)
    logger.info(
        "Slack call summary posted for session=%s channel=%s", call_session_id, channel
    )


def send_call_summary(call_session_id: uuid.UUID) -> None:
    """
    Sync entrypoint — opens its own session, mirrors hubspot_service's
    run_post_call_writeback. Entirely fail-open: never lets a failure here
    affect call completion.
    """
    db: Session = SessionLocal()
    try:
        asyncio.run(_send_call_summary_async(db, call_session_id))
    except Exception as exc:
        logger.warning(
            "Slack post-call summary send failed (non-critical) session=%s: %s",
            call_session_id,
            _safe_error_msg(exc),
            exc_info=True,
        )
    finally:
        db.close()


async def _send_call_summary_in_thread(call_session_id: uuid.UUID) -> None:
    await asyncio.to_thread(send_call_summary, call_session_id)


def schedule_slack_summary(call_session_id: uuid.UUID) -> None:
    """
    Fire-and-forget post-call Slack summary send. Never blocks the caller.

    Mirrors hubspot_service.schedule_hubspot_writeback's dispatch pattern:
    asyncio.create_task + asyncio.to_thread when a loop is running (keeps the
    sync DB work off the event loop), a daemon thread otherwise (e.g. called
    from a sync context with no running loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(
            target=send_call_summary, args=(call_session_id,), daemon=True
        ).start()
        return
    asyncio.create_task(_send_call_summary_in_thread(call_session_id))
