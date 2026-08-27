"""System Webhooks (Call Flow) — dispatch, delivery, and retry.

Four sub-features, three of which fire an actual outbound HTTP call from here:

1. Pre-Inbound Call Webhook — `fetch_pre_inbound_webhook_variables()`. Called
   (by the voice-pipeline agent's follow-up work, not this file) from
   `app/routers/voice.py::handle_incoming_call` before the TwiML response is
   built. Fail-open: any failure path returns `{}`, never raises, never
   blocks the caller beyond `_PRE_INBOUND_TIMEOUT_SECONDS`.
2. Dynamic Inbound Call Routing — a pure config toggle + the `agent_id`
   variable returned by (1); no dispatch logic lives here (voice-pipeline's
   job to wire the routing decision itself).
3. Post-Call Webhook — `schedule_post_call_webhook()` / ARQ job
   `run_post_call_webhook()`.
4. Status Webhook — `schedule_status_webhook()` / ARQ job `run_status_webhook()`.

Shape mirrors `app/services/webhook_service.py` (SSRF-guarded on every
attempt — TOCTOU-safe, httpx.AsyncClient with timeout, isolated DB session
per ARQ delivery) but is call-flow-scoped with `{{...}}` template rendering
instead of that module's workspace-scoped HMAC signing.

Retry policy for (3)/(4): 2 retries at 1 min / 5 min (bounded, lower-stakes
than the generic webhook system's 3 retries) via a single shared ARQ retry
job (`retry_system_webhook_delivery`) parameterized by `kind` — both
post_call and status webhooks re-run their own dispatch function on retry
rather than mutating a stored delivery row (there's no per-attempt row like
`WebhookDelivery`; each `SystemWebhookDeliveryLog` row is one attempt).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db_encryption import decrypt_webhook_headers
from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.models.system_webhook_log import SystemWebhookDeliveryLog
from app.services.system_webhook_field_catalog import build_post_call_payload_context
from app.utils.ssrf import SSRFBlockedError, assert_public_url
from app.utils.webhook_templating import render_json_template, render_template

_MAX_RESPONSE_BODY_CHARS = 500  # matches webhook_service.py, for log consistency
_SSRF_CHECK_TIMEOUT_SECONDS = (
    2.0  # bounds the sync DNS resolution inside assert_public_url
)
_PRE_INBOUND_TIMEOUT_SECONDS = 4.0  # fast — a live inbound call is waiting on TwiML
_PRE_INBOUND_MAX_RESPONSE_BYTES = 65_536  # 64 KiB cap on the raw response body
_PRE_INBOUND_MAX_VARIABLES = 100  # cap on number of variables accepted
_PRE_INBOUND_MAX_VARIABLE_VALUE_CHARS = 4_000  # cap per variable value
_DISPATCH_TIMEOUT_SECONDS = 5.0  # post_call / status — off the call-time critical path
_RETRY_DELAYS_MINUTES = [
    1,
    5,
]  # 2 retries total (lower-stakes than webhook_service's 3)


# ── Shared low-level delivery + logging ─────────────────────────────────────


async def _deliver(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    call_flow_id: uuid.UUID,
    call_session_id: uuid.UUID | None,
    webhook_kind: str,
    event_type: str | None,
    url: str,
    headers: dict[str, str] | None,
    query_params: list[tuple[str, str]] | None,
    json_body: dict | None,
    timeout_seconds: float,
    on_response: Callable[[httpx.Response], None] | None = None,
) -> SystemWebhookDeliveryLog:
    """One outbound delivery attempt, SSRF-guarded fresh on every call
    (TOCTOU-safe, same as `webhook_service._attempt_delivery`), logged to
    `SystemWebhookDeliveryLog` regardless of outcome, and returned.

    `on_response`, if given, is invoked with the raw `httpx.Response` before
    the body is truncated for logging — lets `fetch_pre_inbound_webhook_variables`
    parse the full (possibly >500-char) JSON body without a second HTTP call
    or duplicating this function's SSRF/delivery/logging logic.
    """
    started = time.monotonic()
    delivery_status = "failed"
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None

    try:
        # assert_public_url() does a synchronous, unbounded socket.getaddrinfo()
        # call. Run it off the event loop with its own timeout so a slow/hung
        # DNS resolution for a tenant-supplied hostname can't stall this
        # worker's entire event loop — this fires inline on the Twilio-facing
        # inbound-call request path (fetch_pre_inbound_webhook_variables),
        # not just from a background ARQ job like webhook_service.py's sibling.
        await asyncio.wait_for(
            asyncio.to_thread(assert_public_url, url),
            timeout=_SSRF_CHECK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        error = "SSRF check timed out (DNS resolution too slow)"
        logger.warning(
            "system_webhook: SSRF guard DNS resolution timed out kind=%s call_flow=%s url=%s",
            webhook_kind,
            call_flow_id,
            url,
        )
    except SSRFBlockedError as exc:
        error = f"SSRF blocked: {exc}"[:_MAX_RESPONSE_BODY_CHARS]
        logger.warning(
            "system_webhook: SSRF guard blocked delivery kind=%s call_flow=%s url=%s: %s",
            webhook_kind,
            call_flow_id,
            url,
            exc,
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(
                    url,
                    params=query_params,
                    headers={**(headers or {}), "Content-Type": "application/json"},
                    json=json_body,
                )
            status_code = resp.status_code
            response_body = resp.text[:_MAX_RESPONSE_BODY_CHARS]
            delivery_status = "success" if 200 <= resp.status_code < 300 else "failed"
            if on_response is not None:
                try:
                    on_response(resp)
                except (
                    Exception
                ) as exc:  # defensive — parsing bugs must not fail delivery
                    logger.debug(
                        "system_webhook: on_response callback failed kind=%s: %s",
                        webhook_kind,
                        exc,
                    )
        except httpx.TimeoutException:
            delivery_status = "timeout"
            error = "timeout"
        except httpx.HTTPError as exc:
            delivery_status = "failed"
            error = str(exc)[:_MAX_RESPONSE_BODY_CHARS]

    duration_ms = int((time.monotonic() - started) * 1000)

    log = SystemWebhookDeliveryLog(
        tenant_id=tenant_id,
        call_flow_id=call_flow_id,
        call_session_id=call_session_id,
        webhook_kind=webhook_kind,
        event_type=event_type,
        url=url,
        status=delivery_status,
        status_code=status_code,
        response_body=response_body,
        error=error,
        attempt_count=1,
        duration_ms=duration_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def _decrypt_headers_safe(ciphertext: str | None, db: Session) -> dict[str, str]:
    if not ciphertext:
        return {}
    try:
        return decrypt_webhook_headers(ciphertext, db)
    except ValueError as exc:
        logger.warning("system_webhook: failed to decrypt stored headers: %s", exc)
        return {}


def _render_headers(headers: dict[str, str], context: dict[str, Any]) -> dict[str, str]:
    # Header NAMES are kept literal, not templated: an unresolved template in a
    # header key would silently become "" (send a header with an empty name,
    # which is a protocol error), and there's no legitimate tenant use case for
    # a dynamic header name. Only the value is rendered.
    return {k: render_template(v, context) for k, v in headers.items()}


def _render_query_params(
    raw: list[dict] | None, context: dict[str, Any]
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for p in raw or []:
        key = render_template(str(p.get("key", "")), context)
        value = render_template(str(p.get("value", "")), context)
        result.append((key, value))
    return result


def _strip_metadata_from_dict(obj: Any) -> Any:
    """Recursively strip metadata dictionaries and keys from payloads."""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            k_lower = str(k).lower().strip()
            if k_lower in ("metadata", "call_metadata", "_metadata", "custom_metadata"):
                continue
            cleaned[k] = _strip_metadata_from_dict(v)
        return cleaned
    elif isinstance(obj, list):
        return [_strip_metadata_from_dict(item) for item in obj]
    return obj


def _filter_query_params_metadata(
    params: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Filter out query parameters referencing metadata keys."""
    if not params:
        return []
    filtered: list[tuple[str, str]] = []
    for k, v in params:
        k_lower = str(k).lower().strip()
        if (
            k_lower in ("metadata", "call_metadata", "_metadata", "custom_metadata")
            or k_lower.startswith("metadata[")
            or k_lower.startswith("metadata.")
            or k_lower.startswith("_metadata.")
            or k_lower.startswith("call_metadata.")
            or k_lower.startswith("custom_metadata")
        ):
            continue
        filtered.append((k, v))
    return filtered


# ── (1) Pre-Inbound Call Webhook ────────────────────────────────────────────


async def fetch_pre_inbound_webhook_variables(
    db: Session,
    call_flow: CallFlow,
    from_number: str | None,
    to_number: str | None,
) -> dict[str, str]:
    """Fire the Pre-Inbound Call Webhook and return `{variables}` from its
    JSON response, or `{}` on ANY failure path (never raises). ~4s timeout —
    caller is a live inbound call waiting on TwiML.

    Values in the response's `variables` dict must be strings; non-string
    values are dropped (logged at WARNING) rather than coerced, per spec.
    """
    if not call_flow.pre_inbound_webhook_url:
        return {}

    static_metadata = call_flow.pre_inbound_webhook_static_metadata or {}
    context: dict[str, Any] = {
        "_system": {"phoneNumber": to_number, "fromNumber": from_number},
        "_metadata": static_metadata,
        "_variable": static_metadata,
    }

    url = render_template(call_flow.pre_inbound_webhook_url, context)
    raw_headers = _decrypt_headers_safe(
        call_flow.pre_inbound_webhook_headers_encrypted, db
    )
    headers = _render_headers(raw_headers, context)
    query_params = _render_query_params(
        call_flow.pre_inbound_webhook_query_params, context
    )
    if call_flow.disable_metadata:
        query_params = _filter_query_params_metadata(query_params)

    variables: dict[str, str] = {}

    def _parse_variables(resp: httpx.Response) -> None:
        nonlocal variables
        # Response size guard: this body gets stored unbounded into
        # call_session.call_metadata and re-rendered into every prompt/greeting
        # for the rest of the call, unlike SystemWebhookDeliveryLog's response
        # (which is truncated). A misconfigured or compromised endpoint
        # shouldn't be able to balloon call_metadata or the per-turn render cost.
        if len(resp.content) > _PRE_INBOUND_MAX_RESPONSE_BYTES:
            logger.warning(
                "Pre-inbound webhook: response too large (%d bytes, call_flow=%s) "
                "— ignoring variables",
                len(resp.content),
                call_flow.id,
            )
            return
        try:
            data = resp.json()
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        raw_vars = data.get("variables")
        if not isinstance(raw_vars, dict):
            return
        parsed: dict[str, str] = {}
        for key, value in raw_vars.items():
            if len(parsed) >= _PRE_INBOUND_MAX_VARIABLES:
                logger.warning(
                    "Pre-inbound webhook: variable count cap (%d) reached "
                    "(call_flow=%s) — ignoring remaining keys",
                    _PRE_INBOUND_MAX_VARIABLES,
                    call_flow.id,
                )
                break
            if not isinstance(value, str):
                logger.warning(
                    "Pre-inbound webhook: dropping non-string variable %r "
                    "(call_flow=%s) — values must be strings",
                    key,
                    call_flow.id,
                )
                continue
            parsed[key] = value[:_PRE_INBOUND_MAX_VARIABLE_VALUE_CHARS]
        variables = parsed

    json_body = {
        "from": from_number,
        "to": to_number,
    }
    if not call_flow.disable_metadata:
        json_body["metadata"] = static_metadata

    try:
        await _deliver(
            db,
            tenant_id=call_flow.tenant_id,
            call_flow_id=call_flow.id,
            call_session_id=None,
            webhook_kind="pre_inbound",
            event_type=None,
            url=url,
            headers=headers,
            query_params=query_params,
            json_body=json_body,
            timeout_seconds=_PRE_INBOUND_TIMEOUT_SECONDS,
            on_response=_parse_variables,
        )
    except Exception as exc:  # pragma: no cover - defensive, fail-open by contract
        logger.warning(
            "Pre-inbound webhook fetch raised unexpectedly for call_flow=%s: %s",
            call_flow.id,
            exc,
        )
        return {}

    return variables


# ── (3) Post-Call Webhook ───────────────────────────────────────────────────


def schedule_post_call_webhook(call_session_id: uuid.UUID) -> None:
    """Enqueue the Post-Call Webhook as an ARQ background job. Fire-and-forget
    — never blocks the caller. Fails open if the ARQ pool isn't ready.
    Mirrors `post_call_analysis_service.schedule_run_post_call_analysis` exactly.
    """
    from app.utils.arq_pool import get_arq_pool

    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; post-call webhook skipped for session=%s",
            call_session_id,
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job("run_post_call_webhook", str(call_session_id))
        except Exception as exc:
            logger.warning("Failed to enqueue post-call webhook job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())


async def _dispatch_post_call_webhook(call_session_id: uuid.UUID) -> bool:
    """Build + deliver the Post-Call Webhook payload for one call session.
    Returns True if delivery succeeded OR nothing was configured (no retry
    needed either way); False only on an actual delivery failure/timeout.
    """
    from app.db.session import SessionLocal

    db: Session = SessionLocal()
    try:
        call_session = db.execute(
            select(CallSession).where(CallSession.id == call_session_id)
        ).scalar_one_or_none()
        if not call_session or not call_session.call_flow_id:
            return True

        call_flow = db.execute(
            select(CallFlow).where(
                CallFlow.id == call_session.call_flow_id,
                CallFlow.tenant_id == call_session.tenant_id,
            )
        ).scalar_one_or_none()
        if not call_flow or not call_flow.post_call_webhook_url:
            return True

        call_log = (
            db.execute(
                select(CallLog)
                .where(
                    CallLog.call_session_id == call_session.id,
                    CallLog.tenant_id == call_session.tenant_id,
                )
                .limit(1)
            )
            .scalars()
            .first()
        )

        context = build_post_call_payload_context(db, call_session, call_log)

        if (
            call_flow.post_call_webhook_custom_payload_enabled
            and call_flow.post_call_webhook_custom_payload_template
        ):
            payload = render_json_template(
                call_flow.post_call_webhook_custom_payload_template, context
            )
            if call_flow.disable_metadata:
                payload = _strip_metadata_from_dict(payload)
        else:
            # Default shape: {callId, agentId, timestamp, data}. `data` is the
            # call_metadata + analytics namespaces flattened together — a
            # reasonable, documented subset rather than the full catalog
            # (conversation_data/transcript/header_variables are still
            # reachable via the custom-payload template above).
            if call_flow.disable_metadata:
                data = _strip_metadata_from_dict(context.get("analytics", {}))
            else:
                data = {**context.get("call_metadata", {}), **context.get("analytics", {})}
            payload = {
                "callId": str(call_session.id),
                "agentId": (
                    str(call_session.agent_id) if call_session.agent_id else None
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }

        headers = _render_headers(
            _decrypt_headers_safe(call_flow.post_call_webhook_headers_encrypted, db),
            context,
        )
        query_params = _render_query_params(
            call_flow.post_call_webhook_query_params, context
        )
        if call_flow.disable_metadata:
            query_params = _filter_query_params_metadata(query_params)

        log = await _deliver(
            db,
            tenant_id=call_flow.tenant_id,
            call_flow_id=call_flow.id,
            call_session_id=call_session.id,
            webhook_kind="post_call",
            event_type=None,
            url=call_flow.post_call_webhook_url,
            headers=headers,
            query_params=query_params,
            json_body=payload,
            timeout_seconds=_DISPATCH_TIMEOUT_SECONDS,
        )
        return log.status == "success"
    except Exception as exc:
        logger.warning(
            "Post-call webhook dispatch failed for session=%s: %s", call_session_id, exc
        )
        return False
    finally:
        db.close()


async def run_post_call_webhook(call_session_id: uuid.UUID) -> None:
    """ARQ job body — real implementation invoked by the worker wrapper in
    `app/workers/batch_call_worker.py`."""
    success = await _dispatch_post_call_webhook(call_session_id)
    if not success:
        await _schedule_webhook_retry("post_call", call_session_id, attempt_number=1)


# ── (4) Status Webhook ───────────────────────────────────────────────────────


def schedule_status_webhook(
    call_session_id: uuid.UUID, event_type: str, extra: dict | None = None
) -> None:
    """Enqueue the Status Webhook as an ARQ background job. Fire-and-forget,
    fails open if the ARQ pool isn't ready. Mirrors
    `schedule_post_call_webhook`/`schedule_run_post_call_analysis` exactly.
    """
    from app.utils.arq_pool import get_arq_pool

    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; status webhook (%s) skipped for session=%s",
            event_type,
            call_session_id,
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job(
                "run_status_webhook", str(call_session_id), event_type, extra
            )
        except Exception as exc:
            logger.warning("Failed to enqueue status webhook job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())


def _derive_status_field(event_type: str, extra: dict | None) -> str:
    """Derive the outbound status-webhook payload's `status` field from
    `event_type`/`extra` instead of hardcoding it, so it can't silently go
    stale as more event types are added later.

    As of this writing, every `schedule_status_webhook()` call site
    (`app/routers/voice.py`, `app/voice/tts_stream_mixin.py`,
    `app/services/call_session_service.py`) uses one of `call.connected`,
    `call.ended`, `call.transfer`, or `call.test`. Two of those carry an
    actual outcome signal in `extra["outcome"]`:
    - `call.transfer`: the dial-completion outcome for the transfer attempt
      (`"completed"` == success; anything else — busy/no-answer/failed —
      is not a successful transfer).
    - `call.ended`: the terminal `CallSession.status` the call actually ended
      with (`"completed"` == success; `"failed"`/`"busy"`/`"no_answer"` are
      not). Passed in from
      `CallSessionService.update_call_session_status()`.
    `call.connected` and `call.test` are unconditional lifecycle
    notifications with no failure variant and always report `"success"`.
    """
    if event_type in ("call.transfer", "call.ended") and extra:
        outcome = extra.get("outcome")
        if outcome and outcome != "completed":
            return "failed"
    return "success"


async def _dispatch_status_webhook(
    call_session_id: uuid.UUID, event_type: str, extra: dict | None = None
) -> bool:
    from app.db.session import SessionLocal

    db: Session = SessionLocal()
    try:
        call_session = db.execute(
            select(CallSession).where(CallSession.id == call_session_id)
        ).scalar_one_or_none()
        if not call_session or not call_session.call_flow_id:
            return True

        call_flow = db.execute(
            select(CallFlow).where(
                CallFlow.id == call_session.call_flow_id,
                CallFlow.tenant_id == call_session.tenant_id,
            )
        ).scalar_one_or_none()
        if (
            not call_flow
            or not call_flow.status_webhook_enabled
            or not call_flow.status_webhook_url
        ):
            return True

        payload: dict[str, Any] = {
            "callId": str(call_session.id),
            "apiName": event_type,
            "apiUrl": None,
            "status": _derive_status_field(event_type, extra),
            "statusCode": None,
            "response": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration": call_session.duration,
        }
        if extra:
            payload.update(extra)

        # Render headers/query params from a namespaced context, consistent
        # with the pre-inbound and post-call dispatch paths — NOT from the
        # outgoing payload dict itself. A header template like
        # `{{_metadata.apiKey}}` must resolve against tenant-configured
        # static metadata, not against payload fields like callId/status.
        # NOTE: unlike `pre_inbound_webhook_static_metadata`, `CallFlow` has
        # no `status_webhook_static_metadata` column — adding one is a
        # db-migration-scoped change, out of scope for this fix — so
        # `_metadata`/`_variable` are intentionally omitted here rather than
        # invented from an unrelated field.
        context: dict[str, Any] = {
            "_system": {"callId": str(call_session.id), "eventType": event_type},
        }

        headers = _render_headers(
            _decrypt_headers_safe(call_flow.status_webhook_headers_encrypted, db),
            context,
        )
        query_params = _render_query_params(
            call_flow.status_webhook_query_params, context
        )
        if call_flow.disable_metadata:
            payload = _strip_metadata_from_dict(payload)
            query_params = _filter_query_params_metadata(query_params)

        log = await _deliver(
            db,
            tenant_id=call_flow.tenant_id,
            call_flow_id=call_flow.id,
            call_session_id=call_session.id,
            webhook_kind="status",
            event_type=event_type,
            url=call_flow.status_webhook_url,
            headers=headers,
            query_params=query_params,
            json_body=payload,
            timeout_seconds=_DISPATCH_TIMEOUT_SECONDS,
        )
        return log.status == "success"
    except Exception as exc:
        logger.warning(
            "Status webhook (%s) dispatch failed for session=%s: %s",
            event_type,
            call_session_id,
            exc,
        )
        return False
    finally:
        db.close()


async def run_status_webhook(
    call_session_id: uuid.UUID, event_type: str, extra: dict | None = None
) -> None:
    """ARQ job body — real implementation invoked by the worker wrapper in
    `app/workers/batch_call_worker.py`."""
    success = await _dispatch_status_webhook(call_session_id, event_type, extra)
    if not success:
        await _schedule_webhook_retry(
            "status",
            call_session_id,
            attempt_number=1,
            event_type=event_type,
            extra=extra,
        )


# ── Shared bounded retry (post_call + status) ───────────────────────────────


async def _schedule_webhook_retry(
    kind: str,
    call_session_id: uuid.UUID,
    attempt_number: int,
    event_type: str | None = None,
    extra: dict | None = None,
) -> None:
    """Enqueue an ARQ retry job with the appropriate delay. `attempt_number`
    is 1-indexed (1 = first retry after the initial failure); stops silently
    once `_RETRY_DELAYS_MINUTES` is exhausted (2 retries total)."""
    if attempt_number > len(_RETRY_DELAYS_MINUTES):
        return

    delay_minutes = _RETRY_DELAYS_MINUTES[attempt_number - 1]
    defer_until = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)

    try:
        from app.utils.arq_pool import get_arq_pool

        pool = get_arq_pool()
        _owns_pool = False

        if pool is None:
            import arq

            from app.core.config import settings as cfg

            pool = await arq.create_pool(
                arq.connections.RedisSettings.from_dsn(cfg.REDIS_URL)
            )
            _owns_pool = True

        try:
            await pool.enqueue_job(
                "retry_system_webhook_delivery",
                kind,
                str(call_session_id),
                attempt_number,
                event_type,
                extra,
                _defer_until=defer_until,
            )
            logger.info(
                "system webhook retry enqueued: kind=%s session=%s attempt=%s defer=%s",
                kind,
                call_session_id,
                attempt_number,
                defer_until.isoformat(),
            )
        finally:
            if _owns_pool:
                await pool.aclose()

    except Exception as exc:
        logger.warning(
            "Failed to enqueue system webhook retry (kind=%s session=%s attempt=%s): %s "
            "— delivery will not be retried automatically",
            kind,
            call_session_id,
            attempt_number,
            exc,
        )


async def retry_system_webhook_delivery(
    kind: str,
    call_session_id: uuid.UUID,
    attempt_number: int,
    event_type: str | None = None,
    extra: dict | None = None,
) -> None:
    """ARQ job body — re-attempts a failed post_call or status webhook
    delivery by re-running its dispatch function from current DB state
    (there is no single mutable delivery row to retry in place, unlike
    `webhook_service.retry_webhook_delivery`; each attempt is its own
    `SystemWebhookDeliveryLog` row)."""
    if kind == "post_call":
        success = await _dispatch_post_call_webhook(call_session_id)
    elif kind == "status":
        success = await _dispatch_status_webhook(
            call_session_id, event_type or "", extra
        )
    else:
        logger.warning("retry_system_webhook_delivery: unknown kind=%r", kind)
        return

    if not success and attempt_number < len(_RETRY_DELAYS_MINUTES):
        await _schedule_webhook_retry(
            kind,
            call_session_id,
            attempt_number + 1,
            event_type=event_type,
            extra=extra,
        )


# ── Test delivery (POST /{flow_id}/system-webhooks/test) ───────────────────


async def run_webhook_test(
    db: Session,
    flow_id: uuid.UUID,
    tenant_id: uuid.UUID,
    webhook_kind: str,
) -> SystemWebhookDeliveryLog:
    """Synchronous one-shot delivery attempt against whatever is CURRENTLY
    SAVED for `webhook_kind` on this call flow — no "test unsaved draft"
    mechanism (out of scope). Reuses `_deliver`, the same helper the real
    dispatch paths use.

    Test payloads are synthetic (no real call backs them) — documented
    judgment call since the plan didn't specify test-payload contents beyond
    "test whatever is currently saved."
    """
    call_flow = (
        db.query(CallFlow)
        .filter(
            CallFlow.id == flow_id,
            CallFlow.tenant_id == tenant_id,
            CallFlow.is_deleted.is_(False),
        )
        .first()
    )
    if call_flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call flow {flow_id} not found",
        )

    if webhook_kind == "pre_inbound":
        if not call_flow.pre_inbound_webhook_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pre-Inbound Call Webhook is not configured for this call flow",
            )
        static_metadata = call_flow.pre_inbound_webhook_static_metadata or {}
        context: dict[str, Any] = {
            "_system": {"phoneNumber": "+15555550100", "fromNumber": "+15555550101"},
            "_metadata": static_metadata,
            "_variable": static_metadata,
        }
        url = render_template(call_flow.pre_inbound_webhook_url, context)
        headers = _render_headers(
            _decrypt_headers_safe(call_flow.pre_inbound_webhook_headers_encrypted, db),
            context,
        )
        query_params = _render_query_params(
            call_flow.pre_inbound_webhook_query_params, context
        )
        test_pre_inbound_json = {
            "from": context["_system"]["fromNumber"],
            "to": context["_system"]["phoneNumber"],
        }
        if not call_flow.disable_metadata:
            test_pre_inbound_json["metadata"] = static_metadata
        if call_flow.disable_metadata:
            query_params = _filter_query_params_metadata(query_params)

        return await _deliver(
            db,
            tenant_id=call_flow.tenant_id,
            call_flow_id=call_flow.id,
            call_session_id=None,
            webhook_kind="pre_inbound",
            event_type=None,
            url=url,
            headers=headers,
            query_params=query_params,
            json_body=test_pre_inbound_json,
            timeout_seconds=_PRE_INBOUND_TIMEOUT_SECONDS,
        )

    if webhook_kind == "post_call":
        if not call_flow.post_call_webhook_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Post-Call Webhook is not configured for this call flow",
            )
        payload: dict[str, Any] = {
            "callId": "test-call-id",
            "agentId": str(call_flow.agent_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "test": True,
                "note": "Synthetic test delivery — no real call data",
            },
        }
        if (
            call_flow.post_call_webhook_custom_payload_enabled
            and call_flow.post_call_webhook_custom_payload_template
        ):
            payload = render_json_template(
                call_flow.post_call_webhook_custom_payload_template, payload
            )

        # Headers/query-params must render against the same field-catalog
        # context the real dispatch path uses (`build_post_call_payload_context`
        # in `_dispatch_post_call_webhook`), NOT the synthetic `payload` dict
        # above — otherwise a header/query template referencing
        # `{{header_variables.x}}` or `{{conversation_data.x}}` looks broken
        # when tested even though real dispatch resolves it fine. This
        # endpoint only has `flow_id`/`tenant_id` (no `call_session_id`), so
        # we can't scope to a call that actually belongs to this flow; the
        # tenant's most recent call session (any call flow) is close enough
        # to exercise real field values. If the tenant has no calls yet at
        # all, fall back to an all-empty-namespaces context (rather than the
        # synthetic `payload` dict) so field-catalog templates still render
        # (as empty string) with the same *shape* real dispatch would use,
        # instead of looking structurally different.
        recent_call_session = (
            db.execute(
                select(CallSession)
                .where(CallSession.tenant_id == tenant_id)
                .order_by(CallSession.created_at.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        if recent_call_session is not None:
            recent_call_log = (
                db.execute(
                    select(CallLog)
                    .where(
                        CallLog.call_session_id == recent_call_session.id,
                        CallLog.tenant_id == tenant_id,
                    )
                    .limit(1)
                )
                .scalars()
                .first()
            )
            header_query_context = build_post_call_payload_context(
                db, recent_call_session, recent_call_log
            )
        else:
            header_query_context = {
                "call_metadata": {},
                "conversation_data": {},
                "transcript": {},
                "analytics": {},
                "header_variables": {},
            }

        headers = _render_headers(
            _decrypt_headers_safe(call_flow.post_call_webhook_headers_encrypted, db),
            header_query_context,
        )
        query_params = _render_query_params(
            call_flow.post_call_webhook_query_params, header_query_context
        )
        if call_flow.disable_metadata:
            payload = _strip_metadata_from_dict(payload)
            query_params = _filter_query_params_metadata(query_params)

        return await _deliver(
            db,
            tenant_id=call_flow.tenant_id,
            call_flow_id=call_flow.id,
            call_session_id=None,
            webhook_kind="post_call",
            event_type=None,
            url=call_flow.post_call_webhook_url,
            headers=headers,
            query_params=query_params,
            json_body=payload,
            timeout_seconds=_DISPATCH_TIMEOUT_SECONDS,
        )

    if webhook_kind == "status":
        if not call_flow.status_webhook_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status Webhook is not configured for this call flow",
            )
        payload = {
            "callId": "test-call-id",
            "apiName": "call.test",
            "apiUrl": None,
            "status": "success",
            "statusCode": None,
            "response": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration": None,
        }
        # Same namespaced-context rationale as `_dispatch_status_webhook` —
        # header/query templates render against `_system`, not the outgoing
        # payload dict.
        test_context: dict[str, Any] = {
            "_system": {"callId": "test-call-id", "eventType": "call.test"},
        }
        headers = _render_headers(
            _decrypt_headers_safe(call_flow.status_webhook_headers_encrypted, db),
            test_context,
        )
        query_params = _render_query_params(
            call_flow.status_webhook_query_params, test_context
        )
        if call_flow.disable_metadata:
            payload = _strip_metadata_from_dict(payload)
            query_params = _filter_query_params_metadata(query_params)

        return await _deliver(
            db,
            tenant_id=call_flow.tenant_id,
            call_flow_id=call_flow.id,
            call_session_id=None,
            webhook_kind="status",
            event_type="call.test",
            url=call_flow.status_webhook_url,
            headers=headers,
            query_params=query_params,
            json_body=payload,
            timeout_seconds=_DISPATCH_TIMEOUT_SECONDS,
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown webhook_kind: {webhook_kind!r}",
    )
