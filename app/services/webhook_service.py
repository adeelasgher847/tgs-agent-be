"""
WebhookService — CRUD, signing, delivery, and retry scheduling.

Secret storage: pgp_sym_encrypt via encrypt_webhook_secret / decrypt_stored_webhook_secret.
  Legacy JWT-encrypted secrets (written before v20260612) are transparently
  supported on read; new secrets are always written as pgcrypto.

HMAC signing: HMAC-SHA256 over the JSON payload string.

Retry policy: up to 3 retries with 1-min, 5-min, 30-min gaps (ARQ-backed).

SSRF protection: assert_public_url() is called inside _attempt_delivery() on
  every outbound HTTP request. Creation-time validation alone is insufficient
  because DNS records can change after registration (TOCTOU risk).

PII note: webhook payloads for call events intentionally include phone numbers
  (from_number, to_number) as they are the primary identifiers for call events.
  The SSRF guard in assert_public_url() is the primary control preventing
  delivery to internal/private destinations.

Concurrent delivery: fire_webhooks() delivers to each active endpoint in an
  isolated coroutine (its own DB session) via asyncio.gather so that one slow
  endpoint cannot delay all others. Per-endpoint isolation avoids shared-session
  SQLAlchemy races.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.db_encryption import decrypt_stored_webhook_secret, encrypt_webhook_secret
from app.core.logger import logger
from app.models.webhook import WebhookDelivery, WebhookEndpoint
from app.schemas.webhook import (
    PaginatedWebhookDeliveries,
    WebhookDeliveryOut,
    WebhookEndpointOut,
)
from app.utils.ssrf import SSRFBlockedError, assert_public_url

_DELIVERY_TIMEOUT_SECONDS = 5
_MAX_RESPONSE_BODY_CHARS = 500
_MAX_ATTEMPTS = 4  # 1 initial + 3 retries
_RETRY_DELAYS_MINUTES = [1, 5, 30]


class WebhookService:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Endpoint CRUD ─────────────────────────────────────────────────────────

    def create_endpoint(
        self, workspace_id: uuid.UUID, url: str, raw_secret: str
    ) -> WebhookEndpoint:
        encrypted = encrypt_webhook_secret(raw_secret, self._db)
        endpoint = WebhookEndpoint(
            workspace_id=workspace_id,
            url=str(url),
            secret=encrypted,
        )
        self._db.add(endpoint)
        self._db.commit()
        self._db.refresh(endpoint)
        return endpoint

    def list_endpoints(self, workspace_id: uuid.UUID) -> list[WebhookEndpoint]:
        return (
            self._db.query(WebhookEndpoint)
            .filter(WebhookEndpoint.workspace_id == workspace_id)
            .order_by(WebhookEndpoint.created_at.desc())
            .all()
        )

    def delete_endpoint(
        self, workspace_id: uuid.UUID, endpoint_id: uuid.UUID
    ) -> None:
        endpoint = self._get_endpoint_or_404(workspace_id, endpoint_id)
        self._db.delete(endpoint)
        self._db.commit()

    def list_deliveries(
        self,
        workspace_id: uuid.UUID,
        endpoint_id: uuid.UUID,
        page: int,
        page_size: int,
    ) -> PaginatedWebhookDeliveries:
        self._get_endpoint_or_404(workspace_id, endpoint_id)

        base_q = self._db.query(WebhookDelivery).filter(
            WebhookDelivery.endpoint_id == endpoint_id
        )
        total = base_q.count()
        items = (
            base_q.order_by(WebhookDelivery.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return PaginatedWebhookDeliveries(
            items=[WebhookDeliveryOut.model_validate(d) for d in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    # ── Test ping ─────────────────────────────────────────────────────────────

    async def send_test_ping(
        self, workspace_id: uuid.UUID, endpoint_id: uuid.UUID
    ) -> WebhookDelivery:
        endpoint = self._get_endpoint_or_404(workspace_id, endpoint_id)
        raw_secret = self._decrypt_secret(endpoint)

        payload: dict = {
            "event": "ping",
            "workspace_id": str(workspace_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "Test ping from TGS Voice Agent Platform"},
        }
        delivery = await self._attempt_delivery(
            endpoint=endpoint,
            raw_secret=raw_secret,
            event_type="ping",
            payload=payload,
            existing_delivery=None,
        )
        return delivery

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_endpoint_or_404(
        self, workspace_id: uuid.UUID, endpoint_id: uuid.UUID
    ) -> WebhookEndpoint:
        endpoint = (
            self._db.query(WebhookEndpoint)
            .filter(
                WebhookEndpoint.id == endpoint_id,
                WebhookEndpoint.workspace_id == workspace_id,
            )
            .first()
        )
        if endpoint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Webhook endpoint not found",
            )
        return endpoint

    def _decrypt_secret(self, endpoint: WebhookEndpoint) -> str:
        return decrypt_stored_webhook_secret(endpoint.secret, db=self._db)

    @staticmethod
    def sign_payload(raw_secret: str, payload_json: str) -> str:
        return hmac.new(
            raw_secret.encode(),
            payload_json.encode(),
            hashlib.sha256,
        ).hexdigest()

    async def _attempt_delivery(
        self,
        endpoint: WebhookEndpoint,
        raw_secret: str,
        event_type: str,
        payload: dict,
        existing_delivery: Optional[WebhookDelivery],
    ) -> WebhookDelivery:
        payload_json = json.dumps(payload, default=str)
        signature = self.sign_payload(raw_secret, payload_json)

        http_status: Optional[int] = None
        response_body: Optional[str] = None
        delivered = False

        # SSRF guard — re-validate on every delivery attempt.
        # Creation-time validation is insufficient due to TOCTOU: DNS records
        # can change after an endpoint is registered.
        try:
            assert_public_url(endpoint.url)
        except SSRFBlockedError as exc:
            response_body = f"SSRF blocked: {exc}"[:_MAX_RESPONSE_BODY_CHARS]
            logger.warning(
                "webhook: SSRF guard blocked delivery to endpoint %s (%s): %s",
                endpoint.id,
                endpoint.url,
                exc,
            )
        else:
            try:
                async with httpx.AsyncClient(timeout=_DELIVERY_TIMEOUT_SECONDS) as client:
                    resp = await client.post(
                        endpoint.url,
                        content=payload_json,
                        headers={
                            "Content-Type": "application/json",
                            "X-Webhook-Signature": signature,
                        },
                    )
                http_status = resp.status_code
                response_body = resp.text[:_MAX_RESPONSE_BODY_CHARS]
                delivered = 200 <= resp.status_code < 300
            except httpx.TimeoutException:
                response_body = "timeout"
            except Exception as exc:
                response_body = str(exc)[:_MAX_RESPONSE_BODY_CHARS]

        now = datetime.now(timezone.utc)

        if existing_delivery is None:
            delivery = WebhookDelivery(
                endpoint_id=endpoint.id,
                event_type=event_type,
                payload=payload,
                status="delivered" if delivered else "failed",
                http_status=http_status,
                response_body=response_body,
                attempt_count=1,
                last_attempted_at=now,
            )
            self._db.add(delivery)
        else:
            existing_delivery.attempt_count += 1
            existing_delivery.last_attempted_at = now
            existing_delivery.http_status = http_status
            existing_delivery.response_body = response_body
            existing_delivery.status = "delivered" if delivered else "failed"
            delivery = existing_delivery

        self._db.commit()
        self._db.refresh(delivery)
        return delivery


# ── Per-endpoint isolated delivery (used by fire_webhooks) ───────────────────

async def _deliver_to_endpoint(
    endpoint_id: uuid.UUID,
    event_type: str,
    payload: dict,
) -> None:
    """Deliver *payload* to a single endpoint using its own DB session.

    Isolation via a dedicated session prevents SQLAlchemy shared-session races
    when multiple endpoints are delivered concurrently via asyncio.gather.
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        endpoint = db.get(WebhookEndpoint, endpoint_id)
        if endpoint is None or not endpoint.is_active:
            # Endpoint was deactivated between the list query and delivery.
            return

        try:
            raw_secret = decrypt_stored_webhook_secret(endpoint.secret, db=db)
        except Exception as exc:
            logger.warning(
                "webhook: could not decrypt secret for endpoint %s: %s",
                endpoint_id,
                exc,
            )
            return

        svc = WebhookService(db)
        delivery = await svc._attempt_delivery(
            endpoint=endpoint,
            raw_secret=raw_secret,
            event_type=event_type,
            payload=payload,
            existing_delivery=None,
        )

        if delivery.status != "delivered":
            await _schedule_retry(delivery.id, attempt_number=1)

    except Exception as exc:
        logger.error(
            "_deliver_to_endpoint error (endpoint=%s event=%s): %s",
            endpoint_id,
            event_type,
            exc,
        )
    finally:
        db.close()


# ── Background delivery (called from route background tasks) ──────────────────

async def fire_webhooks(
    workspace_id: uuid.UUID,
    event_type: str,
    data: dict,
) -> None:
    """Deliver a webhook event to all active endpoints for the workspace.

    Opens its own DB session to list active endpoints, then dispatches each
    delivery as an isolated coroutine (via asyncio.gather) so that one slow or
    failing endpoint cannot delay the others.

    Phone numbers (from_number / to_number) may appear in *data* for call
    events. They are intentionally included — the SSRF guard inside
    _attempt_delivery() is the primary control preventing delivery to internal
    or metadata-service destinations.
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        endpoints = (
            db.query(WebhookEndpoint)
            .filter(
                WebhookEndpoint.workspace_id == workspace_id,
                WebhookEndpoint.is_active.is_(True),
            )
            .all()
        )
        endpoint_ids = [e.id for e in endpoints]
    except Exception as exc:
        logger.error(
            "fire_webhooks: failed to query endpoints (workspace=%s event=%s): %s",
            workspace_id,
            event_type,
            exc,
        )
        return
    finally:
        db.close()

    if not endpoint_ids:
        return

    payload: dict = {
        "event": event_type,
        "workspace_id": str(workspace_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    results = await asyncio.gather(
        *[_deliver_to_endpoint(eid, event_type, payload) for eid in endpoint_ids],
        return_exceptions=True,
    )

    for eid, result in zip(endpoint_ids, results):
        if isinstance(result, Exception):
            logger.error(
                "fire_webhooks: unhandled error for endpoint %s (event=%s): %s",
                eid,
                event_type,
                result,
            )


async def retry_webhook_delivery(
    delivery_id: uuid.UUID,
    attempt_number: int,
) -> None:
    """Re-attempt a failed webhook delivery. Called by the ARQ retry job."""
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        delivery = db.get(WebhookDelivery, delivery_id)
        if delivery is None or delivery.status == "delivered":
            return

        endpoint = db.get(WebhookEndpoint, delivery.endpoint_id)
        if endpoint is None or not endpoint.is_active:
            delivery.status = "failed"
            db.commit()
            return

        try:
            raw_secret = decrypt_stored_webhook_secret(endpoint.secret, db=db)
        except Exception as exc:
            logger.warning(
                "webhook retry: could not decrypt secret for endpoint %s: %s",
                endpoint.id,
                exc,
            )
            delivery.status = "failed"
            db.commit()
            return

        delivery.status = "retrying"
        db.commit()

        svc = WebhookService(db)
        delivery = await svc._attempt_delivery(
            endpoint=endpoint,
            raw_secret=raw_secret,
            event_type=delivery.event_type,
            payload=delivery.payload,
            existing_delivery=delivery,
        )

        if delivery.status != "delivered" and attempt_number < _MAX_ATTEMPTS - 1:
            await _schedule_retry(delivery.id, attempt_number=attempt_number + 1)
    except Exception as exc:
        logger.error(
            "retry_webhook_delivery error (delivery=%s): %s", delivery_id, exc
        )
    finally:
        db.close()


async def _schedule_retry(delivery_id: uuid.UUID, attempt_number: int) -> None:
    """Enqueue an ARQ retry job with the appropriate delay."""
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
                "retry_webhook_delivery",
                str(delivery_id),
                attempt_number,
                _defer_until=defer_until,
            )
            logger.info(
                "webhook retry enqueued: delivery=%s attempt=%s defer=%s",
                delivery_id,
                attempt_number,
                defer_until.isoformat(),
            )
        finally:
            if _owns_pool:
                await pool.aclose()

    except Exception as exc:
        logger.warning(
            "Failed to enqueue webhook retry (delivery=%s attempt=%s): %s — "
            "delivery will not be retried automatically",
            delivery_id,
            attempt_number,
            exc,
        )
