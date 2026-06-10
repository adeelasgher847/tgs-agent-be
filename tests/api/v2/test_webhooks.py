"""
API tests for v2 webhooks endpoints.

DB: SQLite in-memory (via existing test conftest pattern).
External HTTP calls: mocked with unittest.mock / respx.

Coverage:
  - POST /webhooks — happy path, HTTPS validation, secret min-length
  - GET /webhooks — list (secret never returned)
  - DELETE /webhooks/{id} — success, 404
  - POST /webhooks/{id}/test — success, delivery logged
  - GET /webhooks/{id}/deliveries — paginated log
  - Auth: readonly role blocked on write endpoints
  - HMAC calculation correctness
  - Retry intervals (1min, 5min, 30min) scheduled correctly
  - Delivery log written on 500 response
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.deps import get_workspace, require_tenant
from app.core.exception_handlers import register_exception_handlers
from app.core.workspace import Workspace


# ── Auth helpers ──────────────────────────────────────────────────────────────

WORKSPACE_ID = uuid.uuid4()
ENDPOINT_ID = uuid.uuid4()
RAW_KEY = f"tgs-{uuid.uuid4().hex}"
KEY_HASH = hashlib.sha256(RAW_KEY.encode()).hexdigest()
AUTH_HEADERS = {"x-api-key": RAW_KEY, "x-workspace-id": str(WORKSPACE_ID)}
SECRET = "super-secret-webhook-key-16chars"


def _mock_workspace() -> Workspace:
    ws = MagicMock(spec=Workspace)
    ws.id = WORKSPACE_ID
    ws.status = "active"
    ws.is_active = True
    return ws


def _mock_principal():
    from app.core.request_auth import ApiKeyPrincipal
    return ApiKeyPrincipal(current_tenant_id=WORKSPACE_ID, api_key_id=uuid.uuid4())


def _mock_endpoint(**overrides) -> MagicMock:
    ep = MagicMock()
    ep.id = overrides.get("id", ENDPOINT_ID)
    ep.workspace_id = WORKSPACE_ID
    ep.url = overrides.get("url", "https://example.com/hook")
    ep.is_active = overrides.get("is_active", True)
    ep.created_at = datetime.now(timezone.utc)
    return ep


def _mock_delivery(**overrides) -> MagicMock:
    d = MagicMock()
    d.id = overrides.get("id", uuid.uuid4())
    d.endpoint_id = ENDPOINT_ID
    d.event_type = overrides.get("event_type", "ping")
    d.payload = {"event": "ping", "workspace_id": str(WORKSPACE_ID)}
    d.status = overrides.get("status", "delivered")
    d.http_status = overrides.get("http_status", 200)
    d.response_body = overrides.get("response_body", "ok")
    d.attempt_count = overrides.get("attempt_count", 1)
    d.last_attempted_at = datetime.now(timezone.utc)
    d.created_at = datetime.now(timezone.utc)
    return d


# ── App factory ───────────────────────────────────────────────────────────────

def _build_app(svc_mock) -> TestClient:
    from app.api.v2.routers import webhooks as wh_module

    ws = _mock_workspace()
    principal = _mock_principal()

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(wh_module.router)

    mini.dependency_overrides[require_tenant] = lambda: principal
    mini.dependency_overrides[get_workspace] = lambda: ws

    def _svc_override():
        yield svc_mock

    mini.dependency_overrides[wh_module._webhook_service] = _svc_override
    mini.dependency_overrides[wh_module._webhook_service_write] = _svc_override

    return TestClient(mini, raise_server_exceptions=False)


# ── POST /webhooks ────────────────────────────────────────────────────────────

class TestCreateWebhookEndpoint:
    def test_creates_endpoint_with_https_url(self):
        ep = _mock_endpoint()
        svc = MagicMock()
        svc.create_endpoint.return_value = ep
        client = _build_app(svc)

        resp = client.post(
            "/webhooks",
            json={"url": "https://example.com/hook", "secret": SECRET},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == str(ep.id)
        assert body["url"] == ep.url
        assert "secret" not in body, "Secret must never be returned"

    def test_rejects_http_url_with_400(self):
        """This project's validation handler returns 400 for RequestValidationError."""
        svc = MagicMock()
        client = _build_app(svc)

        resp = client.post(
            "/webhooks",
            json={"url": "http://insecure.example.com/hook", "secret": SECRET},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 400
        svc.create_endpoint.assert_not_called()

    def test_rejects_secret_under_16_chars_with_400(self):
        """This project's validation handler returns 400 for RequestValidationError."""
        svc = MagicMock()
        client = _build_app(svc)

        resp = client.post(
            "/webhooks",
            json={"url": "https://example.com/hook", "secret": "tooshort"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 400
        svc.create_endpoint.assert_not_called()

    def test_missing_auth_returns_401(self):
        from app.api.v2.routers import webhooks as wh_module

        mini = FastAPI()
        register_exception_handlers(mini)
        mini.include_router(wh_module.router)

        def _unauth():
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized", "message": "Invalid or missing API key"},
            )

        mini.dependency_overrides[require_tenant] = _unauth

        with TestClient(mini, raise_server_exceptions=False) as c:
            resp = c.post(
                "/webhooks",
                json={"url": "https://example.com/hook", "secret": SECRET},
            )

        assert resp.status_code == 401

    def test_readonly_user_blocked_on_post(self):
        from app.api.v2.routers import webhooks as wh_module

        ws = _mock_workspace()
        mini = FastAPI()
        register_exception_handlers(mini)
        mini.include_router(wh_module.router)

        mini.dependency_overrides[get_workspace] = lambda: ws

        def _readonly_svc_write():
            raise HTTPException(
                status_code=403,
                detail="Read-only access cannot modify resources",
            )
            yield  # pragma: no cover — makes this a generator for FastAPI

        mini.dependency_overrides[wh_module._webhook_service_write] = _readonly_svc_write

        with TestClient(mini, raise_server_exceptions=False) as c:
            resp = c.post(
                "/webhooks",
                json={"url": "https://example.com/hook", "secret": SECRET},
                headers=AUTH_HEADERS,
            )

        assert resp.status_code == 403


# ── GET /webhooks ─────────────────────────────────────────────────────────────

class TestListWebhookEndpoints:
    def test_returns_list_without_secret(self):
        endpoints = [_mock_endpoint(), _mock_endpoint(id=uuid.uuid4())]
        svc = MagicMock()
        svc.list_endpoints.return_value = endpoints
        client = _build_app(svc)

        resp = client.get("/webhooks", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        for item in items:
            assert "secret" not in item
            assert "id" in item
            assert "url" in item
            assert "is_active" in item

    def test_empty_list_returns_200(self):
        svc = MagicMock()
        svc.list_endpoints.return_value = []
        client = _build_app(svc)

        resp = client.get("/webhooks", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        assert resp.json() == []


# ── DELETE /webhooks/{id} ─────────────────────────────────────────────────────

class TestDeleteWebhookEndpoint:
    def test_delete_success_returns_204(self):
        svc = MagicMock()
        svc.delete_endpoint.return_value = None
        client = _build_app(svc)

        resp = client.delete(f"/webhooks/{ENDPOINT_ID}", headers=AUTH_HEADERS)

        assert resp.status_code == 204
        svc.delete_endpoint.assert_called_once_with(WORKSPACE_ID, ENDPOINT_ID)

    def test_delete_not_found_returns_404(self):
        svc = MagicMock()
        svc.delete_endpoint.side_effect = HTTPException(
            status_code=404, detail="Webhook endpoint not found"
        )
        client = _build_app(svc)

        resp = client.delete(f"/webhooks/{uuid.uuid4()}", headers=AUTH_HEADERS)

        assert resp.status_code == 404


# ── POST /webhooks/{id}/test ──────────────────────────────────────────────────

class TestTestWebhookEndpoint:
    def test_test_ping_returns_delivery(self):
        delivery = _mock_delivery(event_type="ping", status="delivered", http_status=200)
        svc = MagicMock()
        svc.send_test_ping = AsyncMock(return_value=delivery)
        client = _build_app(svc)

        resp = client.post(f"/webhooks/{ENDPOINT_ID}/test", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["event_type"] == "ping"
        assert body["status"] == "delivered"
        assert body["http_status"] == 200
        assert "secret" not in body

    def test_test_ping_on_missing_endpoint_returns_404(self):
        svc = MagicMock()
        svc.send_test_ping = AsyncMock(
            side_effect=HTTPException(status_code=404, detail="Webhook endpoint not found")
        )
        client = _build_app(svc)

        resp = client.post(f"/webhooks/{uuid.uuid4()}/test", headers=AUTH_HEADERS)

        assert resp.status_code == 404


# ── GET /webhooks/{id}/deliveries ─────────────────────────────────────────────

class TestListWebhookDeliveries:
    def test_returns_paginated_deliveries(self):
        from app.schemas.webhook import PaginatedWebhookDeliveries, WebhookDeliveryOut

        delivery = _mock_delivery()
        paginated = PaginatedWebhookDeliveries(
            items=[
                WebhookDeliveryOut(
                    id=delivery.id,
                    endpoint_id=delivery.endpoint_id,
                    event_type=delivery.event_type,
                    payload=delivery.payload,
                    status=delivery.status,
                    http_status=delivery.http_status,
                    response_body=delivery.response_body,
                    attempt_count=delivery.attempt_count,
                    last_attempted_at=delivery.last_attempted_at,
                    created_at=delivery.created_at,
                )
            ],
            total=1,
            page=1,
            page_size=20,
        )
        svc = MagicMock()
        svc.list_deliveries.return_value = paginated
        client = _build_app(svc)

        resp = client.get(
            f"/webhooks/{ENDPOINT_ID}/deliveries", headers=AUTH_HEADERS
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["status"] == "delivered"


# ── HMAC calculation ──────────────────────────────────────────────────────────

class TestHmacCalculation:
    def test_sign_payload_produces_correct_hmac(self):
        from app.services.webhook_service import WebhookService

        raw_secret = "my-test-secret-key-16plus"
        payload = '{"event": "call.completed"}'
        expected = hmac.new(
            raw_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        result = WebhookService.sign_payload(raw_secret, payload)

        assert result == expected

    def test_different_secrets_produce_different_signatures(self):
        from app.services.webhook_service import WebhookService

        payload = '{"event": "call.started"}'
        sig1 = WebhookService.sign_payload("secret-one-16-chars", payload)
        sig2 = WebhookService.sign_payload("secret-two-16-chars", payload)

        assert sig1 != sig2

    def test_different_payloads_produce_different_signatures(self):
        from app.services.webhook_service import WebhookService

        secret = "shared-secret-16-c"
        sig1 = WebhookService.sign_payload(secret, '{"event": "call.started"}')
        sig2 = WebhookService.sign_payload(secret, '{"event": "call.completed"}')

        assert sig1 != sig2


# ── Retry interval scheduling ─────────────────────────────────────────────────

class TestRetryIntervals:
    """Verify that retry delays match the 1-min / 5-min / 30-min policy."""

    def test_retry_schedule_uses_correct_delays(self):
        from app.services.webhook_service import _RETRY_DELAYS_MINUTES

        assert _RETRY_DELAYS_MINUTES == [1, 5, 30], (
            "Retry delays must be [1, 5, 30] minutes"
        )

    def test_first_retry_deferred_1_minute(self):
        from app.services.webhook_service import _RETRY_DELAYS_MINUTES

        assert _RETRY_DELAYS_MINUTES[0] == 1

    def test_second_retry_deferred_5_minutes(self):
        from app.services.webhook_service import _RETRY_DELAYS_MINUTES

        assert _RETRY_DELAYS_MINUTES[1] == 5

    def test_third_retry_deferred_30_minutes(self):
        from app.services.webhook_service import _RETRY_DELAYS_MINUTES

        assert _RETRY_DELAYS_MINUTES[2] == 30

    @patch("app.utils.arq_pool.get_arq_pool")
    def test_schedule_retry_calls_arq_with_defer_until(self, mock_get_pool):
        """_schedule_retry must pass _defer_until to ARQ enqueue_job."""
        mock_pool = AsyncMock()
        mock_pool.enqueue_job = AsyncMock()
        mock_get_pool.return_value = mock_pool

        from app.services import webhook_service as _wh_mod
        # Patch the imported reference inside the module
        with patch.object(_wh_mod, "_schedule_retry", wraps=_wh_mod._schedule_retry):
            with patch("app.utils.arq_pool.get_arq_pool", return_value=mock_pool):
                delivery_id = uuid.uuid4()
                asyncio.run(_wh_mod._schedule_retry(delivery_id, attempt_number=1))

        mock_pool.enqueue_job.assert_called_once()
        call_kwargs = mock_pool.enqueue_job.call_args
        assert "_defer_until" in call_kwargs.kwargs
        assert call_kwargs.kwargs["_defer_until"] is not None


# ── Delivery logging on 500 ───────────────────────────────────────────────────

class TestDeliveryLogging:
    def test_delivery_logged_as_failed_on_500_response(self):
        """
        When the target endpoint returns HTTP 500, the WebhookDelivery row
        must be persisted with status='failed' and http_status=500.
        """
        import httpx

        from app.models.webhook import WebhookDelivery, WebhookEndpoint
        from app.services.webhook_service import WebhookService

        endpoint = MagicMock(spec=WebhookEndpoint)
        endpoint.id = ENDPOINT_ID
        endpoint.url = "https://example.com/hook"

        db = MagicMock()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock(side_effect=lambda obj: None)

        svc = WebhookService(db)

        captured_delivery: list = []

        original_add = db.add.side_effect

        def _capture_add(obj):
            if isinstance(obj, WebhookDelivery):
                captured_delivery.append(obj)

        db.add.side_effect = _capture_add

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        async def _run():
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                delivery = await svc._attempt_delivery(
                    endpoint=endpoint,
                    raw_secret=SECRET,
                    event_type="call.completed",
                    payload={"event": "call.completed", "workspace_id": str(WORKSPACE_ID)},
                    existing_delivery=None,
                )
            return delivery

        asyncio.run(_run())

        assert len(captured_delivery) == 1
        logged = captured_delivery[0]
        assert logged.status == "failed"
        assert logged.http_status == 500

    def test_delivery_logged_as_delivered_on_200_response(self):
        from app.models.webhook import WebhookDelivery, WebhookEndpoint
        from app.services.webhook_service import WebhookService

        endpoint = MagicMock(spec=WebhookEndpoint)
        endpoint.id = ENDPOINT_ID
        endpoint.url = "https://example.com/hook"

        db = MagicMock()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock(side_effect=lambda obj: None)

        svc = WebhookService(db)

        captured_delivery: list = []

        def _capture_add(obj):
            if isinstance(obj, WebhookDelivery):
                captured_delivery.append(obj)

        db.add.side_effect = _capture_add

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"

        async def _run():
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                delivery = await svc._attempt_delivery(
                    endpoint=endpoint,
                    raw_secret=SECRET,
                    event_type="call.completed",
                    payload={"event": "call.completed"},
                    existing_delivery=None,
                )
            return delivery

        asyncio.run(_run())

        assert len(captured_delivery) == 1
        assert captured_delivery[0].status == "delivered"
        assert captured_delivery[0].http_status == 200

    def test_timeout_logged_as_failed(self):
        import httpx

        from app.models.webhook import WebhookDelivery, WebhookEndpoint
        from app.services.webhook_service import WebhookService

        endpoint = MagicMock(spec=WebhookEndpoint)
        endpoint.id = ENDPOINT_ID
        endpoint.url = "https://example.com/hook"

        db = MagicMock()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock(side_effect=lambda obj: None)

        svc = WebhookService(db)

        captured_delivery: list = []

        def _capture_add(obj):
            if isinstance(obj, WebhookDelivery):
                captured_delivery.append(obj)

        db.add.side_effect = _capture_add

        async def _run():
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
                mock_client_cls.return_value = mock_client

                delivery = await svc._attempt_delivery(
                    endpoint=endpoint,
                    raw_secret=SECRET,
                    event_type="call.started",
                    payload={"event": "call.started"},
                    existing_delivery=None,
                )
            return delivery

        asyncio.run(_run())

        assert len(captured_delivery) == 1
        logged = captured_delivery[0]
        assert logged.status == "failed"
        assert logged.response_body == "timeout"
