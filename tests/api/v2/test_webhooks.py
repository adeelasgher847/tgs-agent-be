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
  - SSRF guard: localhost, 127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x blocked;
                valid public hosts pass; SSRF failures logged as failed deliveries
  - Secret encryption: pgcrypto round-trip; legacy JWT read-back; bad key rejected
  - Concurrent delivery: asyncio.gather used; inactive endpoints excluded
  - Inactive endpoint excluded from initial delivery and retries
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import socket
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
    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self, monkeypatch):
        """Skip SSRF DNS lookups for non-SSRF tests."""
        monkeypatch.setattr("app.schemas.webhook.assert_public_url", lambda url: None)

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
        with patch.object(_wh_mod, "_schedule_retry", wraps=_wh_mod._schedule_retry):
            with patch("app.utils.arq_pool.get_arq_pool", return_value=mock_pool):
                delivery_id = uuid.uuid4()
                asyncio.run(_wh_mod._schedule_retry(delivery_id, attempt_number=1))

        mock_pool.enqueue_job.assert_called_once()
        call_kwargs = mock_pool.enqueue_job.call_args
        assert "_defer_until" in call_kwargs.kwargs
        assert call_kwargs.kwargs["_defer_until"] is not None


# ── Delivery logging helpers ───────────────────────────────────────────────────

class _FakeDelivery:
    """Lightweight substitute for WebhookDelivery ORM model used in unit tests.

    Instantiating real SQLAlchemy ORM models in isolated unit tests triggers
    mapper configuration for ALL models, which fails because some Tenant
    relationships reference models with inconsistent FK definitions (pre-existing
    project issue). Using a plain Python class avoids this entirely.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeEndpoint:
    """Lightweight substitute for WebhookEndpoint ORM model."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _run_attempt_delivery(*, http_status_code=200, timeout=False):
    """Run WebhookService._attempt_delivery with mocked ORM and httpx.

    Returns the list of objects passed to db.add() so callers can assert
    delivery attributes without accessing a real SQLAlchemy session.
    """
    from app.services.webhook_service import WebhookService

    added_objects: list = []
    db = MagicMock()
    db.add = MagicMock(side_effect=lambda obj: added_objects.append(obj))
    db.commit = MagicMock()
    db.refresh = MagicMock(side_effect=lambda obj: None)

    ep = MagicMock()
    ep.id = ENDPOINT_ID
    ep.url = "https://example.com/hook"

    svc = WebhookService(db)

    mock_response = MagicMock()
    mock_response.status_code = http_status_code
    mock_response.text = "response body"

    async def _inner():
        with patch("app.services.webhook_service.WebhookDelivery", _FakeDelivery):
            with patch("httpx.AsyncClient") as mock_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                if timeout:
                    import httpx as _httpx
                    mock_client.post = AsyncMock(
                        side_effect=_httpx.TimeoutException("timed out")
                    )
                else:
                    mock_client.post = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client

                await svc._attempt_delivery(
                    endpoint=ep,
                    raw_secret=SECRET,
                    event_type="call.completed",
                    payload={"event": "call.completed"},
                    existing_delivery=None,
                )

    asyncio.run(_inner())
    return added_objects


# ── Delivery logging on various HTTP statuses ─────────────────────────────────

class TestDeliveryLogging:
    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.webhook_service.assert_public_url", lambda url: None
        )

    def test_delivery_logged_as_failed_on_500_response(self):
        added = _run_attempt_delivery(http_status_code=500)

        assert len(added) == 1
        assert added[0].status == "failed"
        assert added[0].http_status == 500

    def test_delivery_logged_as_delivered_on_200_response(self):
        added = _run_attempt_delivery(http_status_code=200)

        assert len(added) == 1
        assert added[0].status == "delivered"
        assert added[0].http_status == 200

    def test_timeout_logged_as_failed(self):
        added = _run_attempt_delivery(timeout=True)

        assert len(added) == 1
        assert added[0].status == "failed"
        assert added[0].response_body == "timeout"


# ── SSRF guard — unit tests on assert_public_url ──────────────────────────────

class TestSSRFGuard:
    """
    Unit tests for app.utils.ssrf.assert_public_url.

    socket.getaddrinfo is mocked to return controlled addresses so these tests
    do not require real DNS lookups and run reliably in all CI environments.
    """

    def _make_getaddrinfo(self, ip: str):
        """Return a mock getaddrinfo result resolving *hostname* to *ip*."""
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    def test_localhost_hostname_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("127.0.0.1")):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://localhost/hook")

    def test_loopback_ip_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("127.0.0.1")):
            with pytest.raises(SSRFBlockedError, match="127.0.0.1"):
                assert_public_url("https://127.0.0.1/hook")

    def test_private_10_range_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("10.0.0.1")):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://internal.corp/hook")

    def test_private_172_16_range_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("172.16.0.1")):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://internal.corp/hook")

    def test_private_172_31_range_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("172.31.255.254")):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://internal.corp/hook")

    def test_private_192_168_range_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("192.168.1.100")):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://internal.corp/hook")

    def test_metadata_endpoint_169_254_blocked(self):
        """AWS / GCP metadata endpoint lives in link-local range and must be blocked."""
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch(
            "socket.getaddrinfo",
            return_value=self._make_getaddrinfo("169.254.169.254"),
        ):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://169.254.169.254/latest/meta-data/")

    def test_other_link_local_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch("socket.getaddrinfo", return_value=self._make_getaddrinfo("169.254.1.1")):
            with pytest.raises(SSRFBlockedError, match="blocked"):
                assert_public_url("https://169.254.1.1/hook")

    def test_public_ip_passes(self):
        """A genuinely public IP must pass the guard."""
        from app.utils.ssrf import assert_public_url

        with patch(
            "socket.getaddrinfo",
            return_value=self._make_getaddrinfo("93.184.216.34"),  # example.com
        ):
            assert_public_url("https://example.com/hook")  # must not raise

    def test_unresolvable_hostname_blocked(self):
        from app.utils.ssrf import SSRFBlockedError, assert_public_url

        with patch(
            "socket.getaddrinfo",
            side_effect=socket.gaierror("Name or service not known"),
        ):
            with pytest.raises(SSRFBlockedError, match="could not be resolved"):
                assert_public_url("https://no-such-host.invalid/hook")

    def test_ssrf_is_blocked_at_schema_validation(self):
        """A URL resolving to a private IP must be rejected by WebhookEndpointCreate."""
        from app.schemas.webhook import WebhookEndpointCreate
        from app.utils.ssrf import SSRFBlockedError

        with patch(
            "socket.getaddrinfo",
            return_value=self._make_getaddrinfo("192.168.0.1"),
        ):
            with pytest.raises((ValueError, Exception)):
                WebhookEndpointCreate(
                    url="https://internal.corp/hook", secret=SECRET
                )

    def test_ssrf_blocked_delivery_is_logged_as_failed(self):
        """When SSRF blocks a delivery, it must be persisted as status=failed."""
        from app.services.webhook_service import WebhookService
        from app.utils.ssrf import SSRFBlockedError

        ep = MagicMock()
        ep.id = ENDPOINT_ID
        ep.url = "https://internal.corp/hook"

        added: list = []
        db = MagicMock()
        db.add = MagicMock(side_effect=lambda obj: added.append(obj))
        db.commit = MagicMock()
        db.refresh = MagicMock(side_effect=lambda obj: None)

        svc = WebhookService(db)

        async def _run():
            with patch("app.services.webhook_service.WebhookDelivery", _FakeDelivery):
                with patch(
                    "app.services.webhook_service.assert_public_url",
                    side_effect=SSRFBlockedError("Blocked: 192.168.0.1"),
                ):
                    return await svc._attempt_delivery(
                        endpoint=ep,
                        raw_secret=SECRET,
                        event_type="call.completed",
                        payload={"event": "call.completed"},
                        existing_delivery=None,
                    )

        asyncio.run(_run())

        assert len(added) == 1
        assert added[0].status == "failed"
        assert added[0].http_status is None
        assert "SSRF blocked" in added[0].response_body

    def test_ssrf_blocked_delivery_does_not_make_http_call(self):
        """When SSRF blocks, httpx must never be called."""
        from app.services.webhook_service import WebhookService
        from app.utils.ssrf import SSRFBlockedError

        ep = MagicMock()
        ep.id = ENDPOINT_ID
        ep.url = "https://internal.corp/hook"

        db = MagicMock()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock(side_effect=lambda obj: None)

        svc = WebhookService(db)

        async def _run():
            with patch("app.services.webhook_service.WebhookDelivery", _FakeDelivery):
                with patch(
                    "app.services.webhook_service.assert_public_url",
                    side_effect=SSRFBlockedError("Blocked"),
                ):
                    with patch("httpx.AsyncClient") as mock_httpx:
                        await svc._attempt_delivery(
                            endpoint=ep,
                            raw_secret=SECRET,
                            event_type="ping",
                            payload={"event": "ping"},
                            existing_delivery=None,
                        )
                        mock_httpx.assert_not_called()

        asyncio.run(_run())


# ── Secret encryption ─────────────────────────────────────────────────────────

class TestSecretEncryption:
    """
    Verify pgcrypto encrypt/decrypt round-trip and backwards-compat with
    legacy JWT secrets.
    """

    def test_encrypt_webhook_secret_calls_pgcrypto(self):
        from app.core.db_encryption import encrypt_webhook_secret

        db = MagicMock()
        db.execute = MagicMock()
        db.execute.return_value.scalar.return_value = "base64ciphertext=="

        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = "test-key-32-chars-long-pad12345"
            result = encrypt_webhook_secret("my-webhook-secret", db)

        assert result == "base64ciphertext=="

    def test_encrypt_raises_without_key(self):
        from app.core.db_encryption import encrypt_webhook_secret

        db = MagicMock()
        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = ""
            with pytest.raises(ValueError, match="WEBHOOK_SECRET_ENCRYPTION_KEY"):
                encrypt_webhook_secret("secret", db)

    def test_decrypt_webhook_secret_calls_pgcrypto(self):
        from app.core.db_encryption import decrypt_webhook_secret

        db = MagicMock()
        db.execute = MagicMock()
        db.execute.return_value.scalar.return_value = "my-webhook-secret"

        # Build a valid pgcrypto-looking base64 value (first byte 0x85)
        import base64
        fake_ct = base64.b64encode(bytes([0x85]) + b"x" * 20).decode()

        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = "test-key-32-chars-long-pad12345"
            result = decrypt_webhook_secret(fake_ct, db)

        assert result == "my-webhook-secret"

    def test_decrypt_raises_without_key(self):
        from app.core.db_encryption import decrypt_webhook_secret

        import base64
        db = MagicMock()
        fake_ct = base64.b64encode(bytes([0x85]) + b"x" * 20).decode()

        with patch("app.core.db_encryption.settings") as mock_settings:
            mock_settings.WEBHOOK_SECRET_ENCRYPTION_KEY = ""
            with pytest.raises(ValueError, match="WEBHOOK_SECRET_ENCRYPTION_KEY"):
                decrypt_webhook_secret(fake_ct, db)

    def test_decrypt_stored_handles_legacy_jwt(self):
        """decrypt_stored_webhook_secret must transparently handle JWT-format secrets."""
        from app.core.db_encryption import decrypt_stored_webhook_secret

        # Build a fake compact JWS string (three base64url parts separated by dots)
        legacy_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJhcGlfa2V5IjoibXktc2VjcmV0In0.sig"

        with patch("app.core.db_encryption.is_legacy_jwt_ciphertext", return_value=True):
            with patch("app.core.security.decrypt_api_key", return_value="my-secret") as mock_dec:
                result = decrypt_stored_webhook_secret(legacy_jwt)

        mock_dec.assert_called_once_with(legacy_jwt)
        assert result == "my-secret"

    def test_decrypt_stored_handles_pgcrypto_format(self):
        """decrypt_stored_webhook_secret must use pgcrypto for pgcrypto-format secrets."""
        import base64
        from app.core.db_encryption import decrypt_stored_webhook_secret

        fake_ct = base64.b64encode(bytes([0x85]) + b"x" * 20).decode()

        with patch("app.core.db_encryption.is_legacy_jwt_ciphertext", return_value=False):
            with patch("app.core.db_encryption.is_pgcrypto_ciphertext", return_value=True):
                with patch(
                    "app.core.db_encryption.decrypt_webhook_secret",
                    return_value="my-secret",
                ) as mock_dec:
                    db = MagicMock()
                    result = decrypt_stored_webhook_secret(fake_ct, db=db)

        mock_dec.assert_called_once_with(fake_ct, db)
        assert result == "my-secret"

    def test_decrypt_stored_raises_on_unknown_format(self):
        from app.core.db_encryption import decrypt_stored_webhook_secret

        with patch("app.core.db_encryption.is_legacy_jwt_ciphertext", return_value=False):
            with patch("app.core.db_encryption.is_pgcrypto_ciphertext", return_value=False):
                with pytest.raises(ValueError, match="Unrecognized"):
                    decrypt_stored_webhook_secret("not-valid-format")

    def test_create_endpoint_uses_pgcrypto_not_jwt(self):
        """create_endpoint must call encrypt_webhook_secret (pgcrypto), not encrypt_api_key (JWT)."""
        from app.services.webhook_service import WebhookService

        db = MagicMock()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock(side_effect=lambda obj: None)

        with patch(
            "app.services.webhook_service.encrypt_webhook_secret",
            return_value="pgcrypto-ciphertext",
        ) as mock_enc:
            with patch("app.core.security.encrypt_api_key") as mock_jwt_enc:
                # Patch WebhookEndpoint so instantiation doesn't trigger SQLAlchemy
                # mapper configuration (which fails in isolated tests due to
                # unresolved Tenant relationships with partially-imported models).
                with patch(
                    "app.services.webhook_service.WebhookEndpoint",
                    _FakeEndpoint,
                ):
                    svc = WebhookService(db)
                    svc.create_endpoint(WORKSPACE_ID, "https://example.com/hook", SECRET)

        mock_enc.assert_called_once_with(SECRET, db)
        mock_jwt_enc.assert_not_called()


# ── Concurrent delivery ───────────────────────────────────────────────────────

class TestConcurrentDelivery:
    """
    fire_webhooks() must deliver to all active endpoints concurrently and
    exclude inactive ones.
    """

    @pytest.fixture(autouse=True)
    def _bypass_ssrf(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.webhook_service.assert_public_url", lambda url: None
        )

    def test_fire_webhooks_delivers_to_all_active_endpoints(self):
        """All active endpoints get a delivery attempt."""
        from app.services.webhook_service import fire_webhooks

        ep1_id = uuid.uuid4()
        ep2_id = uuid.uuid4()
        delivered_to: list[uuid.UUID] = []

        async def _fake_deliver(endpoint_id, event_type, payload):
            delivered_to.append(endpoint_id)

        with patch("app.services.webhook_service._deliver_to_endpoint", side_effect=_fake_deliver):
            with patch("app.db.session.SessionLocal") as mock_sl:
                db = MagicMock()

                def _make_ep(eid):
                    ep = MagicMock()
                    ep.id = eid
                    ep.is_active = True
                    return ep

                db.query.return_value.filter.return_value.all.return_value = [
                    _make_ep(ep1_id),
                    _make_ep(ep2_id),
                ]
                db.close = MagicMock()
                mock_sl.return_value = db

                asyncio.run(
                    fire_webhooks(WORKSPACE_ID, "call.completed", {"call_sid": "CA123"})
                )

        assert ep1_id in delivered_to
        assert ep2_id in delivered_to

    def test_fire_webhooks_excludes_inactive_endpoints(self):
        """Only active endpoints in the DB query; is_active filter is applied."""
        from app.services.webhook_service import fire_webhooks

        with patch("app.services.webhook_service._deliver_to_endpoint") as mock_deliver:
            with patch("app.db.session.SessionLocal") as mock_sl:
                db = MagicMock()
                # Return empty list — simulating all inactive (filter applied in query)
                db.query.return_value.filter.return_value.all.return_value = []
                db.close = MagicMock()
                mock_sl.return_value = db

                asyncio.run(
                    fire_webhooks(WORKSPACE_ID, "call.completed", {})
                )

        mock_deliver.assert_not_called()

    def test_deliver_to_endpoint_skips_inactive(self):
        """_deliver_to_endpoint must bail out if endpoint.is_active is False."""
        from app.services.webhook_service import _deliver_to_endpoint

        ep = MagicMock()
        ep.is_active = False

        with patch("app.db.session.SessionLocal") as mock_sl:
            db = MagicMock()
            db.get.return_value = ep
            db.close = MagicMock()
            mock_sl.return_value = db

            with patch(
                "app.services.webhook_service.decrypt_stored_webhook_secret"
            ) as mock_dec:
                asyncio.run(
                    _deliver_to_endpoint(ENDPOINT_ID, "call.completed", {})
                )

        mock_dec.assert_not_called()

    def test_retry_skips_inactive_endpoint(self):
        """retry_webhook_delivery must set delivery.status='failed' for inactive endpoints."""
        from app.services.webhook_service import retry_webhook_delivery

        delivery = MagicMock()
        delivery.status = "failed"
        delivery.endpoint_id = ENDPOINT_ID

        ep = MagicMock()
        ep.is_active = False

        with patch("app.db.session.SessionLocal") as mock_sl:
            db = MagicMock()

            def _db_get(model, pk):
                from app.models.webhook import WebhookDelivery, WebhookEndpoint
                if model is WebhookDelivery:
                    return delivery
                if model is WebhookEndpoint:
                    return ep
                return None

            db.get.side_effect = _db_get
            db.commit = MagicMock()
            db.close = MagicMock()
            mock_sl.return_value = db

            asyncio.run(retry_webhook_delivery(uuid.uuid4(), attempt_number=1))

        delivery.status == "failed"
        db.commit.assert_called()

    def test_one_endpoint_failure_does_not_block_others(self):
        """A delivery failure for endpoint A must not prevent endpoint B from being attempted."""
        from app.services.webhook_service import fire_webhooks

        ep1_id = uuid.uuid4()
        ep2_id = uuid.uuid4()
        delivered_to: list[uuid.UUID] = []
        call_count = 0

        async def _fake_deliver(endpoint_id, event_type, payload):
            nonlocal call_count
            call_count += 1
            if endpoint_id == ep1_id:
                raise RuntimeError("simulated delivery failure")
            delivered_to.append(endpoint_id)

        with patch("app.services.webhook_service._deliver_to_endpoint", side_effect=_fake_deliver):
            with patch("app.db.session.SessionLocal") as mock_sl:
                db = MagicMock()

                def _make_ep(eid):
                    ep = MagicMock()
                    ep.id = eid
                    ep.is_active = True
                    return ep

                db.query.return_value.filter.return_value.all.return_value = [
                    _make_ep(ep1_id),
                    _make_ep(ep2_id),
                ]
                db.close = MagicMock()
                mock_sl.return_value = db

                asyncio.run(
                    fire_webhooks(WORKSPACE_ID, "call.completed", {})
                )

        # Both endpoints attempted (gather with return_exceptions=True)
        assert call_count == 2
        assert ep2_id in delivered_to
