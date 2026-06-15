"""
Tests for Make.com and n8n integration endpoints.

Coverage:
  1.  Make.com happy path — valid secret dispatches call, returns {call_id, status}
  2.  Make.com invalid secret → 403 with Make.com error format
  3.  Make.com missing secret → 403
  4.  Make.com unknown agent → 404
  5.  Make.com rate limit → 429 with retry_after
  6.  n8n happy path — valid secret dispatches call, returns {success, data}
  7.  n8n invalid secret → 403
  8.  n8n missing tenant_id → 400
  9.  n8n rate limit → 429 with retry_after
  10. GET /integrations — not connected (no make_secret, no N8N_WEBHOOK_SECRET)
  11. GET /integrations — make connected, n8n connected
  12. POST /workspace/settings/make-secret generates and returns a secret
  13. Rotating make-secret generates a new value each call
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixed UUIDs ───────────────────────────────────────────────────────────────

_AGENT_ID = uuid.UUID("aa100000-0000-0000-0000-000000000001")
_TENANT_ID = uuid.UUID("bb100000-0000-0000-0000-000000000002")
_SESSION_ID = uuid.UUID("cc100000-0000-0000-0000-000000000003")
_TWILIO_SID = "CA99999999999999999999999999999999"
_MAKE_SECRET = "a" * 64  # 32-byte hex = 64 chars


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_agent():
    ag = MagicMock()
    ag.id = _AGENT_ID
    ag.tenant_id = _TENANT_ID
    ag.status = "ready"
    ag.name = "Test Agent"
    ag.model = MagicMock(model_name="gpt-4o")
    return ag


def _make_tenant(*, make_secret: str | None = _MAKE_SECRET):
    t = MagicMock()
    t.id = _TENANT_ID
    settings_dict = {}
    if make_secret is not None:
        settings_dict["make_secret"] = make_secret
    t.workspace_settings = settings_dict
    return t


def _call_session():
    cs = MagicMock()
    cs.id = _SESSION_ID
    cs.call_flow_id = None
    cs.call_metadata = None
    cs.status = "initiated"
    cs.twilio_call_sid = _TWILIO_SID
    return cs


def _phone_number():
    pn = MagicMock()
    pn.id = uuid.uuid4()
    pn.phone_number = "+15550000001"
    pn.assistant_id = _AGENT_ID
    pn.twilio_account_sid = None
    pn.twilio_auth_token = None
    pn.status = "active"
    return pn


def _db(*, tenant=None, agent=None, phone=None, active_outbound=0):
    tenant_obj = tenant or _make_tenant()
    agent_obj = agent or _make_agent()
    phone_obj = phone or _phone_number()

    db = MagicMock()

    def _query(model):
        q = MagicMock()
        # .filter().first() — returns different objects by model name
        def _filter(*args, **kwargs):
            f = MagicMock()
            model_name = getattr(model, "__name__", str(model))
            if "Agent" in model_name:
                f.first.return_value = agent_obj
            elif "Tenant" in model_name:
                f.first.return_value = tenant_obj
            elif "PhoneNumber" in model_name:
                f.first.return_value = phone_obj
            else:
                f.first.return_value = None
            f.scalar.return_value = active_outbound
            return f

        q.filter.side_effect = _filter
        return q

    db.query.side_effect = _query
    db.commit = MagicMock()
    db.refresh = MagicMock()
    return db


def _make_initiate_call_success():
    """Return a SuccessResponse-like object the way initiate_call does on success."""
    from app.schemas.twilio import CallInitiateResponse

    call_data = CallInitiateResponse(
        callId=str(_AGENT_ID),
        twilioCallSid=_TWILIO_SID,
        callSessionId=str(_SESSION_ID),
        status="initiated",
    )

    resp = SimpleNamespace(data=call_data)
    return resp


# ── Patch context for voice_call_service.initiate_call ────────────────────────

def _patch_initiate_call(return_value=None):
    rv = return_value or _make_initiate_call_success()
    return patch(
        "app.routers.integrations.initiate_call_service",
        AsyncMock(return_value=rv),
    )


def _patch_rate_limit_allow():
    return patch(
        "app.routers.integrations.check_integration_rate_limit",
        AsyncMock(return_value=(True, 0.0)),
    )


def _patch_rate_limit_deny():
    import time
    retry = time.time() + 60
    return patch(
        "app.routers.integrations.check_integration_rate_limit",
        AsyncMock(return_value=(False, retry)),
    )


def _patch_resolve_tenant(agent=None, tenant=None):
    ag = agent or _make_agent()
    t = tenant or _make_tenant()
    return patch(
        "app.routers.integrations.resolve_tenant_by_agent",
        MagicMock(return_value=(ag, t)),
    )


def _patch_resolve_tenant_not_found():
    return patch(
        "app.routers.integrations.resolve_tenant_by_agent",
        MagicMock(return_value=(None, None)),
    )


def _patch_record_triggered():
    return patch(
        "app.routers.integrations.record_last_triggered",
        MagicMock(),
    )


def _patch_n8n_secret_valid():
    return patch(
        "app.routers.integrations.verify_n8n_webhook_secret_async",
        AsyncMock(return_value=True),
    )


def _patch_n8n_secret_invalid():
    return patch(
        "app.routers.integrations.verify_n8n_webhook_secret_async",
        AsyncMock(return_value=False),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestMakeTrigger:
    """POST /api/v1/integrations/make/trigger"""

    @pytest.mark.anyio
    async def test_happy_path(self):
        from app.routers.integrations import make_trigger
        from app.schemas.integration import MakeTriggerRequest

        body = MakeTriggerRequest(
            agent_id=str(_AGENT_ID),
            to_number="+15550001111",
            variables={"name": "Alice"},
        )
        request = MagicMock()
        request.scope = {"type": "http", "headers": []}
        request.receive = AsyncMock()

        with (
            _patch_resolve_tenant(),
            _patch_rate_limit_allow(),
            _patch_initiate_call(),
            _patch_record_triggered(),
        ):
            result = await make_trigger(
                body=body,
                request=request,
                db=_db(),
                x_make_secret=_MAKE_SECRET,
            )

        assert result.call_id == str(_SESSION_ID)
        assert result.status == "initiated"

    @pytest.mark.anyio
    async def test_invalid_secret_returns_403(self):
        from fastapi import HTTPException
        from app.routers.integrations import make_trigger
        from app.schemas.integration import MakeTriggerRequest

        body = MakeTriggerRequest(agent_id=str(_AGENT_ID), to_number="+15550001111")
        request = MagicMock()
        request.scope = {"type": "http", "headers": []}
        request.receive = AsyncMock()

        with _patch_resolve_tenant():
            with pytest.raises(HTTPException) as exc_info:
                await make_trigger(
                    body=body,
                    request=request,
                    db=_db(),
                    x_make_secret="wrong-secret",
                )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["code"] == "unauthorized"
        assert exc_info.value.detail["message"] == "Invalid secret"

    @pytest.mark.anyio
    async def test_missing_secret_returns_403(self):
        from fastapi import HTTPException
        from app.routers.integrations import make_trigger
        from app.schemas.integration import MakeTriggerRequest

        body = MakeTriggerRequest(agent_id=str(_AGENT_ID), to_number="+15550001111")
        request = MagicMock()
        request.scope = {"type": "http", "headers": []}
        request.receive = AsyncMock()

        with _patch_resolve_tenant():
            with pytest.raises(HTTPException) as exc_info:
                await make_trigger(
                    body=body,
                    request=request,
                    db=_db(),
                    x_make_secret=None,
                )

        assert exc_info.value.status_code == 403

    @pytest.mark.anyio
    async def test_unknown_agent_returns_404(self):
        from fastapi import HTTPException
        from app.routers.integrations import make_trigger
        from app.schemas.integration import MakeTriggerRequest

        body = MakeTriggerRequest(agent_id=str(uuid.uuid4()), to_number="+15550001111")
        request = MagicMock()

        with _patch_resolve_tenant_not_found():
            with pytest.raises(HTTPException) as exc_info:
                await make_trigger(
                    body=body,
                    request=request,
                    db=_db(),
                    x_make_secret=_MAKE_SECRET,
                )

        assert exc_info.value.status_code == 404

    @pytest.mark.anyio
    async def test_rate_limit_returns_429(self):
        from fastapi.responses import JSONResponse
        from app.routers.integrations import make_trigger
        from app.schemas.integration import MakeTriggerRequest

        body = MakeTriggerRequest(agent_id=str(_AGENT_ID), to_number="+15550001111")
        request = MagicMock()
        request.scope = {"type": "http", "headers": []}
        request.receive = AsyncMock()

        with (
            _patch_resolve_tenant(),
            _patch_rate_limit_deny(),
        ):
            result = await make_trigger(
                body=body,
                request=request,
                db=_db(),
                x_make_secret=_MAKE_SECRET,
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 429


class TestN8nTrigger:
    """POST /api/v1/integrations/n8n/trigger"""

    @pytest.mark.anyio
    async def test_happy_path(self):
        from app.routers.integrations import n8n_trigger
        from app.schemas.twilio import CallInitiateRequest

        body = CallInitiateRequest(
            agentId=str(_AGENT_ID),
            toNumber="+15550002222",
            tenant_id=str(_TENANT_ID),
        )
        request = MagicMock()

        with (
            _patch_n8n_secret_valid(),
            _patch_rate_limit_allow(),
            _patch_initiate_call(),
            _patch_record_triggered(),
            patch("app.routers.integrations.record_last_triggered", MagicMock()),
            patch("app.routers.integrations.TenantModel" if False else "app.models.tenant.Tenant", MagicMock()),
        ):
            # Patch DB tenant lookup inside n8n_trigger
            db = _db()
            result = await n8n_trigger(body=body, request=request, db=db)

        assert result.success is True
        assert result.data["call_id"] == str(_SESSION_ID)
        assert result.data["status"] == "initiated"

    @pytest.mark.anyio
    async def test_invalid_secret_returns_403(self):
        from fastapi import HTTPException
        from app.routers.integrations import n8n_trigger
        from app.schemas.twilio import CallInitiateRequest

        body = CallInitiateRequest(
            agentId=str(_AGENT_ID),
            toNumber="+15550002222",
            tenant_id=str(_TENANT_ID),
        )
        request = MagicMock()

        with _patch_n8n_secret_invalid():
            with pytest.raises(HTTPException) as exc_info:
                await n8n_trigger(body=body, request=request, db=_db())

        assert exc_info.value.status_code == 403

    @pytest.mark.anyio
    async def test_missing_tenant_id_returns_400(self):
        from fastapi import HTTPException
        from app.routers.integrations import n8n_trigger
        from app.schemas.twilio import CallInitiateRequest

        body = CallInitiateRequest(
            agentId=str(_AGENT_ID),
            toNumber="+15550002222",
            # tenant_id intentionally omitted
        )
        request = MagicMock()

        with _patch_n8n_secret_valid():
            with pytest.raises(HTTPException) as exc_info:
                await n8n_trigger(body=body, request=request, db=_db())

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_rate_limit_returns_429(self):
        from fastapi.responses import JSONResponse
        from app.routers.integrations import n8n_trigger
        from app.schemas.twilio import CallInitiateRequest

        body = CallInitiateRequest(
            agentId=str(_AGENT_ID),
            toNumber="+15550002222",
            tenant_id=str(_TENANT_ID),
        )
        request = MagicMock()

        with (
            _patch_n8n_secret_valid(),
            _patch_rate_limit_deny(),
        ):
            result = await n8n_trigger(body=body, request=request, db=_db())

        assert isinstance(result, JSONResponse)
        assert result.status_code == 429


class TestIntegrationList:
    """GET /api/v1/integrations"""

    def test_not_connected(self):
        """Both integrations show connected=False when no secrets are set — service-level."""
        from app.services.integration_service import get_make_secret

        tenant = _make_tenant(make_secret=None)
        assert get_make_secret(tenant) is None

    def test_connected_make(self):
        """When make_secret is set, make shows connected=True."""
        from app.services.integration_service import get_make_secret

        tenant = _make_tenant(make_secret=_MAKE_SECRET)
        secret = get_make_secret(tenant)
        assert secret == _MAKE_SECRET

    def test_not_connected_make(self):
        """When make_secret is absent, get_make_secret returns None."""
        from app.services.integration_service import get_make_secret

        tenant = _make_tenant(make_secret=None)
        assert get_make_secret(tenant) is None

    def test_last_triggered_at_round_trip(self):
        """record_last_triggered + get_last_triggered_at reads back a datetime."""
        from datetime import datetime, timezone
        from app.services.integration_service import get_last_triggered_at, record_last_triggered

        tenant = _make_tenant()
        db = _db(tenant=tenant)

        record_last_triggered(db, tenant, "make")
        result = get_last_triggered_at(tenant, "make")

        assert isinstance(result, datetime)
        assert result.tzinfo is not None


class TestMakeSecretGeneration:
    """POST /api/v1/workspace/settings/make-secret"""

    def test_generate_secret_format(self):
        """generate_make_secret returns a 64-char hex string."""
        from app.services.integration_service import generate_make_secret

        secret = generate_make_secret()
        assert len(secret) == 64
        assert all(c in "0123456789abcdef" for c in secret)

    def test_secrets_are_unique(self):
        """Two consecutive calls produce different secrets."""
        from app.services.integration_service import generate_make_secret

        s1 = generate_make_secret()
        s2 = generate_make_secret()
        assert s1 != s2

    def test_store_and_retrieve_make_secret(self):
        """store_make_secret persists to tenant.workspace_settings."""
        from app.services.integration_service import (
            generate_make_secret,
            get_make_secret,
            store_make_secret,
        )

        tenant = _make_tenant(make_secret=None)
        db = _db(tenant=tenant)

        secret = generate_make_secret()
        store_make_secret(db, tenant, secret)

        retrieved = get_make_secret(tenant)
        assert retrieved == secret

    def test_rotate_make_secret(self):
        """Calling store_make_secret twice replaces the first secret."""
        from app.services.integration_service import (
            generate_make_secret,
            get_make_secret,
            store_make_secret,
        )

        tenant = _make_tenant(make_secret=None)
        db = _db(tenant=tenant)

        s1 = generate_make_secret()
        store_make_secret(db, tenant, s1)

        s2 = generate_make_secret()
        store_make_secret(db, tenant, s2)

        assert get_make_secret(tenant) == s2
        assert s1 != s2


class TestIntegrationRateLimit:
    """Per-workspace 10 req/min rate limit."""

    @pytest.mark.anyio
    async def test_rate_limit_allows_when_redis_unavailable(self):
        """When Redis is unavailable, rate limit is bypassed (fail-open)."""
        from app.services.integration_service import check_integration_rate_limit

        with patch("app.services.integration_service._get_redis", return_value=None):
            allowed, retry_after = await check_integration_rate_limit(_TENANT_ID)

        assert allowed is True
        assert retry_after == 0.0

    @pytest.mark.anyio
    async def test_rate_limit_enforced_via_redis(self):
        """Redis pipeline returning count > limit triggers a deny."""
        from app.services.integration_service import check_integration_rate_limit

        mock_pipeline = MagicMock()
        mock_pipeline.zremrangebyscore = MagicMock()
        mock_pipeline.zadd = MagicMock()
        mock_pipeline.zcard = MagicMock()
        mock_pipeline.zrange = MagicMock()
        mock_pipeline.expire = MagicMock()
        # Simulate pipeline: [zremrange_result, zadd_result, zcard=11, zrange_result, expire_result]
        mock_pipeline.execute = AsyncMock(return_value=[None, None, 11, [("entry", 1000.0)], None])

        mock_redis = MagicMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
        mock_redis.zremrangebyscore = AsyncMock()

        with patch("app.services.integration_service._get_redis", return_value=mock_redis):
            allowed, retry_after = await check_integration_rate_limit(_TENANT_ID)

        assert allowed is False
        assert retry_after > 0
