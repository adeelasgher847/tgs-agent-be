"""Unit tests for System Webhooks' Pre-Inbound Call Webhook + Dynamic Inbound
Call Routing wiring in app/routers/voice.py, plus the connect/transfer
Status Webhook dispatch call sites.

DB: in-memory SQLite fixture, same pattern as
tests/routers/test_voice_twilio_hardening.py.

Coverage:
  - `_resolve_default_inbound_call_flow`: active/inbound/non-deleted match,
    most-recently-updated ordering, None when nothing matches.
  - `handle_incoming_call` Dynamic Inbound Call Routing: valid override,
    malformed UUID fallback, nonexistent/other-tenant/deleted agent
    fallback, toggle-disabled ignored, webhook-failure fallback.
  - `call_session.call_flow_id` / `call_metadata["webhook_variables"]`
    attached correctly; no-webhook-configured behavior unchanged
    (regression guard).
  - `handle_call_events_webhook` connect-event dedup: fires
    schedule_status_webhook("call.connected") only on the transition INTO
    "connected", not on repeat callbacks.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.models.agent import Agent


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


_SHARED_CONN = sqlite3.connect(":memory:", check_same_thread=False)
_engine = create_engine(
    "sqlite://", creator=lambda: _SHARED_CONN, connect_args={"check_same_thread": False}
)
_Session = sessionmaker(bind=_engine)


def _setup_db():
    import app.models.agent  # noqa: F401
    import app.models.call_flow  # noqa: F401
    import app.models.call_log  # noqa: F401
    import app.models.call_session  # noqa: F401
    import app.models.phone_number  # noqa: F401
    import app.models.plan  # noqa: F401
    import app.models.role  # noqa: F401
    import app.models.subscription  # noqa: F401
    import app.models.system_webhook_log  # noqa: F401
    import app.models.tenant  # noqa: F401
    import app.models.usage_record  # noqa: F401
    import app.models.user  # noqa: F401

    Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)


@pytest.fixture()
def db():
    _setup_db()
    session = _Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _bypass_signature_validation():
    with patch.object(settings, "ALLOW_UNAUTHENTICATED_WEBHOOKS", True):
        yield


class _FakeURL:
    def __init__(self, path="/api/v1/voice/incoming"):
        self.path = path
        self.query = ""
        self.scheme = "https"
        self.netloc = "example.com"


class _FakeRequest:
    def __init__(self, form_data: dict, headers: dict | None = None):
        self._form_data = form_data
        self.headers = headers or {}
        self.method = "POST"
        self.url = _FakeURL()

    async def form(self):
        return self._form_data

    async def body(self):
        return b""


def _make_tenant(db, **overrides):
    from app.models.tenant import Tenant

    fields = {
        "name": f"RoutingWS-{uuid.uuid4().hex[:8]}",
        "schema_name": f"routing_ws_{uuid.uuid4().hex[:8]}",
    }
    fields.update(overrides)
    t = Tenant(**fields)
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _make_user(db):
    from app.models.user import User

    u = User(
        id=uuid.uuid4(),
        first_name="Routing",
        last_name="Test",
        email=f"routing-{uuid.uuid4().hex}@example.com",
        hashed_password="x",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_agent(db, tenant_id, *, name="RoutingAgent", is_deleted=False):
    user = _make_user(db)
    a = Agent(
        tenant_id=tenant_id,
        name=name,
        system_prompt="",
        status="ready",
        is_deleted=is_deleted,
        created_by=user.id,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _make_phone_number(db, tenant_id, agent_id, number="+15550009999"):
    from app.models.phone_number import PhoneNumber

    pn = PhoneNumber(
        tenant_id=tenant_id,
        phone_number=number,
        status="active",
        assistant_id=agent_id,
    )
    db.add(pn)
    db.commit()
    db.refresh(pn)
    return pn


def _make_call_flow(
    db,
    tenant_id,
    agent_id,
    *,
    direction="inbound",
    status="active",
    is_deleted=False,
    **overrides,
):
    from app.models.call_flow import CallFlow

    fields = {
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "name": "Routing Flow",
        "direction": direction,
        "status": status,
        "is_deleted": is_deleted,
    }
    fields.update(overrides)
    f = CallFlow(**fields)
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _incoming_call_mocks():
    """Context-manager stack of patches needed to get handle_incoming_call
    past the credit/model/TTS-resolution gate without touching a real Model
    row, TTS provider, or credit ledger — none of which are under test here."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch.object(
            Agent,
            "model",
            new_callable=PropertyMock,
            return_value=MagicMock(model_name="gpt-4o-mini"),
        )
    )
    stack.enter_context(
        patch(
            "app.routers.voice.resolve_tts_runtime",
            return_value=MagicMock(adapter_slug="elevenlabs", is_byo_elevenlabs=False),
        )
    )
    stack.enter_context(
        patch(
            "app.routers.voice.credit_service.has_sufficient_credits",
            return_value=(True, 100.0, 1.0, None),
        )
    )
    return stack


def _run_incoming_call(db, form_data):
    from app.routers.voice import handle_incoming_call

    request = _FakeRequest(form_data)
    with _incoming_call_mocks():
        return asyncio.run(handle_incoming_call(request=request, db=db))


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_default_inbound_call_flow
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveDefaultInboundCallFlow:
    def test_returns_active_inbound_flow(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        flow = _make_call_flow(db, tenant.id, agent.id, direction="inbound")

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is not None
        assert result.id == flow.id

    def test_returns_bidirectional_flow(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        flow = _make_call_flow(db, tenant.id, agent.id, direction="bidirectional")

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is not None
        assert result.id == flow.id

    def test_ignores_outbound_only_flow(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        _make_call_flow(db, tenant.id, agent.id, direction="outbound")

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is None

    def test_ignores_inactive_flow(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        _make_call_flow(db, tenant.id, agent.id, direction="inbound", status="inactive")

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is None

    def test_ignores_deleted_flow(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        _make_call_flow(db, tenant.id, agent.id, direction="inbound", is_deleted=True)

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is None

    def test_returns_none_when_no_flow_exists(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is None

    def test_returns_most_recently_updated_when_multiple_match(self, db):
        from datetime import datetime, timedelta, timezone

        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        older = _make_call_flow(
            db, tenant.id, agent.id, direction="inbound", name="Older"
        )
        newer = _make_call_flow(
            db, tenant.id, agent.id, direction="inbound", name="Newer"
        )

        now = datetime.now(timezone.utc)
        older.updated_at = now - timedelta(days=1)
        newer.updated_at = now
        db.commit()

        result = _resolve_default_inbound_call_flow(db, agent.id, tenant.id)
        assert result is not None
        assert result.id == newer.id

    def test_other_tenants_flow_not_returned(self, db):
        from app.routers.voice import _resolve_default_inbound_call_flow

        tenant_a = _make_tenant(db)
        tenant_b = _make_tenant(db)
        agent_a = _make_agent(db, tenant_a.id)
        _make_call_flow(db, tenant_a.id, agent_a.id, direction="inbound")

        result = _resolve_default_inbound_call_flow(db, agent_a.id, tenant_b.id)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# handle_incoming_call — Dynamic Inbound Call Routing
# ─────────────────────────────────────────────────────────────────────────────


class TestDynamicInboundCallRouting:
    def test_no_flow_configured_behaves_unchanged(self, db):
        """Regression guard: an inbound call with no CallFlow at all must
        route to the number's default agent exactly as before this feature."""
        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        pn = _make_phone_number(db, tenant.id, agent.id)

        resp = _run_incoming_call(
            db, {"CallSid": "CA1", "From": "+15551110000", "To": pn.phone_number}
        )

        assert resp.status_code == 200
        assert str(agent.id) in resp.body.decode()

        from app.models.call_session import CallSession

        session = (
            db.query(CallSession).filter(CallSession.twilio_call_sid == "CA1").first()
        )
        assert session is not None
        assert session.agent_id == agent.id
        assert session.call_flow_id is None
        assert not (session.call_metadata or {}).get("webhook_variables")

    def test_valid_agent_id_override_routes_to_that_agent(self, db):
        tenant = _make_tenant(db)
        default_agent = _make_agent(db, tenant.id, name="DefaultAgent")
        override_agent = _make_agent(db, tenant.id, name="OverrideAgent")
        pn = _make_phone_number(db, tenant.id, default_agent.id)
        _make_call_flow(
            db,
            tenant.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=True,
        )
        override_flow = _make_call_flow(
            db, tenant.id, override_agent.id, name="Override Flow"
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"agent_id": str(override_agent.id)}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA2", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(override_agent.id) in resp.body.decode()

        from app.models.call_session import CallSession

        session = (
            db.query(CallSession).filter(CallSession.twilio_call_sid == "CA2").first()
        )
        assert session.agent_id == override_agent.id
        assert session.call_flow_id == override_flow.id

    def test_malformed_agent_id_falls_back_to_default_without_raising(self, db):
        tenant = _make_tenant(db)
        default_agent = _make_agent(db, tenant.id)
        pn = _make_phone_number(db, tenant.id, default_agent.id)
        _make_call_flow(
            db,
            tenant.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=True,
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"agent_id": "not-a-uuid"}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA3", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(default_agent.id) in resp.body.decode()

    def test_nonexistent_agent_id_falls_back_to_default(self, db):
        tenant = _make_tenant(db)
        default_agent = _make_agent(db, tenant.id)
        pn = _make_phone_number(db, tenant.id, default_agent.id)
        _make_call_flow(
            db,
            tenant.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=True,
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"agent_id": str(uuid.uuid4())}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA4", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(default_agent.id) in resp.body.decode()

    def test_other_tenant_agent_id_falls_back_to_default(self, db):
        tenant_a = _make_tenant(db)
        tenant_b = _make_tenant(db)
        default_agent = _make_agent(db, tenant_a.id)
        other_tenant_agent = _make_agent(db, tenant_b.id, name="OtherTenantAgent")
        pn = _make_phone_number(db, tenant_a.id, default_agent.id)
        _make_call_flow(
            db,
            tenant_a.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=True,
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"agent_id": str(other_tenant_agent.id)}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA5", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(default_agent.id) in resp.body.decode()

    def test_deleted_agent_id_falls_back_to_default(self, db):
        tenant = _make_tenant(db)
        default_agent = _make_agent(db, tenant.id)
        deleted_agent = _make_agent(db, tenant.id, name="DeletedAgent", is_deleted=True)
        pn = _make_phone_number(db, tenant.id, default_agent.id)
        _make_call_flow(
            db,
            tenant.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=True,
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"agent_id": str(deleted_agent.id)}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA6", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(default_agent.id) in resp.body.decode()

    def test_toggle_disabled_ignores_agent_id_even_if_valid(self, db):
        tenant = _make_tenant(db)
        default_agent = _make_agent(db, tenant.id)
        override_agent = _make_agent(db, tenant.id, name="OverrideAgent")
        pn = _make_phone_number(db, tenant.id, default_agent.id)
        _make_call_flow(
            db,
            tenant.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=False,
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"agent_id": str(override_agent.id)}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA7", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(default_agent.id) in resp.body.decode()

    def test_webhook_failure_falls_back_cleanly(self, db):
        """fetch_pre_inbound_webhook_variables returning {} (its fail-open
        contract on any delivery failure) must not block the call."""
        tenant = _make_tenant(db)
        default_agent = _make_agent(db, tenant.id)
        pn = _make_phone_number(db, tenant.id, default_agent.id)
        _make_call_flow(
            db,
            tenant.id,
            default_agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            dynamic_inbound_routing_enabled=True,
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA8", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200
        assert str(default_agent.id) in resp.body.decode()

    def test_webhook_variables_and_call_flow_id_attached_to_session(self, db):
        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        pn = _make_phone_number(db, tenant.id, agent.id)
        flow = _make_call_flow(
            db,
            tenant.id,
            agent.id,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
        )

        with patch(
            "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
            new=AsyncMock(return_value={"account_tier": "gold"}),
        ):
            resp = _run_incoming_call(
                db, {"CallSid": "CA9", "From": "+1", "To": pn.phone_number}
            )

        assert resp.status_code == 200

        from app.models.call_session import CallSession

        session = (
            db.query(CallSession).filter(CallSession.twilio_call_sid == "CA9").first()
        )
        assert session.call_flow_id == flow.id
        assert session.call_metadata["webhook_variables"] == {"account_tier": "gold"}


# ─────────────────────────────────────────────────────────────────────────────
# handle_call_events_webhook — Status Webhook "connect" dedup
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusWebhookConnectDedup:
    def _make_call_session(self, db, tenant_id, agent_id, *, status="ringing"):
        from datetime import datetime, timezone

        from app.models.call_session import CallSession

        user = _make_user(db)
        cs = CallSession(
            id=uuid.uuid4(),
            user_id=user.id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            start_time=datetime.now(timezone.utc),
            status=status,
            call_type="outbound",
            twilio_call_sid="CAconnect1",
        )
        db.add(cs)
        db.commit()
        db.refresh(cs)
        return cs

    def test_fires_once_on_transition_into_connected(self, db):
        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        flow = _make_call_flow(
            db,
            tenant.id,
            agent.id,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        cs = self._make_call_session(db, tenant.id, agent.id, status="ringing")
        cs.call_flow_id = flow.id
        db.commit()

        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest(
            {
                "CallStatus": "in-progress",
                "CallSid": cs.twilio_call_sid,
                "From": "+1",
                "To": "+2",
                "Direction": "inbound",
            }
        )

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring"),
            patch("app.routers.voice.notify_batch_call_ended", new=AsyncMock()),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_schedule,
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=str(agent.id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        mock_schedule.assert_called_once_with(cs.id, "call.connected")

    def test_does_not_refire_on_repeat_connected_callback(self, db):
        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        flow = _make_call_flow(
            db,
            tenant.id,
            agent.id,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        cs = self._make_call_session(db, tenant.id, agent.id, status="connected")
        cs.call_flow_id = flow.id
        db.commit()

        from app.routers.voice import handle_call_events_webhook

        request = _FakeRequest(
            {
                "CallStatus": "in-progress",
                "CallSid": cs.twilio_call_sid,
                "From": "+1",
                "To": "+2",
                "Direction": "inbound",
            }
        )

        with (
            patch("app.routers.voice.credit_service.stop_credit_monitoring"),
            patch("app.routers.voice.notify_batch_call_ended", new=AsyncMock()),
            patch(
                "app.services.system_webhook_service.schedule_status_webhook"
            ) as mock_schedule,
        ):
            asyncio.run(
                handle_call_events_webhook(
                    request=request,
                    background_tasks=BackgroundTasks(),
                    agentId=str(agent.id),
                    userId=None,
                    callSessionId=str(cs.id),
                    timeout=None,
                    body="",
                    db=db,
                )
            )

        mock_schedule.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# transfer_webhook_dial_complete — Status Webhook "transfer" event
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusWebhookTransferDispatch:
    def _make_call_session(self, db, tenant_id, agent_id):
        from datetime import datetime, timezone

        from app.models.call_session import CallSession

        user = _make_user(db)
        cs = CallSession(
            id=uuid.uuid4(),
            user_id=user.id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            start_time=datetime.now(timezone.utc),
            status="active",
            call_type="outbound",
            twilio_call_sid="CAtransfer1",
        )
        db.add(cs)
        db.commit()
        db.refresh(cs)
        return cs

    def test_fires_transfer_event_with_dial_outcome(self, db):
        from app.routers.voice import transfer_webhook_dial_complete

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        flow = _make_call_flow(
            db,
            tenant.id,
            agent.id,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        cs = self._make_call_session(db, tenant.id, agent.id)
        cs.call_flow_id = flow.id
        db.commit()

        request = _FakeRequest(
            {
                "CallSid": cs.twilio_call_sid,
                "DialCallStatus": "completed",
                "DialCallSid": "CAdial1",
            }
        )

        with patch(
            "app.services.system_webhook_service.schedule_status_webhook"
        ) as mock_schedule:
            resp = asyncio.run(
                transfer_webhook_dial_complete(
                    request=request, callSessionId=str(cs.id), db=db
                )
            )

        assert resp.status_code == 200
        mock_schedule.assert_called_once_with(
            cs.id,
            "call.transfer",
            extra={"outcome": "completed", "dial_call_sid": "CAdial1"},
        )

    def test_no_op_when_status_webhook_not_enabled(self, db):
        from app.routers.voice import transfer_webhook_dial_complete

        tenant = _make_tenant(db)
        agent = _make_agent(db, tenant.id)
        flow = _make_call_flow(
            db,
            tenant.id,
            agent.id,
            status_webhook_enabled=False,
            status_webhook_url="https://example.com/status",
        )
        cs = self._make_call_session(db, tenant.id, agent.id)
        cs.call_flow_id = flow.id
        db.commit()

        request = _FakeRequest(
            {
                "CallSid": cs.twilio_call_sid,
                "DialCallStatus": "no-answer",
                "DialCallSid": "CAdial2",
            }
        )

        with patch(
            "app.services.system_webhook_service.schedule_status_webhook"
        ) as mock_schedule:
            resp = asyncio.run(
                transfer_webhook_dial_complete(
                    request=request, callSessionId=str(cs.id), db=db
                )
            )

        assert resp.status_code == 200
        mock_schedule.assert_not_called()
