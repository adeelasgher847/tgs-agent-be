"""Unit tests for app.services.system_webhook_service — dispatch, delivery,
and retry for the System Webhooks (Call Flow) feature.

Mirrors tests/api/v2/test_webhooks.py's httpx-mocking conventions (patch
httpx.AsyncClient.post + assert_public_url) since this module intentionally
parallels app/services/webhook_service.py's SSRF-guarded delivery shape but
isn't the same module.

DB: real SQLite `db` fixture (tests/conftest.py) — SystemWebhookDeliveryLog
rows are asserted via direct queries. `schedule_post_call_webhook`/
`schedule_status_webhook`/`_dispatch_*` open their own SessionLocal, so tests
that hit those functions patch `app.db.session.SessionLocal` to return the
shared test `db` session (same connection).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.system_webhook_log import SystemWebhookDeliveryLog
from app.models.tenant import Tenant
from app.models.user import User


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"SysWebhookSvcWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"sys_webhook_svc_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def agent(db, tenant: Tenant) -> Agent:
    a = Agent(
        tenant_id=tenant.id,
        name="System Webhook Service Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@pytest.fixture
def sw_user(db, tenant: Tenant) -> User:
    u = User(
        email=f"sys-webhook-svc-{uuid.uuid4().hex[:8]}@example.com",
        first_name="SysWebhook",
        last_name="Svc",
        hashed_password="",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_flow(db, tenant, agent, **overrides) -> CallFlow:
    f = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="System Webhook Service Flow",
        direction="inbound",
        **overrides,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _make_session(
    db, tenant, agent, user, *, call_flow=None, **overrides
) -> CallSession:
    from datetime import datetime, timezone

    fields = {
        "user_id": user.id,
        "agent_id": agent.id,
        "tenant_id": tenant.id,
        "call_flow_id": call_flow.id if call_flow else None,
        "status": "active",
        "call_type": "web",
        "start_time": datetime.now(timezone.utc),
    }
    fields.update(overrides)
    session = CallSession(**fields)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


class _NoCloseSessionProxy:
    """Wraps the shared test `db` session so `_dispatch_*`'s own
    `db.close()` (in its `finally` block) doesn't tear down the
    module-scoped fixture session out from under later tests in this file."""

    def __init__(self, real_session):
        object.__setattr__(self, "_real_session", real_session)

    def close(self):
        pass  # deliberately a no-op — see class docstring

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real_session"), name)


@contextmanager
def _session_local_returns(db):
    """Patch SessionLocal() so the module-under-test's own DB session is the
    same connection/transaction as the test's `db` fixture, so committed rows
    are visible to assertions made via `db` afterward."""
    with patch("app.db.session.SessionLocal", return_value=_NoCloseSessionProxy(db)):
        yield


def _mock_response(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_body is not None:
        resp.json = MagicMock(return_value=json_body)
    else:
        resp.json = MagicMock(side_effect=ValueError("no json"))
    return resp


# ── (1) fetch_pre_inbound_webhook_variables ─────────────────────────────────


@pytest.mark.usefixtures("db")
class TestFetchPreInboundWebhookVariables:
    def test_success_returns_variables_and_logs_success(self, db, tenant, agent):
        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://example.com/pre-inbound"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(
            200, json_body={"variables": {"a": "b"}}, text='{"variables": {"a": "b"}}'
        )

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
        ):
            result = _run(
                fetch_pre_inbound_webhook_variables(
                    db, flow, from_number="+15551112222", to_number="+15553334444"
                )
            )

        assert result == {"a": "b"}
        logs = (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .all()
        )
        assert len(logs) == 1
        assert logs[0].status == "success"
        assert logs[0].webhook_kind == "pre_inbound"

    def test_non_2xx_returns_empty_dict_and_logs_failed(self, db, tenant, agent):
        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://example.com/pre-inbound"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(500, text="server error")

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
        ):
            result = _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        assert result == {}
        logs = (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .all()
        )
        assert logs[0].status == "failed"

    def test_timeout_returns_empty_dict_and_logs_timeout(self, db, tenant, agent):
        import httpx

        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://example.com/pre-inbound"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch(
                "httpx.AsyncClient.post",
                new=AsyncMock(side_effect=httpx.TimeoutException("timed out")),
            ),
        ):
            result = _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        assert result == {}
        logs = (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .all()
        )
        assert logs[0].status == "timeout"

    def test_malformed_json_body_returns_empty_dict_never_raises(
        self, db, tenant, agent
    ):
        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://example.com/pre-inbound"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(200, json_body=None, text="not-json{{")

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
        ):
            result = _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        assert result == {}
        # Delivery itself still succeeded (HTTP 200) — only variable parsing failed.
        logs = (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .all()
        )
        assert logs[0].status == "success"

    def test_non_string_variable_value_is_dropped_not_coerced(self, db, tenant, agent):
        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://example.com/pre-inbound"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(
            200,
            json_body={"variables": {"agent_id": "abc", "call_count": 3}},
            text="{}",
        )

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
        ):
            result = _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        assert result == {"agent_id": "abc"}
        assert "call_count" not in result

    def test_no_url_configured_returns_empty_dict_without_delivery_attempt(
        self, db, tenant, agent
    ):
        flow = _make_flow(db, tenant, agent, pre_inbound_webhook_url=None)
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            result = _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        assert result == {}
        mock_post.assert_not_called()
        assert (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .count()
            == 0
        )

    def test_ssrf_guard_invoked_with_rendered_url(self, db, tenant, agent):
        flow = _make_flow(
            db,
            tenant,
            agent,
            pre_inbound_webhook_url="https://example.com/webhook?to={{_system.phoneNumber}}",
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(200, json_body={"variables": {}})

        with (
            patch("app.services.system_webhook_service.assert_public_url") as mock_ssrf,
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
        ):
            _run(
                fetch_pre_inbound_webhook_variables(
                    db, flow, from_number="+1", to_number="+15559990000"
                )
            )

        mock_ssrf.assert_called_once_with("https://example.com/webhook?to=+15559990000")

    def test_ssrf_blocked_url_returns_empty_dict_no_real_network_call(
        self, db, tenant, agent
    ):
        from app.utils.ssrf import SSRFBlockedError

        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://169.254.169.254/hook"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        with (
            patch(
                "app.services.system_webhook_service.assert_public_url",
                side_effect=SSRFBlockedError("blocked: private IP"),
            ),
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
        ):
            result = _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        assert result == {}
        mock_post.assert_not_called()
        logs = (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .all()
        )
        assert logs[0].status == "failed"
        assert "SSRF" in (logs[0].error or "")


# ── (3) Post-Call Webhook ────────────────────────────────────────────────────


@pytest.mark.usefixtures("db")
class TestPostCallWebhookDispatch:
    def test_no_op_when_url_unset(self, db, tenant, agent, sw_user):
        flow = _make_flow(db, tenant, agent, post_call_webhook_url=None)
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import _dispatch_post_call_webhook

        with (
            _session_local_returns(db),
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
        ):
            success = _run(_dispatch_post_call_webhook(session.id))

        assert success is True
        mock_post.assert_not_called()

    def test_no_op_when_call_flow_id_unset(self, db, tenant, agent, sw_user):
        session = _make_session(db, tenant, agent, sw_user, call_flow=None)
        from app.services.system_webhook_service import _dispatch_post_call_webhook

        with (
            _session_local_returns(db),
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
        ):
            success = _run(_dispatch_post_call_webhook(session.id))

        assert success is True
        mock_post.assert_not_called()

    def test_default_payload_shape_when_custom_payload_disabled(
        self, db, tenant, agent, sw_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_webhook_url="https://example.com/post-call",
            post_call_webhook_custom_payload_enabled=False,
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import _dispatch_post_call_webhook

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["json"] = json
            return resp

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            success = _run(_dispatch_post_call_webhook(session.id))

        assert success is True
        payload = captured["json"]
        assert set(payload.keys()) == {"callId", "agentId", "timestamp", "data"}
        assert payload["callId"] == str(session.id)
        assert payload["agentId"] == str(agent.id)

    def test_custom_payload_rendered_via_field_catalog(
        self, db, tenant, agent, sw_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_webhook_url="https://example.com/post-call",
            post_call_webhook_custom_payload_enabled=True,
            post_call_webhook_custom_payload_template={
                "call_id": "{{call_metadata.call_id}}",
                "status": "{{call_metadata.status}}",
            },
        )
        session = _make_session(
            db, tenant, agent, sw_user, call_flow=flow, status="completed"
        )
        from app.services.system_webhook_service import _dispatch_post_call_webhook

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["json"] = json
            return resp

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            success = _run(_dispatch_post_call_webhook(session.id))

        assert success is True
        assert captured["json"] == {
            "call_id": str(session.id),
            "status": "completed",
        }

    def test_failure_triggers_bounded_retry_enqueue(self, db, tenant, agent, sw_user):
        flow = _make_flow(
            db, tenant, agent, post_call_webhook_url="https://example.com/post-call"
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import run_post_call_webhook

        resp = _mock_response(500, text="fail")

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
            patch(
                "app.services.system_webhook_service._schedule_webhook_retry",
                new=AsyncMock(),
            ) as mock_retry,
        ):
            _run(run_post_call_webhook(session.id))

        mock_retry.assert_awaited_once_with("post_call", session.id, attempt_number=1)

    def test_success_does_not_trigger_retry_enqueue(self, db, tenant, agent, sw_user):
        flow = _make_flow(
            db, tenant, agent, post_call_webhook_url="https://example.com/post-call"
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import run_post_call_webhook

        resp = _mock_response(200, text="ok")

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
            patch(
                "app.services.system_webhook_service._schedule_webhook_retry",
                new=AsyncMock(),
            ) as mock_retry,
        ):
            _run(run_post_call_webhook(session.id))

        mock_retry.assert_not_awaited()


# ── (4) Status Webhook ────────────────────────────────────────────────────────


@pytest.mark.usefixtures("db")
class TestStatusWebhookDispatch:
    def test_no_op_when_disabled(self, db, tenant, agent, sw_user):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=False,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import _dispatch_status_webhook

        with (
            _session_local_returns(db),
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
        ):
            success = _run(_dispatch_status_webhook(session.id, "call.connected"))

        assert success is True
        mock_post.assert_not_called()

    def test_no_op_when_no_call_flow(self, db, tenant, agent, sw_user):
        session = _make_session(db, tenant, agent, sw_user, call_flow=None)
        from app.services.system_webhook_service import _dispatch_status_webhook

        with (
            _session_local_returns(db),
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
        ):
            success = _run(_dispatch_status_webhook(session.id, "call.connected"))

        assert success is True
        mock_post.assert_not_called()

    def test_payload_shape_and_extra_merge(self, db, tenant, agent, sw_user):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow, duration=42)
        from app.services.system_webhook_service import _dispatch_status_webhook

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["json"] = json
            return resp

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            success = _run(
                _dispatch_status_webhook(
                    session.id, "call.transfer", extra={"outcome": "completed"}
                )
            )

        assert success is True
        payload = captured["json"]
        assert payload["callId"] == str(session.id)
        assert payload["apiName"] == "call.transfer"
        assert payload["duration"] == 42
        assert payload["outcome"] == "completed"

    def test_failure_triggers_bounded_retry_enqueue(self, db, tenant, agent, sw_user):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import run_status_webhook

        resp = _mock_response(500, text="fail")

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)),
            patch(
                "app.services.system_webhook_service._schedule_webhook_retry",
                new=AsyncMock(),
            ) as mock_retry,
        ):
            _run(run_status_webhook(session.id, "call.ended"))

        mock_retry.assert_awaited_once_with(
            "status",
            session.id,
            attempt_number=1,
            event_type="call.ended",
            extra=None,
        )


def _run(coro):
    import asyncio

    return asyncio.run(coro)
