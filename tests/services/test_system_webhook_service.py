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


# ── _render_headers / _render_query_params ──────────────────────────────────


class TestRenderHeaders:
    """Direct unit tests for `_render_headers` — no DB needed. Covers the
    Issue 1 fix: header NAMES are kept literal (never templated), while
    header VALUES are still rendered against the context."""

    def test_header_key_with_unresolvable_token_is_sent_literal(self):
        from app.services.system_webhook_service import _render_headers

        headers = {"X-{{token}}-Header": "static-value"}
        result = _render_headers(headers, context={})

        # The key must survive byte-for-byte, unresolved template and all —
        # NOT emptied/rendered like a value would be.
        assert "X-{{token}}-Header" in result
        assert result["X-{{token}}-Header"] == "static-value"

    def test_header_value_with_same_token_is_rendered(self):
        from app.services.system_webhook_service import _render_headers

        headers = {"X-Call-Id": "{{_system.callId}}"}
        context = {"_system": {"callId": "abc-123"}}
        result = _render_headers(headers, context)

        assert result["X-Call-Id"] == "abc-123"

    def test_header_value_with_unresolvable_token_renders_empty(self):
        from app.services.system_webhook_service import _render_headers

        headers = {"X-Token": "{{missing.thing}}"}
        result = _render_headers(headers, context={})

        assert result["X-Token"] == ""

    def test_key_literal_value_rendered_in_same_call(self):
        """The core of the fix: exercise both behaviors side by side so a key
        that looks templatable and a value with the identical token diverge —
        key stays literal, value renders (or empties if unresolvable)."""
        from app.services.system_webhook_service import _render_headers

        headers = {"{{_system.eventType}}": "{{_system.eventType}}"}
        context = {"_system": {"eventType": "call.connected"}}
        result = _render_headers(headers, context)

        # Key untouched — still the raw template string.
        assert "{{_system.eventType}}" in result
        # Value resolved.
        assert result["{{_system.eventType}}"] == "call.connected"


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

    def test_sends_json_body_with_from_to_and_static_metadata(self, db, tenant, agent):
        """Issue 2 fix: the outbound POST must carry a real JSON body with
        `from`/`to` nested alongside (not clobbered by) static metadata:
        `{"from": ..., "to": ..., "metadata": {...static_metadata}}`."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            pre_inbound_webhook_static_metadata={"region": "us-east"},
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(200, json_body={"variables": {}}, text="{}")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["json"] = json
            return resp

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            _run(
                fetch_pre_inbound_webhook_variables(
                    db, flow, from_number="+15551112222", to_number="+15553334444"
                )
            )

        assert captured["json"] == {
            "from": "+15551112222",
            "to": "+15553334444",
            "metadata": {"region": "us-east"},
        }

    def test_static_metadata_from_key_does_not_clobber_real_from_number(
        self, db, tenant, agent
    ):
        """Collision-bug fix: a tenant's static metadata may legitimately
        contain a key literally named `"from"` (or `"to"`) — it must land
        nested under `metadata`, not overwrite the real caller-supplied
        `from`/`to` fields via a flat dict-spread."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            pre_inbound_webhook_url="https://example.com/pre-inbound",
            pre_inbound_webhook_static_metadata={
                "from": "spoofed-value",
                "other": "x",
            },
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(200, json_body={"variables": {}}, text="{}")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["json"] = json
            return resp

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            _run(
                fetch_pre_inbound_webhook_variables(
                    db, flow, from_number="+15551112222", to_number="+15553334444"
                )
            )

        json_body = captured["json"]
        assert json_body["from"] == "+15551112222"
        assert json_body["to"] == "+15553334444"
        assert json_body["metadata"]["from"] == "spoofed-value"
        assert json_body["metadata"]["other"] == "x"

    def test_header_key_literal_header_value_rendered_end_to_end(
        self, db, tenant, agent
    ):
        """Issue 1 fix exercised through the real dispatch path: a header key
        with an unresolvable token is sent unrendered, while a header value
        with the same token renders (empties, since it's unresolvable)."""
        flow = _make_flow(
            db, tenant, agent, pre_inbound_webhook_url="https://example.com/pre-inbound"
        )
        from app.services.system_webhook_service import (
            fetch_pre_inbound_webhook_variables,
        )

        resp = _mock_response(200, json_body={"variables": {}}, text="{}")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["headers"] = headers
            return resp

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch(
                "app.services.system_webhook_service._decrypt_headers_safe",
                return_value={"X-{{token}}-Header": "{{token}}"},
            ),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            _run(fetch_pre_inbound_webhook_variables(db, flow, "+1", "+2"))

        headers = captured["headers"]
        # Key kept literal — sent exactly as configured.
        assert "X-{{token}}-Header" in headers
        # Value rendered — unresolvable token becomes empty string, not literal.
        assert headers["X-{{token}}-Header"] == ""

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


# ── run_webhook_test(webhook_kind="post_call") header/query rendering ──────


@pytest.mark.usefixtures("db")
class TestRunWebhookTestPostCallHeaderContext:
    """Issue 3 fix: `run_webhook_test`'s `post_call` branch renders
    headers/query-params against a real field-catalog context
    (`build_post_call_payload_context`), sourced from the tenant's most
    recent `CallSession`, instead of the synthetic test payload dict."""

    def test_resolves_against_tenants_most_recent_call_session(
        self, db, tenant, agent, sw_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_webhook_url="https://example.com/post-call",
        )
        session = _make_session(
            db,
            tenant,
            agent,
            sw_user,
            call_flow=flow,
            status="completed",
            call_metadata={"webhook_variables": {"foo": "bar"}},
        )
        from app.services.system_webhook_service import run_webhook_test

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["headers"] = headers
            captured["params"] = params
            return resp

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch(
                "app.services.system_webhook_service._decrypt_headers_safe",
                return_value={
                    "X-Call-Id": "{{call_metadata.call_id}}",
                    "X-Foo": "{{header_variables.foo}}",
                },
            ),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            _run(run_webhook_test(db, flow.id, tenant.id, "post_call"))

        headers = captured["headers"]
        assert headers["X-Call-Id"] == str(session.id)
        assert headers["X-Foo"] == "bar"

    def test_resolves_query_params_against_recent_call_session(
        self, db, tenant, agent, sw_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_webhook_url="https://example.com/post-call",
            post_call_webhook_query_params=[
                {"key": "call_id", "value": "{{call_metadata.call_id}}"},
                {"key": "status", "value": "{{call_metadata.status}}"},
            ],
        )
        session = _make_session(
            db, tenant, agent, sw_user, call_flow=flow, status="completed"
        )
        from app.services.system_webhook_service import run_webhook_test

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["params"] = params
            return resp

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            _run(run_webhook_test(db, flow.id, tenant.id, "post_call"))

        params = dict(captured["params"])
        assert params["call_id"] == str(session.id)
        assert params["status"] == "completed"

    def test_no_calls_yet_resolves_field_catalog_templates_to_empty_string(
        self, db, tenant, agent
    ):
        """Tenant has no `CallSession` rows at all — header/query templates
        referencing the field catalog must resolve to empty string cleanly,
        not raise."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            post_call_webhook_url="https://example.com/post-call",
        )
        from app.services.system_webhook_service import run_webhook_test

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["headers"] = headers
            captured["params"] = params
            return resp

        with (
            patch("app.services.system_webhook_service.assert_public_url"),
            patch(
                "app.services.system_webhook_service._decrypt_headers_safe",
                return_value={"X-Call-Id": "{{call_metadata.call_id}}"},
            ),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            log = _run(run_webhook_test(db, flow.id, tenant.id, "post_call"))

        assert log.status == "success"
        assert captured["headers"]["X-Call-Id"] == ""


# ── _derive_status_field ─────────────────────────────────────────────────────


class TestDeriveStatusField:
    def test_transfer_with_non_completed_outcome_is_failed(self):
        from app.services.system_webhook_service import _derive_status_field

        assert (
            _derive_status_field("call.transfer", {"outcome": "no-answer"}) == "failed"
        )

    def test_transfer_with_completed_outcome_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert (
            _derive_status_field("call.transfer", {"outcome": "completed"}) == "success"
        )

    def test_transfer_with_no_extra_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.transfer", None) == "success"

    def test_transfer_with_missing_outcome_key_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.transfer", {}) == "success"

    def test_other_event_types_unconditionally_success(self):
        """`call.connected`/`call.test` carry no outcome signal at all — they
        always report success, even if (implausibly) passed an `extra` dict
        that looks failure-like. `call.ended` is exercised separately below
        since it now derives from `extra["outcome"]`, same as `call.transfer`."""
        from app.services.system_webhook_service import _derive_status_field

        for event_type in ("call.connected", "call.test"):
            assert (
                _derive_status_field(event_type, {"outcome": "no-answer"}) == "success"
            )

    def test_ended_with_completed_outcome_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.ended", {"outcome": "completed"}) == "success"

    @pytest.mark.parametrize("outcome", ["no_answer", "failed", "busy"])
    def test_ended_with_non_completed_outcome_is_failed(self, outcome):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.ended", {"outcome": outcome}) == "failed"

    def test_ended_with_no_extra_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.ended", None) == "success"

    def test_ended_with_empty_extra_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.ended", {}) == "success"

    def test_ended_with_missing_outcome_key_is_success(self):
        from app.services.system_webhook_service import _derive_status_field

        assert _derive_status_field("call.ended", {"other": "x"}) == "success"


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

    def test_payload_status_field_failed_for_non_completed_transfer(
        self, db, tenant, agent, sw_user
    ):
        """Issue 4 fix: `status` is derived, not hardcoded to "success"."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
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
            _run(
                _dispatch_status_webhook(
                    session.id, "call.transfer", extra={"outcome": "no-answer"}
                )
            )

        assert captured["json"]["status"] == "failed"

    def test_payload_status_field_success_for_completed_transfer(
        self, db, tenant, agent, sw_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
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
            _run(
                _dispatch_status_webhook(
                    session.id, "call.transfer", extra={"outcome": "completed"}
                )
            )

        assert captured["json"]["status"] == "success"

    def test_payload_status_field_success_for_non_transfer_event(
        self, db, tenant, agent, sw_user
    ):
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
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
            _run(_dispatch_status_webhook(session.id, "call.connected"))

        assert captured["json"]["status"] == "success"

    def test_header_query_context_resolves_system_fields_only(
        self, db, tenant, agent, sw_user
    ):
        """Issue 5 fix: header/query templates render against a namespaced
        `{"_system": {"callId": ..., "eventType": ...}}` context built fresh
        for the status webhook — NOT against the outgoing payload dict. A
        stray `{{status}}`/`{{apiName}}` token (payload fields, not `_system`
        fields) must render empty rather than resolve."""
        flow = _make_flow(
            db,
            tenant,
            agent,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
            status_webhook_query_params=[
                {"key": "call_id", "value": "{{_system.callId}}"},
                {"key": "event", "value": "{{_system.eventType}}"},
                {"key": "stray", "value": "{{status}}"},
            ],
        )
        session = _make_session(db, tenant, agent, sw_user, call_flow=flow)
        from app.services.system_webhook_service import _dispatch_status_webhook

        resp = _mock_response(200, text="ok")
        captured = {}

        async def _fake_post(self, url, params=None, headers=None, json=None):
            captured["headers"] = headers
            captured["params"] = params
            return resp

        with (
            _session_local_returns(db),
            patch("app.services.system_webhook_service.assert_public_url"),
            patch(
                "app.services.system_webhook_service._decrypt_headers_safe",
                return_value={
                    "X-Call-Id": "{{_system.callId}}",
                    "X-Event-Type": "{{_system.eventType}}",
                    "X-Stray": "{{apiName}}",
                },
            ),
            patch("httpx.AsyncClient.post", new=_fake_post),
        ):
            _run(_dispatch_status_webhook(session.id, "call.connected"))

        headers = captured["headers"]
        assert headers["X-Call-Id"] == str(session.id)
        assert headers["X-Event-Type"] == "call.connected"
        # Not resolvable against `_system`-only context — renders empty.
        assert headers["X-Stray"] == ""

        params = dict(captured["params"])
        assert params["call_id"] == str(session.id)
        assert params["event"] == "call.connected"
        assert params["stray"] == ""


def _run(coro):
    import asyncio

    return asyncio.run(coro)
