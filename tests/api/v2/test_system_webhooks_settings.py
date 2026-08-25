"""Tests for PUT /api/v2/flows/{flow_id}/system-webhooks-settings and
POST /api/v2/flows/{flow_id}/system-webhooks/test.

Mirrors tests/api/v2/test_post_call_actions_settings.py's fixture/app-factory
pattern for RBAC + tenant isolation + audit-log coverage.

Coverage:
  - Admin/API-key principal can save all four sub-features in one PUT
  - Non-admin principal is forbidden
  - Unknown flow_id / other-tenant flow both 404 (tenant isolation)
  - SSRF-blocked URL rejected at schema-validation time (400)
  - Header partial-update asymmetry: `None` leaves stored headers unchanged,
    `{}` explicitly clears them, a dict replaces them
  - Audit log excludes header/query-param *values*, only logs
    presence-booleans / URLs / toggles / query-param keys
  - POST .../system-webhooks/test delegates to run_webhook_test and returns
    the SystemWebhookTestResult shape
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers


def _build_app(db_override, principal, *, forbidden=False):
    from app.api.deps import get_db, require_admin_or_api_key
    from app.api.v2.routers.flows import router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(router)

    if forbidden:

        def _raise_forbidden():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )

        mini.dependency_overrides[require_admin_or_api_key] = _raise_forbidden
    else:
        mini.dependency_overrides[require_admin_or_api_key] = lambda: principal

    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _bypass_ssrf(monkeypatch):
    """Skip real SSRF DNS lookups for tests that aren't specifically testing
    the SSRF guard itself — mirrors tests/api/v2/test_webhooks.py."""
    monkeypatch.setattr("app.schemas.call_flow.assert_public_url", lambda url: None)


@pytest.fixture(autouse=True)
def _fake_pgcrypto_header_encryption(monkeypatch):
    """The `db` fixture here is SQLite, which has no pgp_sym_encrypt function.
    Stand in a deterministic, reversible base64-json "encryption" so
    call_flow_service.update_system_webhooks_settings's header-encryption
    call sites can be exercised end-to-end without a real Postgres."""
    import base64
    import json

    def _fake_encrypt(headers, db):
        if not headers:
            return None
        return "b64:" + base64.b64encode(json.dumps(headers).encode()).decode()

    monkeypatch.setattr(
        "app.services.call_flow_service.encrypt_webhook_headers", _fake_encrypt
    )


@pytest.fixture
def workspace(db):
    from app.models.tenant import Tenant

    tenant = Tenant(
        name=f"SysWebhooksWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"sys_webhooks_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def agent(db, workspace):
    from app.models.agent import Agent

    a = Agent(
        tenant_id=workspace.id,
        name="System Webhooks Test Agent",
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
def flow(db, workspace, agent):
    from app.models.call_flow import CallFlow

    f = CallFlow(
        tenant_id=workspace.id,
        agent_id=agent.id,
        name="System Webhooks Test Flow",
        direction="inbound",
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def _admin_principal(tenant_id: uuid.UUID) -> MagicMock:
    principal = MagicMock()
    principal.id = uuid.uuid4()
    principal.current_tenant_id = tenant_id
    return principal


_FULL_PAYLOAD = {
    "pre_inbound_webhook_url": "https://example.com/pre-inbound",
    "pre_inbound_webhook_headers": {"X-Api-Key": "abc123"},
    "pre_inbound_webhook_query_params": [{"key": "src", "value": "twilio"}],
    "pre_inbound_webhook_static_metadata": {"region": "us-east"},
    "dynamic_inbound_routing_enabled": True,
    "post_call_webhook_url": "https://example.com/post-call",
    "post_call_webhook_headers": {"Authorization": "Bearer xyz"},
    "post_call_webhook_query_params": [{"key": "env", "value": "prod"}],
    "post_call_webhook_custom_payload_enabled": True,
    "post_call_webhook_custom_payload_template": {
        "callId": "{{call_metadata.call_id}}"
    },
    "status_webhook_enabled": True,
    "status_webhook_url": "https://example.com/status",
    "status_webhook_headers": {"X-Token": "tok"},
    "status_webhook_query_params": [{"key": "v", "value": "1"}],
}


@pytest.mark.usefixtures("db")
class TestUpdateSystemWebhooksSettings:
    def test_admin_can_save_all_four_sub_features(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["pre_inbound_webhook_url"] == "https://example.com/pre-inbound"
        assert body["pre_inbound_webhook_headers_configured"] is True
        assert body["pre_inbound_webhook_query_params"] == [
            {"key": "src", "value": "twilio"}
        ]
        assert body["pre_inbound_webhook_static_metadata"] == {"region": "us-east"}
        assert body["dynamic_inbound_routing_enabled"] is True
        assert body["post_call_webhook_url"] == "https://example.com/post-call"
        assert body["post_call_webhook_headers_configured"] is True
        assert body["post_call_webhook_custom_payload_enabled"] is True
        assert body["post_call_webhook_custom_payload_template"] == {
            "callId": "{{call_metadata.call_id}}"
        }
        assert body["status_webhook_enabled"] is True
        assert body["status_webhook_url"] == "https://example.com/status"
        assert body["status_webhook_headers_configured"] is True

        db.refresh(flow)
        assert flow.pre_inbound_webhook_url == "https://example.com/pre-inbound"
        assert flow.pre_inbound_webhook_headers_encrypted is not None
        assert flow.dynamic_inbound_routing_enabled is True

    def test_non_admin_principal_is_forbidden(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal, forbidden=True)

        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD
        )

        assert resp.status_code == 403

    def test_unknown_flow_returns_404(self, db, workspace):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{uuid.uuid4()}/system-webhooks-settings", json=_FULL_PAYLOAD
        )

        assert resp.status_code == 404

    def test_flow_from_other_tenant_returns_404(self, db, flow):
        from app.models.tenant import Tenant

        other_tenant = Tenant(
            name=f"OtherSysWebhooksWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"other_sys_webhooks_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(other_tenant)
        db.commit()
        db.refresh(other_tenant)

        principal = _admin_principal(other_tenant.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD
        )

        assert resp.status_code == 404
        db.refresh(flow)
        assert flow.pre_inbound_webhook_url is None

    def test_ssrf_blocked_url_rejected_at_save_time(self, db, workspace, flow):
        from app.utils.ssrf import SSRFBlockedError

        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        with patch(
            "app.schemas.call_flow.assert_public_url",
            side_effect=SSRFBlockedError("blocked: private IP"),
        ):
            resp = client.put(
                f"/flows/{flow.id}/system-webhooks-settings",
                json={
                    **_FULL_PAYLOAD,
                    "post_call_webhook_url": "https://169.254.169.254/hook",
                },
            )

        assert resp.status_code == 400

    def test_non_https_url_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings",
            json={**_FULL_PAYLOAD, "status_webhook_url": "http://example.com/status"},
        )

        assert resp.status_code == 400

    def test_header_none_leaves_stored_headers_unchanged(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        first = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD
        )
        assert first.status_code == 200, first.text
        db.refresh(flow)
        stored_ciphertext = flow.pre_inbound_webhook_headers_encrypted
        assert stored_ciphertext is not None

        second_payload = dict(_FULL_PAYLOAD)
        second_payload["pre_inbound_webhook_headers"] = None
        second_payload["pre_inbound_webhook_url"] = "https://example.com/pre-inbound-v2"
        second = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=second_payload
        )
        assert second.status_code == 200, second.text

        db.refresh(flow)
        assert flow.pre_inbound_webhook_url == "https://example.com/pre-inbound-v2"
        assert flow.pre_inbound_webhook_headers_encrypted == stored_ciphertext
        # Response still reports headers as configured, even though we didn't
        # resend them on this PUT.
        assert second.json()["pre_inbound_webhook_headers_configured"] is True

    def test_header_explicit_empty_dict_clears_stored_headers(
        self, db, workspace, flow
    ):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        client.put(f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD)
        db.refresh(flow)
        assert flow.pre_inbound_webhook_headers_encrypted is not None

        cleared_payload = dict(_FULL_PAYLOAD)
        cleared_payload["pre_inbound_webhook_headers"] = {}
        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=cleared_payload
        )
        assert resp.status_code == 200, resp.text

        db.refresh(flow)
        assert flow.pre_inbound_webhook_headers_encrypted is None
        assert resp.json()["pre_inbound_webhook_headers_configured"] is False

    def test_header_dict_replaces_stored_headers(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        client.put(f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD)
        db.refresh(flow)
        first_ciphertext = flow.pre_inbound_webhook_headers_encrypted

        replaced_payload = dict(_FULL_PAYLOAD)
        replaced_payload["pre_inbound_webhook_headers"] = {"X-New-Header": "new-value"}
        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=replaced_payload
        )
        assert resp.status_code == 200, resp.text

        db.refresh(flow)
        assert flow.pre_inbound_webhook_headers_encrypted is not None
        assert flow.pre_inbound_webhook_headers_encrypted != first_ciphertext

    def test_full_replace_semantics_for_url_toggle_query_param_fields(
        self, db, workspace, flow
    ):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        client.put(f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD)

        # Omitting query params / toggles on the second PUT reverts them to
        # schema defaults (full-replace, not merge) — unlike headers.
        minimal_payload = {
            "pre_inbound_webhook_url": None,
            "pre_inbound_webhook_headers": None,
            "pre_inbound_webhook_query_params": None,
            "pre_inbound_webhook_static_metadata": None,
            "dynamic_inbound_routing_enabled": False,
            "post_call_webhook_url": None,
            "post_call_webhook_headers": None,
            "post_call_webhook_query_params": None,
            "post_call_webhook_custom_payload_enabled": False,
            "post_call_webhook_custom_payload_template": None,
            "status_webhook_enabled": False,
            "status_webhook_url": None,
            "status_webhook_headers": None,
            "status_webhook_query_params": None,
        }
        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings", json=minimal_payload
        )
        assert resp.status_code == 200, resp.text

        db.refresh(flow)
        assert flow.pre_inbound_webhook_url is None
        assert flow.pre_inbound_webhook_query_params == []
        assert flow.dynamic_inbound_routing_enabled is False
        assert flow.post_call_webhook_url is None
        assert flow.status_webhook_enabled is False
        # Headers untouched by the full-replace (None => unchanged, still set
        # from the first PUT).
        assert flow.pre_inbound_webhook_headers_encrypted is not None

    def test_extra_fields_rejected(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.put(
            f"/flows/{flow.id}/system-webhooks-settings",
            json={**_FULL_PAYLOAD, "unexpected_field": "nope"},
        )
        assert resp.status_code == 400

    def test_update_fires_audit_event_excluding_secret_values(
        self, db, workspace, flow
    ):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        with patch("app.api.v2.routers.flows.log_audit_event") as mock_log_audit:
            resp = client.put(
                f"/flows/{flow.id}/system-webhooks-settings", json=_FULL_PAYLOAD
            )

        assert resp.status_code == 200, resp.text
        mock_log_audit.assert_called_once()
        kwargs = mock_log_audit.call_args.kwargs
        assert kwargs["action"] == "system_webhooks_settings.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == flow.id
        assert kwargs["actor_user_id"] == principal.id

        new_value = kwargs["new_value"]
        # URLs, toggles, header-presence booleans, query-param KEYS: present.
        assert new_value["pre_inbound_webhook_url"] == "https://example.com/pre-inbound"
        assert new_value["pre_inbound_webhook_headers_configured"] is True
        assert new_value["pre_inbound_webhook_query_param_keys"] == ["src"]
        assert new_value["dynamic_inbound_routing_enabled"] is True
        assert new_value["post_call_webhook_headers_configured"] is True
        assert new_value["status_webhook_headers_configured"] is True

        # Header VALUES and query-param VALUES must never appear anywhere in
        # the logged payload — serialize and scan for the secret substrings.
        import json as _json

        serialized = _json.dumps(new_value)
        assert "abc123" not in serialized
        assert "Bearer xyz" not in serialized
        assert "twilio" not in serialized
        assert "prod" not in serialized
        assert "tok" not in serialized
        # And no raw headers dict / query-params list structure at all.
        assert "pre_inbound_webhook_headers" not in new_value
        assert "post_call_webhook_headers" not in new_value
        assert "status_webhook_headers" not in new_value


@pytest.mark.usefixtures("db")
class TestSystemWebhookTestEndpoint:
    def test_returns_test_result_shape_and_is_admin_gated(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        fake_log = MagicMock()
        fake_log.status = "success"
        fake_log.status_code = 200
        fake_log.response_body = '{"variables": {}}'
        fake_log.error = None
        fake_log.duration_ms = 42

        with patch(
            "app.api.v2.routers.flows.run_webhook_test",
            new=AsyncMock(return_value=fake_log),
        ) as mock_run:
            resp = client.post(
                f"/flows/{flow.id}/system-webhooks/test",
                json={"webhook_kind": "pre_inbound"},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {
            "status": "success",
            "status_code": 200,
            "response_body": '{"variables": {}}',
            "error": None,
            "duration_ms": 42,
        }
        mock_run.assert_awaited_once()
        call_args = mock_run.call_args.args
        assert call_args[1] == flow.id
        assert call_args[2] == workspace.id
        assert call_args[3] == "pre_inbound"

    def test_non_admin_principal_is_forbidden(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal, forbidden=True)

        resp = client.post(
            f"/flows/{flow.id}/system-webhooks/test",
            json={"webhook_kind": "post_call"},
        )

        assert resp.status_code == 403

    def test_writes_a_delivery_log_row_end_to_end(self, db, workspace, agent):
        """Not mocking run_webhook_test itself here — exercises the real
        service function with an SSRF-bypassed httpx mock, confirming a
        SystemWebhookDeliveryLog row is actually persisted."""
        from app.models.call_flow import CallFlow
        from app.models.system_webhook_log import SystemWebhookDeliveryLog

        flow = CallFlow(
            tenant_id=workspace.id,
            agent_id=agent.id,
            name="Test-Delivery Flow",
            direction="inbound",
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status-hook",
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)

        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"

        with (
            patch(
                "app.services.system_webhook_service.assert_public_url",
                return_value=None,
            ),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)),
        ):
            resp = client.post(
                f"/flows/{flow.id}/system-webhooks/test",
                json={"webhook_kind": "status"},
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "success"

        logs = (
            db.query(SystemWebhookDeliveryLog)
            .filter(SystemWebhookDeliveryLog.call_flow_id == flow.id)
            .all()
        )
        assert len(logs) == 1
        assert logs[0].webhook_kind == "status"
        assert logs[0].status == "success"

    def test_unconfigured_webhook_kind_returns_400(self, db, workspace, flow):
        principal = _admin_principal(workspace.id)
        client = _build_app(db, principal)

        resp = client.post(
            f"/flows/{flow.id}/system-webhooks/test",
            json={"webhook_kind": "post_call"},
        )

        assert resp.status_code == 400
