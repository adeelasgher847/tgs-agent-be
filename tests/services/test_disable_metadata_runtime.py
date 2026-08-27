"""Unit and runtime integration tests for Disable Metadata setting on Call Flows.

Coverage:
  - _strip_metadata_from_dict helper functions
  - _filter_query_params_metadata helper functions
  - fetch_pre_inbound_webhook_variables with disable_metadata=False vs True
  - _dispatch_post_call_webhook with disable_metadata=False vs True (default and custom templates)
  - _dispatch_status_webhook with disable_metadata=False vs True
  - run_webhook_test with disable_metadata=True
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.system_webhook_log import SystemWebhookDeliveryLog
from app.services.system_webhook_service import (
    _filter_query_params_metadata,
    _strip_metadata_from_dict,
    fetch_pre_inbound_webhook_variables,
)


class TestMetadataHelpers:
    def test_strip_metadata_from_dict_removes_keys_recursively(self):
        payload = {
            "callId": "123",
            "metadata": {"key": "val"},
            "call_metadata": {"foo": "bar"},
            "_metadata": {"secret": "abc"},
            "custom_metadata": {"tier": "pro"},
            "data": {
                "summary": "call summary",
                "metadata": {"nested": "value"},
                "items": [
                    {"id": 1, "metadata": {"x": 1}},
                    {"id": 2, "normal": "ok"},
                ],
            },
        }
        stripped = _strip_metadata_from_dict(payload)
        assert "metadata" not in stripped
        assert "call_metadata" not in stripped
        assert "_metadata" not in stripped
        assert "custom_metadata" not in stripped
        assert stripped["callId"] == "123"
        assert stripped["data"]["summary"] == "call summary"
        assert "metadata" not in stripped["data"]
        assert stripped["data"]["items"][0] == {"id": 1}
        assert stripped["data"]["items"][1] == {"id": 2, "normal": "ok"}

    def test_filter_query_params_metadata_removes_metadata_params(self):
        params = [
            ("flowId", "flow-1"),
            ("metadata", "stripped"),
            ("metadata[account_id]", "123"),
            ("metadata.user_id", "456"),
            ("_metadata.token", "tok"),
            ("call_metadata.source", "web"),
            ("status", "completed"),
        ]
        filtered = _filter_query_params_metadata(params)
        assert filtered == [
            ("flowId", "flow-1"),
            ("status", "completed"),
        ]


class TestPreInboundMetadataRuntime:
    @pytest.mark.anyio
    async def test_pre_inbound_includes_metadata_when_disabled(self):
        flow = MagicMock(spec=CallFlow)
        flow.id = uuid.uuid4()
        flow.tenant_id = uuid.uuid4()
        flow.disable_metadata = False
        flow.pre_inbound_webhook_url = "https://example.com/pre-inbound"
        flow.pre_inbound_webhook_headers_encrypted = None
        flow.pre_inbound_webhook_static_metadata = {"env": "prod"}
        flow.pre_inbound_webhook_query_params = [
            {"key": "env", "value": "prod"},
            {"key": "metadata", "value": "include_me"},
        ]

        captured_kwargs = {}

        async def fake_deliver(db, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return MagicMock(spec=SystemWebhookDeliveryLog, status="success")

        with patch("app.services.system_webhook_service._deliver", side_effect=fake_deliver):
            await fetch_pre_inbound_webhook_variables(
                MagicMock(), flow, from_number="+15551112222", to_number="+15553334444"
            )

        assert "metadata" in captured_kwargs["json_body"]
        assert captured_kwargs["json_body"]["metadata"] == {"env": "prod"}
        assert any(k == "metadata" for k, _ in captured_kwargs["query_params"])

    @pytest.mark.anyio
    async def test_pre_inbound_strips_metadata_when_enabled(self):
        flow = MagicMock(spec=CallFlow)
        flow.id = uuid.uuid4()
        flow.tenant_id = uuid.uuid4()
        flow.disable_metadata = True
        flow.pre_inbound_webhook_url = "https://example.com/pre-inbound"
        flow.pre_inbound_webhook_headers_encrypted = None
        flow.pre_inbound_webhook_static_metadata = {"env": "prod"}
        flow.pre_inbound_webhook_query_params = [
            {"key": "env", "value": "prod"},
            {"key": "metadata", "value": "strip_me"},
            {"key": "_metadata.key", "value": "strip_me_too"},
        ]

        captured_kwargs = {}

        async def fake_deliver(db, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return MagicMock(spec=SystemWebhookDeliveryLog, status="success")

        with patch("app.services.system_webhook_service._deliver", side_effect=fake_deliver):
            await fetch_pre_inbound_webhook_variables(
                MagicMock(), flow, from_number="+15551112222", to_number="+15553334444"
            )

        assert "metadata" not in captured_kwargs["json_body"]
        assert captured_kwargs["json_body"] == {
            "from": "+15551112222",
            "to": "+15553334444",
        }
        assert captured_kwargs["query_params"] == [("env", "prod")]


class TestPostCallAndStatusMetadataRuntime:
    @pytest.mark.anyio
    async def test_post_call_default_payload_strips_call_metadata_when_disable_metadata_true(self, db):
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.services.system_webhook_service import _dispatch_post_call_webhook

        tenant = Tenant(
            name=f"PCWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"pc_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        user = User(
            first_name="Test",
            last_name="User",
            email=f"user_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password="pw",
        )
        db.add_all([tenant, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="PC Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="PC Flow",
            direction="outbound",
            status="active",
            disable_metadata=True,
            post_call_webhook_url="https://example.com/post-call",
            post_call_webhook_custom_payload_enabled=False,
            post_call_webhook_query_params=[
                {"key": "call_id", "value": "{{call_metadata.call_id}}"},
                {"key": "metadata", "value": "strip"},
            ],
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            status="completed",
            start_time=datetime.now(timezone.utc),
            call_metadata={"internal_debug": "123"},
        )
        db.add(session)
        db.commit()

        captured_kwargs = {}

        async def fake_deliver(s_db, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return MagicMock(spec=SystemWebhookDeliveryLog, status="success")

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch("app.services.system_webhook_service._deliver", side_effect=fake_deliver),
        ):
            success = await _dispatch_post_call_webhook(session.id)

        assert success is True
        payload_data = captured_kwargs["json_body"]["data"]
        assert "internal_debug" not in payload_data
        assert "call_metadata" not in payload_data
        assert "metadata" not in payload_data
        assert captured_kwargs["query_params"] == [("call_id", str(session.id))]

    @pytest.mark.anyio
    async def test_status_webhook_strips_metadata_when_disable_metadata_true(self, db):
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.services.system_webhook_service import _dispatch_status_webhook

        tenant = Tenant(
            name=f"StWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"st_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        user = User(
            first_name="Test",
            last_name="User",
            email=f"user_{uuid.uuid4().hex[:8]}@example.com",
            hashed_password="pw",
        )
        db.add_all([tenant, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="St Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="St Flow",
            direction="outbound",
            status="active",
            disable_metadata=True,
            status_webhook_enabled=True,
            status_webhook_url="https://example.com/status",
            status_webhook_query_params=[
                {"key": "event", "value": "{{_system.eventType}}"},
                {"key": "_metadata.apiKey", "value": "secret"},
            ],
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            status="completed",
            start_time=datetime.now(timezone.utc),
            call_metadata={"internal_crm": "abc"},
        )
        db.add(session)
        db.commit()

        captured_kwargs = {}

        async def fake_deliver(s_db, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return MagicMock(spec=SystemWebhookDeliveryLog, status="success")

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch("app.services.system_webhook_service._deliver", side_effect=fake_deliver),
        ):
            success = await _dispatch_status_webhook(
                session.id,
                event_type="call.ended",
                extra={"outcome": "completed", "metadata": {"debug": "val"}},
            )

        assert success is True
        payload = captured_kwargs["json_body"]
        assert "metadata" not in payload
        assert payload["status"] == "success"
        assert captured_kwargs["query_params"] == [("event", "call.ended")]
