"""Unit and runtime integration tests for Inbound Call Redirection with Conditional Rules Engine.

Coverage:
  - Evaluation of redirect conditions with AND logic and operators (exists, not_empty, equals, not_equals)
  - Template token rendering for departure announcements ({{_metadata.key}}, {{key}})
  - Inbound webhook handle_incoming_call runtime execution:
    - Unconditional redirect when enabled (returns <Dial>)
    - Conditional redirect when conditions match (returns <Dial>)
    - Fall-through to regular AI agent when conditions do not match
    - Spoken announcement <Say> before <Dial> when enabled
    - Fall-through when redirection is disabled
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.model import Model
from app.models.phone_number import PhoneNumber
from app.models.provider import Provider
from app.models.tenant import Tenant
from app.models.user import User
from app.routers.voice import router
from app.services.call_flow_service import call_flow_service


class TestEvaluateRedirectConditions:
    def test_empty_conditions_returns_true(self):
        assert (
            call_flow_service.evaluate_redirect_conditions([], {"tier": "vip"})
            is True
        )

    def test_exists_operator(self):
        conditions = [
            {"variable": "tier", "operator": "exists"},
            {"variable": "{{_metadata.company}}", "operator": "exists"},
        ]
        context = {
            "tier": "vip",
            "_metadata": {"company": "Acme"},
        }
        assert (
            call_flow_service.evaluate_redirect_conditions(conditions, context)
            is True
        )

        missing_context = {"tier": "vip"}
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, missing_context
            )
            is False
        )

    def test_not_empty_operator(self):
        conditions = [
            {"variable": "caller_name", "operator": "not_empty"},
        ]
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"caller_name": "Alice"}
            )
            is True
        )
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"caller_name": "   "}
            )
            is False
        )
        assert (
            call_flow_service.evaluate_redirect_conditions(conditions, {})
            is False
        )

    def test_equals_operator_case_insensitive(self):
        conditions = [
            {"variable": "tier", "operator": "equals", "value": "VIP"},
        ]
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"tier": "vip"}
            )
            is True
        )
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"tier": "Standard"}
            )
            is False
        )
        assert (
            call_flow_service.evaluate_redirect_conditions(conditions, {})
            is False
        )

    def test_not_equals_operator(self):
        conditions = [
            {
                "variable": "account_status",
                "operator": "not_equals",
                "value": "suspended",
            },
        ]
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"account_status": "active"}
            )
            is True
        )
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"account_status": "suspended"}
            )
            is False
        )

    def test_and_logic_requires_all_rules_to_pass(self):
        conditions = [
            {"variable": "tier", "operator": "equals", "value": "gold"},
            {"variable": "region", "operator": "equals", "value": "us"},
        ]
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"tier": "gold", "region": "us"}
            )
            is True
        )
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, {"tier": "gold", "region": "eu"}
            )
            is False
        )

    def test_evaluate_conditions_with_pydantic_models_and_enums(self):
        from app.schemas.call_flow import (
            RedirectCondition,
            RedirectConditionOperatorEnum,
        )

        conditions = [
            RedirectCondition(
                variable="tier",
                operator=RedirectConditionOperatorEnum.equals,
                value="VIP",
            ),
            RedirectCondition(
                variable="department",
                operator=RedirectConditionOperatorEnum.not_empty,
                value=None,
            ),
        ]
        context = {
            "tier": "vip",
            "_metadata": {"department": "sales"},
        }
        assert (
            call_flow_service.evaluate_redirect_conditions(conditions, context)
            is True
        )

    def test_evaluate_conditions_multi_dot_nested_path(self):
        conditions = [
            {
                "variable": "{{_metadata.customer.plan}}",
                "operator": "equals",
                "value": "enterprise",
            },
        ]
        context = {
            "_metadata": {
                "customer": {
                    "plan": "Enterprise",
                }
            }
        }
        assert (
            call_flow_service.evaluate_redirect_conditions(conditions, context)
            is True
        )

        mismatch_context = {
            "_metadata": {
                "customer": {
                    "plan": "starter",
                }
            }
        }
        assert (
            call_flow_service.evaluate_redirect_conditions(
                conditions, mismatch_context
            )
            is False
        )


class TestRenderRedirectMessageTemplate:
    def test_render_template_tokens(self):
        template = "Hello {{caller_name}}, forwarding you to {{_metadata.department}} at {{_metadata.company}}."
        context = {
            "caller_name": "Alice",
            "_metadata": {
                "department": "Support",
                "company": "Acme Corp",
            },
        }
        rendered = call_flow_service.render_redirect_message_template(
            template, context
        )
        assert (
            rendered
            == "Hello Alice, forwarding you to Support at Acme Corp."
        )

    def test_unmatched_tokens_replaced_with_empty_string(self):
        template = "Connecting to {{unknown_token}} team."
        context = {"caller_name": "Bob"}
        rendered = call_flow_service.render_redirect_message_template(
            template, context
        )
        assert rendered == "Connecting to  team."

    def test_none_values_rendered_as_empty_string_not_literal_none(self):
        template = "Hello {{caller_name}}, forwarding to {{_metadata.company}}."
        context = {
            "caller_name": None,
            "_metadata": {
                "company": "Acme Corp",
            },
        }
        rendered = call_flow_service.render_redirect_message_template(
            template, context
        )
        assert rendered == "Hello , forwarding to Acme Corp."

    def test_multi_dot_nested_template_token(self):
        template = "Connecting you to {{_metadata.org.team}} department."
        context = {
            "_metadata": {
                "org": {
                    "team": "Billing",
                }
            }
        }
        rendered = call_flow_service.render_redirect_message_template(
            template, context
        )
        assert rendered == "Connecting you to Billing department."

    def test_empty_template_returns_empty_string(self):
        assert (
            call_flow_service.render_redirect_message_template("", {"a": "b"})
            == ""
        )


class TestInboundRedirectionWebhookRuntime:
    @pytest.fixture
    def app(self, db):
        mini = FastAPI()
        mini.include_router(router)
        from app.api.deps import get_db

        mini.dependency_overrides[get_db] = lambda: db
        return mini

    @pytest.fixture
    def client(self, app):
        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def setup_inbound_entities(self, db):
        tenant = Tenant(
            name=f"InboundRedirect-{uuid.uuid4().hex[:8]}",
            schema_name=f"inbound_redir_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        user = User(
            email=f"redir_{uuid.uuid4().hex[:6]}@example.com",
            first_name="Redirect",
            last_name="Tester",
            hashed_password="hashed_pw_123",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        model = db.execute(
            select(Model).where(Model.model_name == "gpt-4o-mini")
        ).scalar_one_or_none()
        if not model:
            provider = Provider(name="openai", is_active=True)
            db.add(provider)
            db.commit()
            model = Model(
                provider_id=provider.id,
                model_name="gpt-4o-mini",
                archive=False,
            )
            db.add(model)
            db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="Inbound Redirection Agent",
            status="active",
            llm_model="gpt-4o-mini",
            model_id=model.id,
            created_by=user.id,
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-1",
            tts_language="en",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)

        phone = PhoneNumber(
            tenant_id=tenant.id,
            phone_number=f"+15550{uuid.uuid4().int % 10000000:07d}",
            status="active",
            assistant_id=agent.id,
        )
        db.add(phone)
        db.commit()
        db.refresh(phone)

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="Inbound Redirection Flow",
            direction="inbound",
            status="active",
            redirect_inbound_calls_enabled=True,
            redirect_forward_phone_number="+14155552671",
            redirect_conditions=[],
            redirect_speak_message_enabled=False,
            redirect_message=None,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)

        return tenant, agent, phone, flow

    def test_unconditional_redirect_returns_dial_twiml(
        self, client, setup_inbound_entities
    ):
        _, _, phone, flow = setup_inbound_entities

        with (
            patch("app.routers.voice.settings.ALLOW_UNAUTHENTICATED_WEBHOOKS", True),
            patch(
                "app.routers.voice.credit_service.has_sufficient_credits",
                return_value=(True, 100, 1, None),
            ),
        ):
            resp = client.post(
                "/incoming",
                data={
                    "CallSid": "CA_REDIRECT_TEST_1",
                    "From": "+12025550199",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        assert "<Dial" in resp.text
        assert flow.redirect_forward_phone_number in resp.text
        assert "<Say" not in resp.text

    def test_conditional_redirect_matches_returns_dial_twiml(
        self, db, client, setup_inbound_entities
    ):
        _, _, phone, flow = setup_inbound_entities
        flow.redirect_conditions = [
            {"variable": "vip_status", "operator": "equals", "value": "true"},
        ]
        flow.redirect_speak_message_enabled = True
        flow.redirect_message = (
            "Connecting our VIP caller to the executive line."
        )
        db.commit()

        mock_webhook_vars = {"vip_status": "true"}

        with (
            patch("app.routers.voice.settings.ALLOW_UNAUTHENTICATED_WEBHOOKS", True),
            patch(
                "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
                new=AsyncMock(return_value=mock_webhook_vars),
            ),
            patch(
                "app.routers.voice.credit_service.has_sufficient_credits",
                return_value=(True, 100, 1, None),
            ),
        ):
            flow.pre_inbound_webhook_url = "https://example.com/pre-inbound"
            db.commit()

            resp = client.post(
                "/incoming",
                data={
                    "CallSid": "CA_REDIRECT_TEST_2",
                    "From": "+12025550199",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "<Say>Connecting our VIP caller to the executive line.</Say>" in resp.text
        assert "<Dial" in resp.text
        assert flow.redirect_forward_phone_number in resp.text

    def test_conditional_redirect_mismatch_falls_through_to_ai_agent(
        self, db, client, setup_inbound_entities
    ):
        _, agent, phone, flow = setup_inbound_entities
        flow.redirect_conditions = [
            {"variable": "vip_status", "operator": "equals", "value": "true"},
        ]
        flow.pre_inbound_webhook_url = "https://example.com/pre-inbound"
        db.commit()

        # Webhook returns standard tier, NOT vip
        mock_webhook_vars = {"vip_status": "false"}

        with (
            patch("app.routers.voice.settings.ALLOW_UNAUTHENTICATED_WEBHOOKS", True),
            patch(
                "app.services.system_webhook_service.fetch_pre_inbound_webhook_variables",
                new=AsyncMock(return_value=mock_webhook_vars),
            ),
            patch(
                "app.routers.voice.credit_service.has_sufficient_credits",
                return_value=(True, 100, 1, None),
            ),
            patch(
                "app.routers.voice.build_streaming_twiml",
                return_value="<Response><Connect><Stream url='wss://example.com'/></Connect></Response>",
            ),
        ):
            resp = client.post(
                "/incoming",
                data={
                    "CallSid": "CA_REDIRECT_TEST_3",
                    "From": "+12025550199",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        # Should not dial forwarding number, but connect to stream
        assert "<Dial" not in resp.text
        assert "<Stream" in resp.text

    def test_disabled_redirection_falls_through_to_ai_agent(
        self, db, client, setup_inbound_entities
    ):
        _, agent, phone, flow = setup_inbound_entities
        flow.redirect_inbound_calls_enabled = False
        db.commit()

        with (
            patch("app.routers.voice.settings.ALLOW_UNAUTHENTICATED_WEBHOOKS", True),
            patch(
                "app.routers.voice.credit_service.has_sufficient_credits",
                return_value=(True, 100, 1, None),
            ),
            patch(
                "app.routers.voice.build_streaming_twiml",
                return_value="<Response><Connect><Stream url='wss://example.com'/></Connect></Response>",
            ),
        ):
            resp = client.post(
                "/incoming",
                data={
                    "CallSid": "CA_REDIRECT_TEST_4",
                    "From": "+12025550199",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "<Dial" not in resp.text
        assert "<Stream" in resp.text

    def test_inbound_redirection_call_session_db_error_resilience(
        self, db, client, setup_inbound_entities
    ):
        _, _, phone, flow = setup_inbound_entities

        with (
            patch("app.routers.voice.settings.ALLOW_UNAUTHENTICATED_WEBHOOKS", True),
            patch(
                "app.routers.voice.credit_service.has_sufficient_credits",
                return_value=(True, 100, 1, None),
            ),
            patch.object(db, "commit", side_effect=Exception("DB connection error")),
        ):
            resp = client.post(
                "/incoming",
                data={
                    "CallSid": "CA_REDIRECT_TEST_DB_FAIL",
                    "From": "+12025550199",
                    "To": phone.phone_number,
                },
            )

        # Call redirection should still succeed even if CallSession persistence encounters an error
        assert resp.status_code == 200
        assert "<Dial" in resp.text
        assert flow.redirect_forward_phone_number in resp.text
