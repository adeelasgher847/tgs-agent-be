"""Unit and runtime integration tests for Inbound Rules & Number Blocking Engine.

Coverage:
  - Caller matching with digit normalization and national variants (+1, 10-digit, 11-digit)
  - Twilio incoming call rejection (<Reject reason="busy"/>) when caller is denied
  - CallSession recording with ended_reason="Blocked by inbound rule set"
  - Fall-through to regular AI voice agent when caller is not in rule set
  - Fall-through when flow has no rule set or rule set is deleted
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.inbound_rule import InboundRule, InboundRuleSet
from app.models.model import Model
from app.models.phone_number import PhoneNumber
from app.models.provider import Provider
from app.models.tenant import Tenant
from app.models.user import User
from app.routers.voice import router
from app.services.inbound_rules_service import inbound_rules_service


class TestInboundRulesServiceUnit:
    def test_is_number_blocked_exact_and_variants(self, db):
        tenant = Tenant(
            name=f"UnitWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"unit_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(tenant)
        db.commit()

        rs = InboundRuleSet(
            tenant_id=tenant.id,
            name="Blocklist Unit",
        )
        db.add(rs)
        db.commit()

        # Rule added with 10 digits
        r1 = InboundRule(
            tenant_id=tenant.id,
            rule_set_id=rs.id,
            phone_number_pattern="5559876543",
            normalized_digits="5559876543",
            action="deny",
        )
        db.add(r1)
        db.commit()

        # Check with 10 digits
        blocked, rule = inbound_rules_service.is_number_blocked(
            db, tenant.id, rs.id, "5559876543"
        )
        assert blocked is True
        assert rule.id == r1.id

        # Check with +1 E.164
        blocked, rule = inbound_rules_service.is_number_blocked(
            db, tenant.id, rs.id, "+15559876543"
        )
        assert blocked is True

        # Check with formatted string
        blocked, rule = inbound_rules_service.is_number_blocked(
            db, tenant.id, rs.id, "+1 (555) 987-6543"
        )
        assert blocked is True

        # Check unblocked number
        blocked, rule = inbound_rules_service.is_number_blocked(
            db, tenant.id, rs.id, "+15550009999"
        )
        assert blocked is False
        assert rule is None

        # Check international number (+44)
        r_uk = InboundRule(
            tenant_id=tenant.id,
            rule_set_id=rs.id,
            phone_number_pattern="+44 20 7946 0958",
            normalized_digits="442079460958",
            action="deny",
        )
        db.add(r_uk)
        db.commit()

        blocked, rule = inbound_rules_service.is_number_blocked(
            db, tenant.id, rs.id, "+44 20 7946 0958"
        )
        assert blocked is True
        assert rule.id == r_uk.id

        # Check soft-deleted rule is NOT blocked
        r_uk.is_deleted = True
        db.commit()

        blocked, rule = inbound_rules_service.is_number_blocked(
            db, tenant.id, rs.id, "+44 20 7946 0958"
        )
        assert blocked is False
        assert rule is None

        # Check empty or non-digit input
        assert (
            inbound_rules_service.is_number_blocked(
                db, tenant.id, rs.id, ""
            )[0]
            is False
        )
        assert (
            inbound_rules_service.is_number_blocked(
                db, tenant.id, rs.id, "anonymous"
            )[0]
            is False
        )


class TestInboundRulesWebhookRuntime:
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
            name=f"InboundBlock-{uuid.uuid4().hex[:8]}",
            schema_name=f"inbound_block_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        user = User(
            email=f"block_{uuid.uuid4().hex[:6]}@example.com",
            first_name="Block",
            last_name="Tester",
            hashed_password="pw",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        model = (
            db.query(Model).filter(Model.model_name == "gpt-4o-mini").first()
        )
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
            name="Inbound Block Agent",
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

        rule_set = InboundRuleSet(
            tenant_id=tenant.id,
            name="Test Spammers",
        )
        db.add(rule_set)
        db.commit()
        db.refresh(rule_set)

        blocked_rule = InboundRule(
            tenant_id=tenant.id,
            rule_set_id=rule_set.id,
            phone_number_pattern="+1 (555) 999-8888",
            normalized_digits="15559998888",
            label="Spam Robocall",
            action="deny",
        )
        db.add(blocked_rule)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="Inbound Block Flow",
            direction="inbound",
            status="active",
            inbound_rule_set_id=rule_set.id,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)

        return tenant, agent, phone, flow, rule_set

    def test_inbound_call_from_blocked_number_rejected(
        self, db, client, setup_inbound_entities
    ):
        tenant, agent, phone, flow, rule_set = setup_inbound_entities

        with patch("app.routers.voice.settings.ALLOW_UNAUTHENTICATED_WEBHOOKS", True):
            resp = client.post(
                "/incoming",
                data={
                    "CallSid": "CA_BLOCK_TEST_1",
                    "From": "+1 (555) 999-8888",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        assert "<Reject" in resp.text
        assert 'reason="busy"' in resp.text

        # Verify CallSession persistence
        session = (
            db.query(CallSession)
            .filter(CallSession.twilio_call_sid == "CA_BLOCK_TEST_1")
            .first()
        )
        assert session is not None
        assert session.tenant_id == tenant.id
        assert session.agent_id == agent.id
        assert session.call_flow_id == flow.id
        assert session.from_number == "+1 (555) 999-8888"
        assert session.customer_phone_number == "+1 (555) 999-8888"
        assert session.to_number == phone.phone_number
        assert session.assistant_phone_number == phone.phone_number
        assert session.call_type == "inbound"
        assert session.status == "completed"
        assert session.ended_reason == "Blocked by inbound rule set"

    def test_inbound_call_from_unblocked_number_proceeds(
        self, client, setup_inbound_entities
    ):
        _, _, phone, flow, _ = setup_inbound_entities

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
                    "CallSid": "CA_BLOCK_TEST_2",
                    "From": "+1 (555) 111-2222",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "<Reject" not in resp.text
        assert "<Stream" in resp.text

    def test_inbound_call_when_no_ruleset_attached(
        self, db, client, setup_inbound_entities
    ):
        _, _, phone, flow, _ = setup_inbound_entities
        flow.inbound_rule_set_id = None
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
                    "CallSid": "CA_BLOCK_TEST_3",
                    "From": "+1 (555) 999-8888",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "<Reject" not in resp.text
        assert "<Stream" in resp.text

    def test_inbound_call_when_ruleset_deleted_proceeds(
        self, db, client, setup_inbound_entities
    ):
        _, _, phone, flow, rule_set = setup_inbound_entities
        # Soft delete the rule set and its rules
        rule_set.is_deleted = True
        for r in rule_set.rules:
            r.is_deleted = True
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
                    "CallSid": "CA_BLOCK_TEST_4",
                    "From": "+1 (555) 999-8888",
                    "To": phone.phone_number,
                },
            )

        assert resp.status_code == 200
        assert "<Reject" not in resp.text
        assert "<Stream" in resp.text
