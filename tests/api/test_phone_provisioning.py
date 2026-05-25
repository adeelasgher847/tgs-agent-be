"""
Integration tests for Sprint 2 phone provisioning endpoints.

Covers:
  GET  /api/v1/phone-numbers/search
  POST /api/v1/phone-numbers/purchase
  POST /api/v1/telephony/external
  POST /api/v1/telephony/bind
  POST /api/v1/telephony/unbind
  GET  /api/v1/phone-numbers (binding status + agent name)

Twilio HTTP calls are mocked at the service-method boundary using unittest.mock.patch
so no real Twilio API is hit and no real numbers are purchased.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.middleware.api_key_middleware import _attach_workspace_context
from app.models.agent import Agent
from app.models.phone_number import NumberConfiguration, PhoneNumber
from app.models.tenant import Tenant
from app.schemas.agent import AgentStatusEnum, agent_to_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_for(tenant: Tenant) -> Dict[str, Any]:
    return {
        "api_key_id": str(uuid.uuid4()),
        "tenant_id": str(tenant.id),
        "key_is_active": True,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": getattr(tenant, "schema_name", "test"),
            "status": "active",
        },
    }


def _headers(tenant: Tenant) -> Dict[str, str]:
    return {"X-API-Key": "test-phone-key", "X-Workspace-ID": str(tenant.id)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def phone_tenant(db):
    t = Tenant(name="Phone Test Tenant", schema_name="phone_test_schema", status="active")
    db.add(t)
    db.commit()
    db.refresh(t)
    yield t
    # cleanup
    db.query(PhoneNumber).filter(PhoneNumber.tenant_id == t.id).delete()
    db.query(Agent).filter(Agent.tenant_id == t.id).delete()
    db.delete(t)
    db.commit()


@pytest.fixture
def phone_agent(db, phone_tenant):
    agent = Agent(
        name="Test Phone Agent",
        tenant_id=phone_tenant.id,
        status="pending",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


@pytest.fixture
def authed_client(client: TestClient, phone_tenant: Tenant):
    payload = _payload_for(phone_tenant)

    async def _resolve(_key_hash, _workspace_id):
        return payload

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield client


# ---------------------------------------------------------------------------
# GET /api/v1/phone-numbers/search
# ---------------------------------------------------------------------------


class TestSearchPhoneNumbers:
    def test_search_returns_available_numbers(self, authed_client: TestClient, phone_tenant: Tenant):
        mock_result = [
            {
                "phone_number": "+61212345678",
                "friendly_name": "+61 2 1234 5678",
                "locality": "Sydney",
                "region": "NSW",
                "country": "AU",
                "capabilities": {"voice": True, "sms": True, "mms": False},
                "beta": False,
            }
        ]
        with patch(
            "app.services.twilio_service.TwilioService.search_available_numbers",
            return_value=mock_result,
        ):
            resp = authed_client.get(
                "/api/v1/phone-numbers/search",
                params={"country": "AU", "type": "local", "areaCode": "02"},
                headers=_headers(phone_tenant),
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["total"] == 1
        assert body["data"]["available_numbers"][0]["phone_number"] == "+61212345678"

    def test_search_twilio_failure_returns_502(self, authed_client: TestClient, phone_tenant: Tenant):
        with patch(
            "app.services.twilio_service.TwilioService.search_available_numbers",
            side_effect=Exception("Twilio down"),
        ):
            resp = authed_client.get(
                "/api/v1/phone-numbers/search",
                params={"country": "AU"},
                headers=_headers(phone_tenant),
            )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /api/v1/phone-numbers/purchase
# ---------------------------------------------------------------------------


class TestPurchasePhoneNumber:
    def _mock_purchase(self, phone_number: str = "+61212345678") -> Dict[str, Any]:
        return {
            "sid": "PN" + "x" * 32,
            "phone_number": phone_number,
            "friendly_name": phone_number,
            "voice_url": "https://example.com/incoming",
            "voice_method": "POST",
            "status_callback": "https://example.com/status",
            "status_callback_method": "POST",
            "capabilities": {"voice": True, "sms": True, "mms": False},
            "date_created": "2026-05-26T00:00:00+00:00",
            "date_updated": "2026-05-26T00:00:00+00:00",
        }

    def test_purchase_saves_to_db(self, authed_client: TestClient, phone_tenant: Tenant, db):
        number = "+61298765432"
        with patch(
            "app.services.twilio_service.TwilioService.purchase_phone_number",
            return_value=self._mock_purchase(number),
        ):
            resp = authed_client.post(
                "/api/v1/phone-numbers/purchase",
                json={"phone_number": number, "label": "Main line"},
                headers=_headers(phone_tenant),
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["phone_number"] == number
        assert body["data"]["provider"] == "twilio"
        assert body["data"]["twilio_sid"] is not None

        # Verify persisted in DB
        pn = db.query(PhoneNumber).filter(PhoneNumber.phone_number == number).first()
        assert pn is not None
        assert str(pn.tenant_id) == str(phone_tenant.id)
        config = (
            db.query(NumberConfiguration)
            .filter(NumberConfiguration.phone_number_id == pn.id)
            .first()
        )
        assert config is not None
        assert config.recording_enabled is False
        assert config.max_duration_seconds == 3600

        db.query(NumberConfiguration).filter(
            NumberConfiguration.phone_number_id == pn.id
        ).delete()
        db.delete(pn)
        db.commit()

    def test_purchase_invalid_e164_returns_4xx(self, authed_client: TestClient, phone_tenant: Tenant):
        resp = authed_client.post(
            "/api/v1/phone-numbers/purchase",
            json={"phone_number": "0412345678"},  # missing +
            headers=_headers(phone_tenant),
        )
        # App global handler maps Pydantic validation errors → 400
        assert resp.status_code in (400, 422)

    def test_purchase_duplicate_returns_409(self, authed_client: TestClient, phone_tenant: Tenant, db):
        number = "+61299887766"
        # Pre-insert the number
        pn = PhoneNumber(
            phone_number=number,
            tenant_id=phone_tenant.id,
            provider="twilio",
            status="active",
        )
        db.add(pn)
        db.commit()

        with patch(
            "app.services.twilio_service.TwilioService.purchase_phone_number",
            return_value=self._mock_purchase(number),
        ):
            resp = authed_client.post(
                "/api/v1/phone-numbers/purchase",
                json={"phone_number": number},
                headers=_headers(phone_tenant),
            )
        assert resp.status_code == 409

        # cleanup
        db.delete(pn)
        db.commit()


# ---------------------------------------------------------------------------
# POST /api/v1/telephony/external
# ---------------------------------------------------------------------------


class TestRegisterExternalNumber:
    def test_register_external_creates_provider_external(
        self, authed_client: TestClient, phone_tenant: Tenant, db
    ):
        number = "+61411222333"
        resp = authed_client.post(
            "/api/v1/telephony/external",
            json={
                "phone_number": number,
                "label": "BYO SIP line",
                "sip_username": "sip_user@example.com",
                "sip_password": "supersecret",
            },
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["provider"] == "external"
        assert body["data"]["phone_number"] == number

        pn = db.query(PhoneNumber).filter(PhoneNumber.phone_number == number).first()
        assert pn is not None
        assert pn.provider == "external"
        assert pn.sip_username == "sip_user@example.com"
        assert pn.sip_password is not None  # encrypted, not plaintext

        # cleanup
        db.delete(pn)
        db.commit()

    def test_register_external_invalid_e164(
        self, authed_client: TestClient, phone_tenant: Tenant
    ):
        resp = authed_client.post(
            "/api/v1/telephony/external",
            json={
                "phone_number": "04123",
                "sip_username": "u",
                "sip_password": "p",
            },
            headers=_headers(phone_tenant),
        )
        # App global handler maps Pydantic validation errors → 400
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# GET /api/v1/telephony/bindings
# ---------------------------------------------------------------------------


class TestListBoundBindings:
    def test_list_bound_agents_returns_agent_id(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent, db
    ):
        pn = PhoneNumber(
            phone_number="+61400000088",
            tenant_id=phone_tenant.id,
            provider="twilio",
            status="active",
            assistant_id=phone_agent.id,
        )
        db.add(pn)
        db.commit()

        resp = authed_client.get(
            "/api/v1/telephony/bindings",
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 200, resp.text
        bindings = resp.json()["data"]["bindings"]
        match = [b for b in bindings if b["agent_id"] == str(phone_agent.id)]
        assert len(match) == 1
        assert match[0]["number_id"] == str(pn.id)
        assert match[0]["phone_number"] == "+61400000088"

        db.delete(pn)
        db.commit()


# ---------------------------------------------------------------------------
# POST /api/v1/telephony/bind and /unbind
# ---------------------------------------------------------------------------


class TestBindUnbind:
    def _create_number(self, db, tenant_id, phone: str = "+61400000001") -> PhoneNumber:
        pn = PhoneNumber(
            phone_number=phone,
            tenant_id=tenant_id,
            provider="twilio",
            status="active",
        )
        db.add(pn)
        db.commit()
        db.refresh(pn)
        return pn

    def test_bind_sets_agent_ready(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent, db
    ):
        pn = self._create_number(db, phone_tenant.id, "+61400000002")
        resp = authed_client.post(
            "/api/v1/telephony/bind",
            json={"number_id": str(pn.id), "agent_id": str(phone_agent.id)},
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["agent_id"] == str(phone_agent.id)
        assert body["data"]["agent_status"] == "ready"

        db.refresh(pn)
        db.refresh(phone_agent)
        assert pn.assistant_id == phone_agent.id
        assert phone_agent.status == "ready"
        assert agent_to_out(phone_agent).status == AgentStatusEnum.ready

        # cleanup
        pn.assistant_id = None
        db.commit()
        db.delete(pn)
        db.commit()

    def test_bind_duplicate_returns_409(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent, db
    ):
        pn = self._create_number(db, phone_tenant.id, "+61400000003")
        pn.assistant_id = phone_agent.id
        db.commit()

        resp = authed_client.post(
            "/api/v1/telephony/bind",
            json={"number_id": str(pn.id), "agent_id": str(phone_agent.id)},
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 409

        # cleanup
        pn.assistant_id = None
        db.commit()
        db.delete(pn)
        db.commit()

    def test_unbind_clears_binding_and_sets_pending(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent, db
    ):
        pn = self._create_number(db, phone_tenant.id, "+61400000004")
        pn.assistant_id = phone_agent.id
        phone_agent.status = "ready"
        db.commit()

        resp = authed_client.post(
            "/api/v1/telephony/unbind",
            json={"number_id": str(pn.id)},
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["agent_id"] is None
        assert body["data"]["agent_status"] == "pending"

        db.refresh(pn)
        db.refresh(phone_agent)
        assert pn.assistant_id is None
        assert phone_agent.status == "pending"

        db.delete(pn)
        db.commit()

    def test_unbind_unbound_number_returns_409(
        self, authed_client: TestClient, phone_tenant: Tenant, db
    ):
        pn = self._create_number(db, phone_tenant.id, "+61400000005")

        resp = authed_client.post(
            "/api/v1/telephony/unbind",
            json={"number_id": str(pn.id)},
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 409

        db.delete(pn)
        db.commit()

    def test_bind_unknown_number_returns_404(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent
    ):
        resp = authed_client.post(
            "/api/v1/telephony/bind",
            json={"number_id": str(uuid.uuid4()), "agent_id": str(phone_agent.id)},
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 404

    def test_bind_accepts_ticket_camel_case_aliases(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent, db
    ):
        pn = self._create_number(db, phone_tenant.id, "+61400000010")
        resp = authed_client.post(
            "/api/v1/telephony/bind",
            json={"numberId": str(pn.id), "agentId": str(phone_agent.id)},
            headers=_headers(phone_tenant),
        )
        assert resp.status_code == 200, resp.text
        db.refresh(pn)
        assert pn.assistant_id == phone_agent.id
        pn.assistant_id = None
        db.commit()
        db.delete(pn)
        db.commit()


# ---------------------------------------------------------------------------
# GET /api/v1/phone-numbers (binding status + agent name)
# ---------------------------------------------------------------------------


class TestListPhoneNumbers:
    def test_list_includes_agent_name(
        self, authed_client: TestClient, phone_tenant: Tenant, phone_agent: Agent, db
    ):
        pn = PhoneNumber(
            phone_number="+61400000099",
            tenant_id=phone_tenant.id,
            provider="twilio",
            status="active",
            assistant_id=phone_agent.id,
        )
        db.add(pn)
        db.commit()

        resp = authed_client.get("/api/v1/phone-numbers/", headers=_headers(phone_tenant))
        assert resp.status_code == 200, resp.text
        items = resp.json()["data"]["phone_numbers"]
        match = [i for i in items if i["phone_number"] == "+61400000099"]
        assert len(match) == 1
        assert match[0]["agent_id"] == str(phone_agent.id)
        assert match[0]["agent_name"] == phone_agent.name
        assert match[0]["binding_status"] == "bound"

        db.delete(pn)
        db.commit()
