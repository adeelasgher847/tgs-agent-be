"""Tests for the A/B Prompt Testing system.

Coverage:
  1. Random variant assignment matches split_ratio within 5% tolerance over 1000 calls
  2. No assignment when A/B testing is disabled or prompts are missing
  3. PUT /api/v2/flows/{id}/ab-test validates prompt ownership and split_ratio bounds
  4. GET /api/v2/flows/{id}/ab-results computes calls/completed/failed/avg_duration/
     transfer_rate/success_rate correctly per variant
  5. Statistical significance guardrail: <30 calls on either variant -> inconclusive
  6. Statistical significance: clear separation -> significant + correct recommended variant
  7. PUT /api/v2/flows/{id}/ab-test/winner promotes current_prompt_id and disables the test
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.user import User
from app.services.ab_testing_service import ab_testing_service

_API_KEY = "test-ab-testing-key"


# ─────────────────────────────────────────────────────────────── helpers ──


def _payload_for(tenant: Tenant) -> dict:
    return {
        "api_key_id": str(uuid.uuid4()),
        "tenant_id": str(tenant.id),
        "key_is_active": True,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": tenant.schema_name,
            "status": "active",
            "credits": 0.0,
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
        },
    }


def _headers(tenant: Tenant) -> dict:
    return {"x-api-key": _API_KEY, "x-workspace-id": str(tenant.id)}


@pytest.fixture
def auth_tenant(db) -> Tenant:
    t = Tenant(
        name=f"AbWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"ab_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def test_agent(db, auth_tenant: Tenant) -> Agent:
    a = Agent(
        tenant_id=auth_tenant.id,
        name="AB Test Agent",
        status="active",
        llm_model="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        tts_voice_external_id="voice-x",
        tts_language="en",
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    print("DEBUG AGENT IN FIXTURE:", a)
    print("DEBUG SESSION ID IN FIXTURE:", id(db))
    print("DEBUG ALL AGENTS IN FIXTURE DB:", db.query(Agent).all())
    return a


@pytest.fixture
def test_user(db, auth_tenant: Tenant) -> User:
    u = User(
        email=f"ab-user-{uuid.uuid4().hex[:6]}@example.com",
        first_name="AB",
        last_name="User",
        hashed_password="",
        current_tenant_id=auth_tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def authed_client(client, auth_tenant: Tenant):
    payload = _payload_for(auth_tenant)

    async def _resolve(_key_hash, _workspace_id):
        return payload

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield client


def _create_flow_with_two_prompts(client, tenant, agent) -> dict:
    created = client.post(
        "/api/v1/call-flows/",
        json={
            "name": "AB Flow",
            "direction": "outbound",
            "agentId": str(agent.id),
            "prompt": "Prompt version A",
        },
        headers=_headers(tenant),
    ).json()
    prompt_a_id = created["currentPromptId"]

    updated = client.put(
        f"/api/v1/call-flows/{created['id']}",
        json={"prompt": "Prompt version B"},
        headers=_headers(tenant),
    ).json()
    prompt_b_id = updated["currentPromptId"]

    return {"flow_id": created["id"], "prompt_a_id": prompt_a_id, "prompt_b_id": prompt_b_id}


def _make_call_session(
    db,
    *,
    tenant: Tenant,
    agent: Agent,
    user: User,
    flow_id: uuid.UUID,
    variant: str,
    status: str = "completed",
    duration: int = None,
    transferred: bool = False,
    success_evaluation: str = None,
) -> CallSession:
    session = CallSession(
        user_id=user.id,
        agent_id=agent.id,
        tenant_id=tenant.id,
        call_flow_id=flow_id,
        ab_variant=variant,
        status=status,
        duration=duration,
        transferred=transferred,
        success_evaluation=success_evaluation,
        start_time=datetime.utcnow(),
    )
    db.add(session)
    db.commit()
    return session


# ──────────────────────────────────────────────────────────────── tests ──


class TestRandomAssignment:
    def _flow_stub(self, *, split_ratio: float = 0.5) -> MagicMock:
        flow = MagicMock(spec=CallFlow)
        flow.ab_test_enabled = True
        flow.ab_prompt_a_id = uuid.uuid4()
        flow.ab_prompt_b_id = uuid.uuid4()
        flow.ab_split_ratio = split_ratio
        return flow

    @pytest.mark.parametrize("split_ratio", [0.5, 0.3, 0.7])
    def test_split_matches_ratio_within_tolerance(self, split_ratio):
        flow = self._flow_stub(split_ratio=split_ratio)
        n = 1000
        a_count = sum(1 for _ in range(n) if ab_testing_service.pick_variant(flow) == "a")
        observed_ratio = a_count / n
        assert abs(observed_ratio - split_ratio) < 0.05

    def test_disabled_flow_returns_none(self):
        flow = self._flow_stub()
        flow.ab_test_enabled = False
        assert ab_testing_service.pick_variant(flow) is None

    def test_missing_prompt_ids_returns_none(self):
        flow = self._flow_stub()
        flow.ab_prompt_b_id = None
        assert ab_testing_service.pick_variant(flow) is None


@pytest.mark.usefixtures("db")
class TestAbTestConfigEndpoint:
    def test_enable_ab_test_returns_updated_config(
        self, authed_client, auth_tenant, test_agent
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)

        resp = authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test",
            json={
                "enabled": True,
                "prompt_a_id": flow["prompt_a_id"],
                "prompt_b_id": flow["prompt_b_id"],
                "split_ratio": 0.4,
            },
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ab_test_enabled"] is True
        assert body["ab_prompt_a_id"] == flow["prompt_a_id"]
        assert body["ab_prompt_b_id"] == flow["prompt_b_id"]
        assert body["ab_split_ratio"] == pytest.approx(0.4)

    def test_split_ratio_out_of_bounds_returns_400(
        self, authed_client, auth_tenant, test_agent
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        resp = authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test",
            json={
                "enabled": True,
                "prompt_a_id": flow["prompt_a_id"],
                "prompt_b_id": flow["prompt_b_id"],
                "split_ratio": 0.95,
            },
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_prompt_not_belonging_to_flow_returns_400(
        self, authed_client, auth_tenant, test_agent
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        other_flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)

        resp = authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test",
            json={
                "enabled": True,
                "prompt_a_id": other_flow["prompt_a_id"],
                "prompt_b_id": flow["prompt_b_id"],
                "split_ratio": 0.5,
            },
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_same_prompts_for_both_variants_returns_400(
        self, authed_client, auth_tenant, test_agent
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        resp = authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test",
            json={
                "enabled": True,
                "prompt_a_id": flow["prompt_a_id"],
                "prompt_b_id": flow["prompt_a_id"],
                "split_ratio": 0.5,
            },
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400
        assert "must be different prompt versions" in resp.text


@pytest.mark.usefixtures("db")
class TestAbResults:
    def test_metrics_calculated_correctly(
        self, authed_client, auth_tenant, test_agent, test_user, db
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        flow_id = uuid.UUID(flow["flow_id"])

        # Variant A: 2 completed (100s, 200s), 1 failed, 1 transferred, 1 success
        _make_call_session(
            db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
            variant="a", status="completed", duration=100, success_evaluation="success",
        )
        _make_call_session(
            db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
            variant="a", status="completed", duration=200, transferred=True,
        )
        _make_call_session(
            db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
            variant="a", status="failed", duration=None,
        )

        # Variant B: 1 completed (50s), 1 failed
        _make_call_session(
            db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
            variant="b", status="completed", duration=50, success_evaluation="success",
        )
        _make_call_session(
            db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
            variant="b", status="failed",
        )

        resp = authed_client.get(
            f"/api/v2/flows/{flow['flow_id']}/ab-results",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        va = body["variant_a"]
        assert va["calls"] == 3
        assert va["completed"] == 2
        assert va["failed"] == 1
        assert va["avg_duration"] == pytest.approx(150.0)
        assert va["transfer_rate"] == pytest.approx(1 / 3)
        assert va["success_rate"] == pytest.approx(1 / 3)

        vb = body["variant_b"]
        assert vb["calls"] == 2
        assert vb["completed"] == 1
        assert vb["failed"] == 1
        assert vb["avg_duration"] == pytest.approx(50.0)
        assert vb["transfer_rate"] == pytest.approx(0.0)
        assert vb["success_rate"] == pytest.approx(0.5)

        # Below the 30-call guardrail -> always inconclusive
        assert body["statistical_significance"] is False
        assert body["recommended_variant"] == "inconclusive"

    def test_guardrail_below_30_calls_is_inconclusive(
        self, authed_client, auth_tenant, test_agent, test_user, db
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        flow_id = uuid.UUID(flow["flow_id"])

        # 29 calls each, all completed (huge rate difference would otherwise be significant)
        for _ in range(29):
            _make_call_session(
                db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
                variant="a", status="completed", duration=10,
            )
        for _ in range(29):
            _make_call_session(
                db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
                variant="b", status="failed",
            )

        resp = authed_client.get(
            f"/api/v2/flows/{flow['flow_id']}/ab-results",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["statistical_significance"] is False
        assert body["recommended_variant"] == "inconclusive"

    def test_significant_difference_recommends_higher_completion_variant(
        self, authed_client, auth_tenant, test_agent, test_user, db
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        flow_id = uuid.UUID(flow["flow_id"])

        # Variant A: 40/50 completed
        for i in range(50):
            _make_call_session(
                db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
                variant="a", status="completed" if i < 40 else "failed", duration=100,
            )
        # Variant B: 20/50 completed
        for i in range(50):
            _make_call_session(
                db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
                variant="b", status="completed" if i < 20 else "failed", duration=100,
            )

        resp = authed_client.get(
            f"/api/v2/flows/{flow['flow_id']}/ab-results",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["statistical_significance"] is True
        assert body["recommended_variant"] == "a"

    def test_no_significant_difference_is_inconclusive(
        self, authed_client, auth_tenant, test_agent, test_user, db
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        flow_id = uuid.UUID(flow["flow_id"])

        # Variant A: 26/50 completed, Variant B: 24/50 completed -> not significant
        for i in range(50):
            _make_call_session(
                db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
                variant="a", status="completed" if i < 26 else "failed", duration=100,
            )
        for i in range(50):
            _make_call_session(
                db, tenant=auth_tenant, agent=test_agent, user=test_user, flow_id=flow_id,
                variant="b", status="completed" if i < 24 else "failed", duration=100,
            )

        resp = authed_client.get(
            f"/api/v2/flows/{flow['flow_id']}/ab-results",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["statistical_significance"] is False
        assert body["recommended_variant"] == "inconclusive"


@pytest.mark.usefixtures("db")
class TestWinnerPromotion:
    def test_promote_winner_updates_current_prompt_and_disables_test(
        self, authed_client, auth_tenant, test_agent, db
    ):
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test",
            json={
                "enabled": True,
                "prompt_a_id": flow["prompt_a_id"],
                "prompt_b_id": flow["prompt_b_id"],
                "split_ratio": 0.5,
            },
            headers=_headers(auth_tenant),
        )

        resp = authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test/winner",
            json={"variant": "b"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["currentPromptId"] == flow["prompt_b_id"]

        row = db.query(CallFlow).filter(CallFlow.id == uuid.UUID(flow["flow_id"])).first()
        assert row.ab_test_enabled is False
        assert str(row.current_prompt_id) == flow["prompt_b_id"]

    def test_promote_winner_missing_variant_prompt_returns_400(
        self, authed_client, auth_tenant, test_agent, db
    ):
        # Create a flow with only prompt A wired into ab_prompt_a_id, leave ab_prompt_b_id unset
        flow = _create_flow_with_two_prompts(authed_client, auth_tenant, test_agent)
        authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test",
            json={
                "enabled": True,
                "prompt_a_id": flow["prompt_a_id"],
                "prompt_b_id": flow["prompt_b_id"],
                "split_ratio": 0.5,
            },
            headers=_headers(auth_tenant),
        )
        # Manually clear ab_prompt_b_id to simulate an incomplete config
        row = db.query(CallFlow).filter(CallFlow.id == uuid.UUID(flow["flow_id"])).first()
        row.ab_prompt_b_id = None
        db.commit()

        resp = authed_client.put(
            f"/api/v2/flows/{flow['flow_id']}/ab-test/winner",
            json={"variant": "b"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400
