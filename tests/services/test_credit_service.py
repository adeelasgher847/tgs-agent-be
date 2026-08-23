"""Plan-aware credit billing coverage for app.services.credit_service.

Coverage:
  - get_effective_rate_per_minute: PricingConfig row vs. flat default fallback.
  - has_sufficient_credits: plan-aware pre-call gate (remaining included
    minutes covers the call even at zero credits; pay-as-you-go still
    requires positive credits).
  - _finalize_call_credits (the shared free/billable split logic used by
    both the periodic tick and the final flush): a tick fully within the
    plan's remaining included minutes deducts 0 credits; a tick that
    exceeds it is charged only for the overage portion; a tenant with no
    active plan is charged from the first second.
  - _monitor_and_deduct_credits: a tick fully covered by the plan never
    force-ends the call even at tenant.credits == 0; credits exhausted
    mid-overage still force-ends the call (existing behavior preserved).
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.models.plan import Plan
from app.models.pricing_configs import PricingConfig
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.user import User
from app.models.agent import Agent
from app.models.call_session import CallSession
from app.models.usage_record import UsageRecord
from app.services.credit_service import (
    CreditService,
    DEFAULT_PER_MINUTE_RATE,
    OPENAI_REALTIME_SURCHARGE_PER_MINUTE,
    ELEVENLABS_TTS_SURCHARGE_PER_MINUTE,
)


class _CloseSafeSessionWrapper:
    """Wraps a shared test session so credit_service's internal
    `SessionLocal()` calls reuse it instead of opening a real DB connection,
    and so its `.close()` doesn't tear down the fixture session."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def override_session_local(db):
    with patch("app.db.session.SessionLocal", lambda: _CloseSafeSessionWrapper(db)):
        yield


@pytest.fixture
def service():
    """Fresh CreditService instance per test — avoids cross-test state leakage
    in the singleton's per-call tracking dicts."""
    return CreditService()


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Credit Test Tenant {suffix}",
        schema_name=f"credit_test_{suffix}",
        status="active",
        credits=Decimal("0"),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def user(db, tenant):
    suffix = uuid.uuid4().hex[:8]
    u = User(
        email=f"credit_test_{suffix}@example.com",
        hashed_password="x",
        first_name="Credit",
        last_name="Tester",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def call_session(db, tenant, user):
    agent = Agent(tenant_id=tenant.id, name="Test Agent")
    db.add(agent)
    db.commit()
    db.refresh(agent)

    cs = CallSession(
        user_id=user.id,
        agent_id=agent.id,
        tenant_id=tenant.id,
        start_time=datetime.now(timezone.utc),
        status="active",
        call_type="inbound",
        twilio_call_sid="CA_test_123",
    )
    db.add(cs)
    db.commit()
    db.refresh(cs)
    return cs


def _activate_plan(db, tenant, user, *, included_minutes):
    plan = Plan(
        name=f"studio_{uuid.uuid4().hex[:8]}",
        display_name="Studio",
        price_monthly=9900,
        crm_type=None,
        included_minutes=included_minutes,
        monthly_credits=Decimal("50.00"),
        is_active=True,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    now = datetime.now(timezone.utc)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=plan.id,
        crm_type=None,
        status="active",
        current_period_start=now - timedelta(days=1),
        current_period_end=now + timedelta(days=29),
    )
    db.add(sub)
    db.commit()
    return plan, sub


# ---------------------------------------------------------------------------
# get_effective_rate_per_minute
# ---------------------------------------------------------------------------


def test_effective_rate_defaults_when_no_pricing_config(db, tenant, service):
    rate = service.get_effective_rate_per_minute(db, tenant.id)
    assert rate == DEFAULT_PER_MINUTE_RATE


def test_effective_rate_uses_configured_pricing(db, tenant, service):
    db.add(PricingConfig(workspace_id=tenant.id, per_minute_rate=Decimal("0.25")))
    db.commit()
    rate = service.get_effective_rate_per_minute(db, tenant.id)
    assert rate == Decimal("0.25")


# ---------------------------------------------------------------------------
# has_sufficient_credits
# ---------------------------------------------------------------------------


def test_has_sufficient_credits_allowed_by_plan_even_at_zero_credits(db, tenant, user, service):
    _activate_plan(db, tenant, user, included_minutes=100)
    tenant.credits = Decimal("0")
    db.commit()

    ok, current_credits, _, reason = service.has_sufficient_credits(db, tenant.id)
    assert ok is True
    assert current_credits == 0.0
    assert reason == "ok"


def test_has_sufficient_credits_pay_as_you_go_requires_credits(db, tenant, service):
    tenant.credits = Decimal("0")
    db.commit()
    ok, _, _, reason = service.has_sufficient_credits(db, tenant.id)
    assert ok is False
    assert "no included minutes" in reason and "no credit" in reason

    tenant.credits = Decimal("5")
    db.commit()
    ok, _, _, reason = service.has_sufficient_credits(db, tenant.id)
    assert ok is True
    assert reason == "ok"


def test_has_sufficient_credits_realtime_model_blocked_at_zero_credits_despite_plan(
    db, tenant, user, service
):
    """A gpt-realtime call must be blocked at 0 credits even with plenty of
    included minutes remaining, since the surcharge draws from credits from
    the very first tick regardless of the included-minutes allowance."""
    _activate_plan(db, tenant, user, included_minutes=100)
    tenant.credits = Decimal("0")
    db.commit()

    ok, _, _, reason = service.has_sufficient_credits(db, tenant.id, model_name="gpt-realtime")
    assert ok is False
    assert "OpenAI Realtime" in reason

    tenant.credits = Decimal("1")
    db.commit()
    ok, _, _, reason = service.has_sufficient_credits(db, tenant.id, model_name="gpt-realtime-2")
    assert ok is True
    assert reason == "ok"


def test_has_sufficient_credits_non_realtime_model_unaffected_at_zero_credits(
    db, tenant, user, service
):
    """Non-realtime models keep today's behavior: included minutes alone are
    enough to allow the call, even at 0 credits."""
    _activate_plan(db, tenant, user, included_minutes=100)
    tenant.credits = Decimal("0")
    db.commit()

    ok, _, _, reason = service.has_sufficient_credits(db, tenant.id, model_name="gpt-4o-mini")
    assert ok is True
    assert reason == "ok"


def test_has_sufficient_credits_elevenlabs_tts_blocked_at_zero_credits_despite_plan(
    db, tenant, user, service
):
    """An agent resolved to the ElevenLabs TTS provider must be blocked at 0
    credits even with plenty of included minutes remaining — same mechanic as
    the realtime surcharge."""
    _activate_plan(db, tenant, user, included_minutes=100)
    tenant.credits = Decimal("0")
    db.commit()

    ok, _, _, reason = service.has_sufficient_credits(
        db, tenant.id, model_name="gpt-4o-mini", tts_provider_slug="elevenlabs"
    )
    assert ok is False
    assert "ElevenLabs" in reason

    tenant.credits = Decimal("1")
    db.commit()
    ok, _, _, reason = service.has_sufficient_credits(
        db, tenant.id, model_name="gpt-4o-mini", tts_provider_slug="elevenlabs"
    )
    assert ok is True
    assert reason == "ok"


def test_has_sufficient_credits_byo_elevenlabs_unaffected_at_zero_credits(
    db, tenant, user, service
):
    """BYO ElevenLabs must behave like a non-surcharged TTS provider at the
    pre-call gate: included minutes alone are enough, even at 0 credits."""
    _activate_plan(db, tenant, user, included_minutes=100)
    tenant.credits = Decimal("0")
    db.commit()

    ok, _, _, reason = service.has_sufficient_credits(
        db,
        tenant.id,
        model_name="gpt-4o-mini",
        tts_provider_slug="elevenlabs",
        is_byo_elevenlabs=True,
    )
    assert ok is True
    assert reason == "ok"


def test_has_sufficient_credits_non_elevenlabs_tts_unaffected_at_zero_credits(
    db, tenant, user, service
):
    """A non-ElevenLabs TTS provider (e.g. google) keeps today's behavior:
    included minutes alone are enough to allow the call, even at 0 credits."""
    _activate_plan(db, tenant, user, included_minutes=100)
    tenant.credits = Decimal("0")
    db.commit()

    ok, _, _, reason = service.has_sufficient_credits(
        db, tenant.id, model_name="gpt-4o-mini", tts_provider_slug="google"
    )
    assert ok is True
    assert reason == "ok"


def test_get_active_surcharges_realtime_model_excludes_elevenlabs_check(service):
    """Architectural guarantee: OpenAI Realtime speech-to-speech calls bypass
    the separate TTS synthesis pipeline entirely, so even if the agent's
    configured tts_provider_slug happens to be "elevenlabs" (an unused, stale
    field for a realtime call), only the realtime surcharge is returned — the
    two surcharges never stack on the same call in this codebase today."""
    surcharges = service.get_active_surcharges(
        model_name="gpt-realtime", tts_provider_slug="elevenlabs"
    )
    keys = [s.key for s in surcharges]
    assert keys == ["openai_realtime"]


def test_get_active_surcharges_non_realtime_elevenlabs_returns_single_surcharge(service):
    surcharges = service.get_active_surcharges(
        model_name="gpt-4o-mini", tts_provider_slug="elevenlabs"
    )
    keys = [s.key for s in surcharges]
    assert keys == ["elevenlabs_tts"]


def test_get_active_surcharges_none_when_neither_applies(service):
    assert service.get_active_surcharges(model_name="gpt-4o-mini", tts_provider_slug="google") == []


def test_get_active_surcharges_byo_elevenlabs_exempted(service):
    """Fix #1: BYO ElevenLabs must NOT incur the ElevenLabs surcharge — the
    platform pays nothing when the tenant supplies their own key."""
    surcharges = service.get_active_surcharges(
        model_name="gpt-4o-mini", tts_provider_slug="elevenlabs", is_byo_elevenlabs=True
    )
    assert surcharges == []


def test_get_active_surcharges_platform_elevenlabs_still_charged(service):
    """Sanity counterpart: platform ElevenLabs (is_byo_elevenlabs=False, the
    default) is still charged."""
    surcharges = service.get_active_surcharges(
        model_name="gpt-4o-mini", tts_provider_slug="elevenlabs", is_byo_elevenlabs=False
    )
    keys = [s.key for s in surcharges]
    assert keys == ["elevenlabs_tts"]


def test_get_active_surcharges_realtime_fallen_back_switches_to_elevenlabs(service):
    """Fix #4: once a realtime-configured call has fallen back to the legacy
    pipeline, the realtime surcharge must stop applying and the ElevenLabs
    surcharge (if applicable) must be evaluated for real instead of being
    skipped by the realtime early-return."""
    surcharges = service.get_active_surcharges(
        model_name="gpt-realtime",
        tts_provider_slug="elevenlabs",
        realtime_fallen_back=True,
    )
    keys = [s.key for s in surcharges]
    assert keys == ["elevenlabs_tts"]


def test_get_active_surcharges_realtime_fallen_back_byo_elevenlabs_exempted(service):
    """Fix #1 + #4 combined: post-fallback BYO ElevenLabs is exempt from
    both the (now-inactive) realtime surcharge and the ElevenLabs surcharge."""
    surcharges = service.get_active_surcharges(
        model_name="gpt-realtime",
        tts_provider_slug="elevenlabs",
        is_byo_elevenlabs=True,
        realtime_fallen_back=True,
    )
    assert surcharges == []


def test_get_active_surcharges_realtime_not_fallen_back_unaffected(service):
    """`realtime_fallen_back` defaults to False and must not change existing
    (non-fallback) realtime behavior."""
    surcharges = service.get_active_surcharges(
        model_name="gpt-realtime", tts_provider_slug="elevenlabs"
    )
    keys = [s.key for s in surcharges]
    assert keys == ["openai_realtime"]


# ---------------------------------------------------------------------------
# _finalize_call_credits — free/billable split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_within_free_allowance_deducts_zero(db, tenant, user, call_session, service):
    _activate_plan(db, tenant, user, included_minutes=10)  # 600s remaining, 0 used
    tenant.credits = Decimal("0")
    db.commit()

    call_id_str = str(call_session.id)
    service._accumulated_seconds[call_id_str] = 120.0  # 2 minutes, well within 600s allowance

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-4o-mini", 12.0, call_session
    )

    db.refresh(tenant)
    assert tenant.credits == Decimal("0")

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert record is not None
    assert record.credits_charged == Decimal("0")
    assert round(float(record.billable_minutes), 2) == 2.0


@pytest.mark.asyncio
async def test_finalize_overage_charges_only_billable_portion(db, tenant, user, call_session, service):
    plan, sub = _activate_plan(db, tenant, user, included_minutes=1)  # 60s remaining, 0 used so far
    tenant.credits = Decimal("10")
    db.commit()

    call_id_str = str(call_session.id)
    # 90s accumulated: 60s free, 30s billable (0.5 min) at 12 credits/min => 6 credits
    service._accumulated_seconds[call_id_str] = 90.0

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-4o-mini", 12.0, call_session
    )

    db.refresh(tenant)
    assert tenant.credits == Decimal("10") - Decimal("6")

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert record is not None
    assert round(float(record.credits_charged), 2) == 6.0
    assert round(float(record.billable_minutes), 2) == 1.5


@pytest.mark.asyncio
async def test_finalize_pay_as_you_go_charges_from_first_second(db, tenant, call_session, service):
    # No active subscription at all.
    tenant.credits = Decimal("5")
    db.commit()

    call_id_str = str(call_session.id)
    service._accumulated_seconds[call_id_str] = 30.0  # 0.5 min, fully billable

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-4o-mini", 12.0, call_session
    )

    db.refresh(tenant)
    # deduct_credits never allows a negative balance (clamped to 0), even though
    # the nominal charge (6) exceeds the starting balance (5).
    assert tenant.credits == Decimal("0")

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert round(float(record.credits_charged), 2) == 6.0


# ---------------------------------------------------------------------------
# _finalize_call_credits — OpenAI Realtime surcharge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_realtime_within_allowance_charges_surcharge_only(
    db, tenant, user, call_session, service
):
    """Within the plan's included minutes: base rate stays waived, but the
    flat $0.50/min surcharge is still charged on the full accumulated time."""
    _activate_plan(db, tenant, user, included_minutes=10)  # 600s remaining, 0 used
    tenant.credits = Decimal("10")
    db.commit()

    call_id_str = str(call_session.id)
    service._accumulated_seconds[call_id_str] = 120.0  # 2 minutes, fully within allowance

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-realtime", 12.0, call_session,
    )

    db.refresh(tenant)
    expected_surcharge = Decimal("2") * OPENAI_REALTIME_SURCHARGE_PER_MINUTE  # 2 min * 0.50
    assert tenant.credits == Decimal("10") - expected_surcharge

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert record is not None
    assert round(float(record.credits_charged), 2) == float(expected_surcharge)


@pytest.mark.asyncio
async def test_finalize_realtime_overage_charges_base_plus_surcharge(
    db, tenant, user, call_session, service
):
    """Once included minutes are exhausted: total charge is base rate +
    surcharge on the full duration ($0.62/min at the default 0.12 base)."""
    _activate_plan(db, tenant, user, included_minutes=1)  # 60s remaining, 0 used so far
    tenant.credits = Decimal("100")
    db.commit()

    call_id_str = str(call_session.id)
    # 120s accumulated: 60s free (base waived), 60s billable at 12 credits/min => 12 credits base
    # Surcharge applies to the FULL 120s => 2 min * 0.50 = 1.0 credit
    service._accumulated_seconds[call_id_str] = 120.0

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-realtime-2", 12.0, call_session,
    )

    db.refresh(tenant)
    expected_base = Decimal("12")  # 1 billable minute * 12 credits/min
    expected_surcharge = Decimal("2") * OPENAI_REALTIME_SURCHARGE_PER_MINUTE  # 2 min * 0.50
    expected_total = expected_base + expected_surcharge
    assert tenant.credits == Decimal("100") - expected_total

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert round(float(record.credits_charged), 2) == float(expected_total)


@pytest.mark.asyncio
async def test_finalize_non_realtime_model_unaffected_by_surcharge_logic(
    db, tenant, user, call_session, service
):
    """Sanity check: a non-realtime, non-ElevenLabs model/TTS combo (the
    defaults) behaves identically to the pre-existing tests — no surcharges
    apply."""
    _activate_plan(db, tenant, user, included_minutes=10)
    tenant.credits = Decimal("10")
    db.commit()

    call_id_str = str(call_session.id)
    service._accumulated_seconds[call_id_str] = 120.0

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-4o-mini", 12.0, call_session,
    )

    db.refresh(tenant)
    assert tenant.credits == Decimal("10")


# ---------------------------------------------------------------------------
# _finalize_call_credits — ElevenLabs TTS surcharge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_elevenlabs_within_allowance_charges_surcharge_only(
    db, tenant, user, call_session, service
):
    """Within the plan's included minutes: base rate stays waived, but the
    flat $0.05/min ElevenLabs surcharge is still charged on the full
    accumulated time."""
    _activate_plan(db, tenant, user, included_minutes=10)  # 600s remaining, 0 used
    tenant.credits = Decimal("10")
    db.commit()

    call_id_str = str(call_session.id)
    service._accumulated_seconds[call_id_str] = 120.0  # 2 minutes, fully within allowance

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-4o-mini", 12.0, call_session,
        tts_provider_slug="elevenlabs",
    )

    db.refresh(tenant)
    expected_surcharge = Decimal("2") * ELEVENLABS_TTS_SURCHARGE_PER_MINUTE  # 2 min * 0.05
    assert tenant.credits == Decimal("10") - expected_surcharge

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert record is not None
    assert round(float(record.credits_charged), 4) == float(expected_surcharge)


@pytest.mark.asyncio
async def test_finalize_elevenlabs_overage_charges_base_plus_surcharge(
    db, tenant, user, call_session, service
):
    """Once included minutes are exhausted: total charge is base rate +
    ElevenLabs surcharge on the full duration ($0.17/min at the default 0.12
    base)."""
    _activate_plan(db, tenant, user, included_minutes=1)  # 60s remaining, 0 used so far
    tenant.credits = Decimal("100")
    db.commit()

    call_id_str = str(call_session.id)
    # 120s accumulated: 60s free (base waived), 60s billable at 12 credits/min => 12 credits base
    # Surcharge applies to the FULL 120s => 2 min * 0.05 = 0.10 credit
    service._accumulated_seconds[call_id_str] = 120.0

    await service._finalize_call_credits(
        db, call_session.id, tenant.id, "gpt-4o-mini", 12.0, call_session,
        tts_provider_slug="elevenlabs",
    )

    db.refresh(tenant)
    expected_base = Decimal("12")  # 1 billable minute * 12 credits/min
    expected_surcharge = Decimal("2") * ELEVENLABS_TTS_SURCHARGE_PER_MINUTE  # 2 min * 0.05
    expected_total = expected_base + expected_surcharge
    assert tenant.credits == Decimal("100") - expected_total

    record = db.query(UsageRecord).filter(UsageRecord.workspace_id == tenant.id).first()
    assert round(float(record.credits_charged), 4) == float(expected_total)


def test_describe_surcharges_stacks_additively_if_both_active(service):
    """Structural stacking check: if a call's active-surcharges list somehow
    contained both rules (not reachable via get_active_surcharges today per
    the architectural guarantee tested in
    test_get_active_surcharges_realtime_model_excludes_elevenlabs_check —
    OpenAI Realtime bypasses the separate TTS pipeline entirely — but the
    deduction math itself (_describe_surcharges, the shared computation used
    by both _monitor_and_deduct_credits and _finalize_call_credits) must sum
    every entry in the list additively so future stacking works without code
    changes here)."""
    surcharges = list(service.get_surcharge_catalog())  # both rules, forced
    total, _desc = service._describe_surcharges(60.0, surcharges)  # 1 minute

    expected_surcharge = OPENAI_REALTIME_SURCHARGE_PER_MINUTE + ELEVENLABS_TTS_SURCHARGE_PER_MINUTE
    assert round(total, 4) == float(expected_surcharge)


# ---------------------------------------------------------------------------
# _monitor_and_deduct_credits — force-end behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_loop_does_not_force_end_within_free_allowance(db, tenant, user, call_session, service):
    """A tick fully within remaining included minutes must not force-end the
    call, even at tenant.credits == 0."""
    _activate_plan(db, tenant, user, included_minutes=100)  # far more than one tick's worth
    tenant.credits = Decimal("0")
    db.commit()

    call_id_str = str(call_session.id)
    past = datetime.now(timezone.utc) - timedelta(seconds=30)
    service._call_start_times[call_id_str] = past
    service._last_deduction_time[call_id_str] = past
    service._accumulated_seconds[call_id_str] = 0.0

    calls = {"n": 0}

    async def _fake_sleep(_interval):
        calls["n"] += 1
        if calls["n"] >= 2:
            # End the loop deterministically after one processed tick by
            # flipping the call to a terminal status for the next iteration's
            # status check (mirrors a real call hangup).
            fresh = db.query(CallSession).filter(CallSession.id == call_session.id).first()
            fresh.status = "completed"
            fresh.ended_reason = "Customer hung up"
            db.commit()

    with patch("asyncio.sleep", side_effect=_fake_sleep), \
         patch.object(service, "_end_twilio_call", new=AsyncMock()) as mock_end_twilio, \
         patch("app.routers.general_websocket.broadcast_call_status_update", new=AsyncMock()) as mock_broadcast:
        await service._monitor_and_deduct_credits(call_session.id, tenant.id, "gpt-4o-mini", 12.0)

    db.refresh(tenant)
    db.refresh(call_session)

    assert tenant.credits == Decimal("0")
    assert call_session.ended_reason == "Customer hung up"  # not force-ended for credits
    mock_end_twilio.assert_not_called()
    mock_broadcast.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_loop_force_ends_call_when_credits_exhausted_mid_overage(db, tenant, call_session, service):
    """Pay-as-you-go tenant with insufficient credits to cover overage must
    still be force-ended (existing behavior, unchanged by the plan split)."""
    tenant.credits = Decimal("0.05")  # not enough to cover the tick
    db.commit()

    call_id_str = str(call_session.id)
    # Naive (see _NaiveUtcDatetime note below) so it's directly comparable to
    # the patched module's `datetime.now()` inside the monitor loop.
    past = datetime.utcnow() - timedelta(seconds=60)
    service._call_start_times[call_id_str] = past
    service._last_deduction_time[call_id_str] = past
    service._accumulated_seconds[call_id_str] = 0.0

    # SQLite (used by the "db" test fixture) drops tzinfo on datetime columns
    # on round-trip, so call_session.start_time comes back naive once
    # re-queried inside the monitor loop's own session. Real Postgres
    # (DateTime(timezone=True)) preserves tzinfo end-to-end and never hits
    # this mismatch — this shim only neutralizes the SQLite-only artifact so
    # the force-end duration computation (`end_time - start_time`) doesn't
    # raise on a naive/aware subtraction in this test environment.
    class _NaiveUtcDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.utcnow()

    with patch("asyncio.sleep", new=AsyncMock()), \
         patch.object(service, "_end_twilio_call", new=AsyncMock()) as mock_end_twilio, \
         patch("app.routers.general_websocket.broadcast_call_status_update", new=AsyncMock()) as mock_broadcast, \
         patch("app.services.credit_service.datetime", _NaiveUtcDatetime):
        await service._monitor_and_deduct_credits(call_session.id, tenant.id, "gpt-4o-mini", 12.0)

    db.refresh(tenant)
    db.refresh(call_session)

    assert tenant.credits == Decimal("0")
    assert call_session.status == "completed"
    assert call_session.ended_reason == "Insufficient credits"
    mock_end_twilio.assert_awaited_once()
    mock_broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_monitor_loop_force_ends_realtime_call_when_surcharge_exhausts_credits_within_free_allowance(
    db, tenant, user, call_session, service
):
    """A realtime-model call must still be force-ended when credits run out,
    even while nominally "within the free minutes" for the base rate —
    the surcharge alone can exhaust the balance."""
    _activate_plan(db, tenant, user, included_minutes=100)  # far more than one tick's worth
    tenant.credits = Decimal("0.05")  # not enough to cover even one tick's surcharge
    db.commit()

    call_id_str = str(call_session.id)
    past = datetime.utcnow() - timedelta(seconds=30)
    service._call_start_times[call_id_str] = past
    service._last_deduction_time[call_id_str] = past
    service._accumulated_seconds[call_id_str] = 0.0

    class _NaiveUtcDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.utcnow()

    with patch("asyncio.sleep", new=AsyncMock()), \
         patch.object(service, "_end_twilio_call", new=AsyncMock()) as mock_end_twilio, \
         patch("app.routers.general_websocket.broadcast_call_status_update", new=AsyncMock()) as mock_broadcast, \
         patch("app.services.credit_service.datetime", _NaiveUtcDatetime):
        await service._monitor_and_deduct_credits(
            call_session.id, tenant.id, "gpt-realtime", 12.0
        )

    db.refresh(tenant)
    db.refresh(call_session)

    assert tenant.credits == Decimal("0")
    assert call_session.status == "completed"
    assert call_session.ended_reason == "Insufficient credits"
    mock_end_twilio.assert_awaited_once()
    mock_broadcast.assert_awaited_once()


# ---------------------------------------------------------------------------
# Fix #4 — mid-call OpenAI Realtime fallback re-evaluates surcharges per tick
# ---------------------------------------------------------------------------


def test_resolve_current_surcharges_reads_fallback_flag_from_call_metadata(
    call_session, service
):
    """`_resolve_current_surcharges` must treat a call as fallen-back once
    CallSession.call_metadata["openai_realtime_fallback"]["fell_back"] is
    True — the shape VoiceOrchestrator._fallback_to_legacy_pipeline_openai
    persists."""
    # Before fallback: realtime model → realtime surcharge only.
    surcharges = service._resolve_current_surcharges(
        call_session, "gpt-realtime", "elevenlabs", False
    )
    assert [s.key for s in surcharges] == ["openai_realtime"]

    # Orchestrator persists the fallback flag on call_metadata.
    call_session.call_metadata = {
        "openai_realtime_fallback": {"fell_back": True, "reason": "timeout"}
    }

    surcharges = service._resolve_current_surcharges(
        call_session, "gpt-realtime", "elevenlabs", False
    )
    assert [s.key for s in surcharges] == ["elevenlabs_tts"]


@pytest.mark.asyncio
async def test_monitor_loop_switches_surcharge_after_mid_call_realtime_fallback(
    db, tenant, user, call_session, service
):
    """End-to-end fix #4 scenario: a call configured for an OpenAI Realtime
    model bills the realtime surcharge on its first tick; once the
    orchestrator's fallback flag lands on call_metadata (simulated here the
    same way VoiceOrchestrator._fallback_to_legacy_pipeline_openai persists
    it), the very next tick must stop charging the realtime surcharge and
    start charging the ElevenLabs surcharge instead."""
    _activate_plan(db, tenant, user, included_minutes=100)  # base rate fully waived
    tenant.credits = Decimal("10")
    db.commit()

    call_id_str = str(call_session.id)
    past = datetime.now(timezone.utc) - timedelta(seconds=60)
    service._call_start_times[call_id_str] = past
    service._last_deduction_time[call_id_str] = past
    service._accumulated_seconds[call_id_str] = 0.0

    calls = {"n": 0}

    async def _fake_sleep(_interval):
        calls["n"] += 1
        if calls["n"] == 2:
            # Orchestrator falls back mid-call — persist the flag exactly as
            # VoiceOrchestrator._fallback_to_legacy_pipeline_openai does.
            fresh = db.query(CallSession).filter(CallSession.id == call_session.id).first()
            meta = dict(fresh.call_metadata or {})
            meta["openai_realtime_fallback"] = {"fell_back": True, "reason": "test"}
            fresh.call_metadata = meta
            db.commit()
            # Force a second deterministic ~60s tick.
            service._last_deduction_time[call_id_str] = datetime.now(timezone.utc) - timedelta(seconds=60)
        elif calls["n"] >= 3:
            fresh = db.query(CallSession).filter(CallSession.id == call_session.id).first()
            fresh.status = "completed"
            fresh.ended_reason = "Customer hung up"
            db.commit()

    with patch("asyncio.sleep", side_effect=_fake_sleep), \
         patch.object(service, "_end_twilio_call", new=AsyncMock()), \
         patch("app.routers.general_websocket.broadcast_call_status_update", new=AsyncMock()):
        await service._monitor_and_deduct_credits(
            call_session.id,
            tenant.id,
            "gpt-realtime",
            12.0,
            tts_provider_slug="elevenlabs",
            is_byo_elevenlabs=False,
        )

    db.refresh(tenant)
    db.refresh(call_session)

    # Tick 1 (not yet fallen back): ~1 min * $0.50/min realtime surcharge.
    # Tick 2 (fallen back): ~1 min * $0.05/min ElevenLabs surcharge instead of
    # a second realtime surcharge (which would total ~$1.00 if the bug were
    # still present).
    expected_total = OPENAI_REALTIME_SURCHARGE_PER_MINUTE + ELEVENLABS_TTS_SURCHARGE_PER_MINUTE
    charged = float(Decimal("10") - tenant.credits)
    assert charged == pytest.approx(float(expected_total), abs=0.01)
    assert call_session.call_metadata.get("openai_realtime_fallback", {}).get("fell_back") is True
    assert call_session.ended_reason == "Customer hung up"
