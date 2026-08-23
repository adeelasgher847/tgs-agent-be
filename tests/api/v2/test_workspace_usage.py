"""GET /api/v2/workspace/usage — plan-aware usage endpoint.

Prior to the Studio/Agency plan work this endpoint crashed for any tenant
without the old (removed) per-model credit-cost table. Coverage:
  - Returns correctly with an active plan + some recorded usage
    (minutes_included / overage_minutes reflect the plan's included_minutes).
  - Does not raise for a tenant with no active plan (pay-as-you-go).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_db, require_billing
from app.core.exception_handlers import register_exception_handlers
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.usage_record import UsageRecord
from app.models.user import User


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"UsageEP Tenant {suffix}",
        schema_name=f"usageep_{suffix}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def billing_user(db, tenant):
    suffix = uuid.uuid4().hex[:8]
    u = User(
        email=f"billing_{suffix}@test.com",
        current_tenant_id=tenant.id,
        first_name="Bill",
        last_name="User",
        hashed_password="X",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _client(db, user):
    from app.api.v2.routers.workspace import v2_router

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(v2_router, prefix="/workspace")

    app.dependency_overrides[require_billing] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def test_usage_endpoint_with_active_plan_and_recorded_usage(db, tenant, billing_user):
    suffix = uuid.uuid4().hex[:8]
    plan = Plan(
        name=f"studio_{suffix}",
        display_name="Studio",
        price_monthly=9900,
        crm_type=None,
        included_minutes=100,
        is_active=True,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    now = datetime.now(timezone.utc)
    sub = Subscription(
        user_id=billing_user.id,
        tenant_id=tenant.id,
        plan_id=plan.id,
        crm_type=None,
        status="active",
        current_period_start=now - timedelta(days=1),
        current_period_end=now + timedelta(days=29),
    )
    db.add(sub)
    db.add(
        UsageRecord(
            workspace_id=tenant.id,
            billable_minutes=Decimal("130"),
            credits_charged=Decimal("3.60"),
            recorded_at=now,
        )
    )
    db.commit()

    client = _client(db, billing_user)
    res = client.get("/workspace/usage")

    assert res.status_code == 200
    data = res.json()
    assert Decimal(str(data["minutes_included"])) == Decimal("100")
    assert Decimal(str(data["minutes_used_this_cycle"])) == Decimal("130")
    assert Decimal(str(data["overage_minutes"])) == Decimal("30")

    # GAP fix: the realtime/ElevenLabs surcharges must be visible via the API,
    # not just documented in a source comment.
    surcharge_keys = {s["key"] for s in data["available_surcharges"]}
    assert surcharge_keys == {"openai_realtime", "elevenlabs_tts"}
    realtime = next(s for s in data["available_surcharges"] if s["key"] == "openai_realtime")
    assert Decimal(str(realtime["rate_per_minute"])) == Decimal("0.50")
    elevenlabs = next(s for s in data["available_surcharges"] if s["key"] == "elevenlabs_tts")
    assert Decimal(str(elevenlabs["rate_per_minute"])) == Decimal("0.05")


def test_usage_endpoint_no_active_plan_does_not_raise(db, tenant, billing_user):
    client = _client(db, billing_user)
    res = client.get("/workspace/usage")

    assert res.status_code == 200
    data = res.json()
    assert Decimal(str(data["minutes_included"])) == Decimal("0")
    assert Decimal(str(data["overage_minutes"])) == Decimal("0")
    assert {s["key"] for s in data["available_surcharges"]} == {"openai_realtime", "elevenlabs_tts"}
