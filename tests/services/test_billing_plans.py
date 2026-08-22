"""Studio/Agency plan-aware billing_service coverage.

Coverage:
  - get_included_minutes_status: no subscription -> included=0; active Studio
    subscription with 0 usage -> remaining == included_minutes; usage recorded
    within the current cycle reduces remaining; usage recorded before the
    cycle start does not count.
  - get_or_create_subscription / update_subscription with tenant_id: creates a
    tenant-scoped core-plan row distinct from user_id+crm_type rows, and
    coexists with legacy CRM-addon rows for the same user.
  - sync_payment_status plan_purchase branch: core plan grants monthly_credits;
    CRM plan grants zero credits (unchanged).
  - handle_subscription_renewed: rolls the period forward and re-grants
    monthly_credits.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.usage_record import UsageRecord
from app.models.user import User
from app.services.billing_service import BillingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Plan Test Tenant {suffix}",
        schema_name=f"plan_test_{suffix}",
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
        email=f"plan_test_{suffix}@example.com",
        hashed_password="x",
        first_name="Plan",
        last_name="Tester",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def studio_plan(db):
    suffix = uuid.uuid4().hex[:8]
    p = Plan(
        name=f"studio_{suffix}",
        display_name="Studio",
        price_monthly=9900,
        crm_type=None,
        included_minutes=100,
        monthly_credits=Decimal("50.00"),
        max_subaccounts=3,
        free_phone_numbers=1,
        is_active=True,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@pytest.fixture
def crm_plan(db):
    suffix = uuid.uuid4().hex[:8]
    p = Plan(
        name=f"monday_crm_{suffix}",
        display_name="Monday CRM Addon",
        price_monthly=1900,
        crm_type="monday",
        is_active=True,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ---------------------------------------------------------------------------
# get_included_minutes_status
# ---------------------------------------------------------------------------


def test_no_active_subscription_included_is_zero(db, tenant):
    included, used, remaining = BillingService.get_included_minutes_status(db, tenant.id)
    assert included == Decimal("0")
    assert remaining == Decimal("0")


def test_active_studio_subscription_no_usage_remaining_equals_included(db, tenant, user, studio_plan):
    now = datetime.now(timezone.utc)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        current_period_start=now - timedelta(days=1),
        current_period_end=now + timedelta(days=29),
    )
    db.add(sub)
    db.commit()

    included, used, remaining = BillingService.get_included_minutes_status(db, tenant.id)
    assert included == Decimal("100")
    assert used == Decimal("0")
    assert remaining == Decimal("100")


def test_usage_within_cycle_reduces_remaining(db, tenant, user, studio_plan):
    now = datetime.now(timezone.utc)
    cycle_start = now - timedelta(days=1)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        current_period_start=cycle_start,
        current_period_end=now + timedelta(days=29),
    )
    db.add(sub)
    db.commit()

    db.add(
        UsageRecord(
            workspace_id=tenant.id,
            billable_minutes=Decimal("30"),
            credits_charged=Decimal("0"),
            recorded_at=now,
        )
    )
    db.commit()

    included, used, remaining = BillingService.get_included_minutes_status(db, tenant.id)
    assert included == Decimal("100")
    assert used == Decimal("30")
    assert remaining == Decimal("70")


def test_usage_outside_cycle_window_does_not_count(db, tenant, user, studio_plan):
    now = datetime.now(timezone.utc)
    cycle_start = now - timedelta(days=1)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        current_period_start=cycle_start,
        current_period_end=now + timedelta(days=29),
    )
    db.add(sub)
    db.commit()

    # Usage recorded before the current cycle started must not count.
    db.add(
        UsageRecord(
            workspace_id=tenant.id,
            billable_minutes=Decimal("999"),
            credits_charged=Decimal("0"),
            recorded_at=cycle_start - timedelta(days=5),
        )
    )
    db.commit()

    included, used, remaining = BillingService.get_included_minutes_status(db, tenant.id)
    assert used == Decimal("0")
    assert remaining == Decimal("100")


# ---------------------------------------------------------------------------
# get_or_create_subscription / update_subscription — tenant_id path vs. legacy
# ---------------------------------------------------------------------------


def test_get_or_create_subscription_tenant_scoped_row(db, tenant, user):
    sub = BillingService.get_or_create_subscription(db, user_id=user.id, tenant_id=tenant.id)
    assert sub.tenant_id == tenant.id
    assert sub.crm_type is None

    # Idempotent: fetching again returns the same row.
    sub2 = BillingService.get_or_create_subscription(db, user_id=user.id, tenant_id=tenant.id)
    assert sub2.id == sub.id


def test_tenant_scoped_and_crm_subscriptions_coexist_for_same_user(db, tenant, user, crm_plan):
    tenant_sub = BillingService.get_or_create_subscription(db, user_id=user.id, tenant_id=tenant.id)

    crm_sub = BillingService.update_subscription(
        db=db,
        user_id=user.id,
        plan_id=crm_plan.id,
        crm_type="monday",
    )

    assert tenant_sub.id != crm_sub.id
    assert tenant_sub.tenant_id == tenant.id
    assert crm_sub.tenant_id is None
    assert crm_sub.crm_type == "monday"


def test_update_subscription_legacy_crm_path_unchanged(db, tenant, user, crm_plan):
    """Re-run of the core test_billing_idempotency-style scenario: update_subscription
    without tenant_id behaves exactly as before (user_id + crm_type lookup/create)."""
    first = BillingService.update_subscription(
        db=db,
        user_id=user.id,
        plan_id=crm_plan.id,
        crm_type="monday",
        stripe_subscription_id="sub_abc",
    )
    assert first.user_id == user.id
    assert first.crm_type == "monday"
    assert first.tenant_id is None

    second = BillingService.update_subscription(
        db=db,
        user_id=user.id,
        plan_id=crm_plan.id,
        crm_type="monday",
        stripe_subscription_id="sub_abc_updated",
    )
    # Same row updated in place, not a duplicate.
    assert second.id == first.id
    assert second.stripe_subscription_id == "sub_abc_updated"


def test_update_subscription_tenant_scoped_updates_existing_row(db, tenant, user, studio_plan):
    now = datetime.now(timezone.utc)
    first = BillingService.update_subscription(
        db=db,
        user_id=user.id,
        plan_id=studio_plan.id,
        tenant_id=tenant.id,
        current_period_start=now,
        current_period_end=now + timedelta(days=30),
    )
    assert first.tenant_id == tenant.id

    second = BillingService.update_subscription(
        db=db,
        user_id=user.id,
        plan_id=studio_plan.id,
        tenant_id=tenant.id,
        status="past_due",
    )
    assert second.id == first.id
    assert second.status == "past_due"


# ---------------------------------------------------------------------------
# sync_payment_status — plan_purchase branch
# ---------------------------------------------------------------------------


class _FakeStripeSession:
    payment_status = "paid"

    def __init__(self, *, tenant_id, user_id, plan_id, session_id="cs_plan_test", crm_type=None):
        self._data = {
            "id": session_id,
            "payment_status": "paid",
            "amount_total": 9900,
            "customer": "cus_test123",
            "subscription": None,
            "metadata": {
                "tenant_id": str(tenant_id),
                "user_id": str(user_id),
                "plan_id": str(plan_id),
                "purchase_type": "plan_purchase",
                **({"crm_type": crm_type} if crm_type else {}),
            },
        }

    def __getitem__(self, key):
        return self._data[key]


@patch("stripe.checkout.Session.retrieve")
def test_sync_payment_status_core_plan_grants_monthly_credits(mock_retrieve, db, tenant, user, studio_plan):
    mock_retrieve.return_value = _FakeStripeSession(
        tenant_id=tenant.id, user_id=user.id, plan_id=studio_plan.id, session_id="cs_core_1"
    )

    result = BillingService.sync_payment_status(db, "cs_core_1", "evt_core_1")

    assert result["status"] == "success"
    assert Decimal(str(result["credits_added"])) == Decimal("50.00")
    assert result["crm_type"] is None

    db.refresh(tenant)
    assert tenant.credits == Decimal("50.00")

    sub = BillingService.get_workspace_subscription(db, tenant.id)
    assert sub is not None
    assert sub.plan_id == studio_plan.id


@patch("stripe.checkout.Session.retrieve")
def test_sync_payment_status_crm_plan_grants_zero_credits(mock_retrieve, db, tenant, user, crm_plan):
    mock_retrieve.return_value = _FakeStripeSession(
        tenant_id=tenant.id, user_id=user.id, plan_id=crm_plan.id, session_id="cs_crm_1", crm_type="monday"
    )

    result = BillingService.sync_payment_status(db, "cs_crm_1", "evt_crm_1")

    assert result["status"] == "success"
    assert result["credits_added"] == 0
    assert result["crm_type"] == "monday"

    db.refresh(tenant)
    assert tenant.credits == Decimal("0")


# ---------------------------------------------------------------------------
# handle_subscription_renewed
# ---------------------------------------------------------------------------


def test_handle_subscription_renewed_rolls_period_and_regrants_credits(db, tenant, user, studio_plan):
    now = datetime.now(timezone.utc)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        stripe_subscription_id="sub_renew_1",
        current_period_start=now - timedelta(days=30),
        current_period_end=now,
    )
    db.add(sub)
    tenant.credits = Decimal("5.00")
    db.commit()

    new_start = now
    new_end = now + timedelta(days=30)
    BillingService.handle_subscription_renewed(db, "sub_renew_1", new_start, new_end)

    db.refresh(sub)
    db.refresh(tenant)

    # SQLite drops tzinfo on round-trip; compare naive wall-clock values.
    assert sub.current_period_start.replace(tzinfo=None) == new_start.replace(tzinfo=None)
    assert sub.current_period_end.replace(tzinfo=None) == new_end.replace(tzinfo=None)
    assert tenant.credits == Decimal("5.00") + Decimal("50.00")


def test_handle_subscription_renewed_unknown_subscription_is_noop(db):
    # Should not raise for an unrecognized stripe_subscription_id.
    BillingService.handle_subscription_renewed(
        db, "sub_does_not_exist", datetime.now(timezone.utc), datetime.now(timezone.utc)
    )


def test_handle_subscription_renewed_same_invoice_twice_grants_credits_once(db, tenant, user, studio_plan):
    """Regression: Stripe redelivers the same `invoice.payment_succeeded` event (at-least-once
    delivery) — the second call with the same invoice_id must be a no-op."""
    now = datetime.now(timezone.utc)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        stripe_subscription_id="sub_renew_dup",
        current_period_start=now - timedelta(days=30),
        current_period_end=now,
    )
    db.add(sub)
    tenant.credits = Decimal("5.00")
    db.commit()

    new_start = now
    new_end = now + timedelta(days=30)

    BillingService.handle_subscription_renewed(
        db, "sub_renew_dup", new_start, new_end, invoice_id="in_dup_1"
    )
    BillingService.handle_subscription_renewed(
        db, "sub_renew_dup", new_start, new_end, invoice_id="in_dup_1"
    )

    db.refresh(tenant)
    # Only one grant despite two calls with the same invoice id.
    assert tenant.credits == Decimal("5.00") + Decimal("50.00")


def test_handle_subscription_renewed_same_period_via_checkout_and_invoice_grants_once(
    db, tenant, user, studio_plan
):
    """Regression: the tenant's FIRST invoice fires both `checkout.session.completed`
    (which grants monthly_credits via sync_payment_status) and `invoice.payment_succeeded`
    for that same period — the renewal path must not grant credits a second time for a
    period the subscription already has recorded."""
    now = datetime.now(timezone.utc)
    period_start = now
    period_end = now + timedelta(days=30)

    # Simulate what sync_payment_status's plan_purchase branch already did: it created
    # the subscription with this exact period recorded, and granted credits once.
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        stripe_subscription_id="sub_first_invoice",
        current_period_start=period_start,
        current_period_end=period_end,
    )
    db.add(sub)
    tenant.credits = Decimal("50.00")  # already granted once via checkout.session.completed
    db.commit()

    # The subscription's first invoice.payment_succeeded event arrives for the same period.
    BillingService.handle_subscription_renewed(
        db, "sub_first_invoice", period_start, period_end, invoice_id="in_first_1"
    )

    db.refresh(tenant)
    # Must NOT be double-granted.
    assert tenant.credits == Decimal("50.00")


def test_handle_subscription_renewed_new_period_still_grants_credits(db, tenant, user, studio_plan):
    """Sanity check: a genuine subsequent renewal (new period, new invoice) still grants
    credits — the idempotency guards must not block legitimate renewals."""
    now = datetime.now(timezone.utc)
    sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=studio_plan.id,
        crm_type=None,
        status="active",
        stripe_subscription_id="sub_ongoing",
        current_period_start=now - timedelta(days=60),
        current_period_end=now - timedelta(days=30),
    )
    db.add(sub)
    tenant.credits = Decimal("0")
    db.commit()

    # First renewal (period 1 -> period 2)
    BillingService.handle_subscription_renewed(
        db, "sub_ongoing", now - timedelta(days=30), now, invoice_id="in_period_2"
    )
    db.refresh(tenant)
    assert tenant.credits == Decimal("50.00")

    # Second, genuinely later renewal (period 2 -> period 3) with a different invoice id.
    BillingService.handle_subscription_renewed(
        db, "sub_ongoing", now, now + timedelta(days=30), invoice_id="in_period_3"
    )
    db.refresh(tenant)
    assert tenant.credits == Decimal("100.00")
