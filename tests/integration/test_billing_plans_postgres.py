"""Studio/Agency plan-aware subscription integrity, against real PostgreSQL.

`Subscription.uq_tenant_core_subscription` is a *partial* unique index
(`postgresql_where="crm_type IS NULL AND tenant_id IS NOT NULL"`). SQLite has
no `sqlite_where` fallback declared for this particular index (unlike, say,
`Tenant.uq_tenant_name_active`), so under the SQLite unit-test fixture it
silently compiles down to a plain unique index on `tenant_id` — stricter than
production, since it would also reject a legitimate CRM-addon row sharing a
tenant_id. Real Postgres semantics can only be verified against a real engine.

Requires TEST_DATABASE_URL. Skipped otherwise.

Coverage:
  1. test_second_core_subscription_for_same_tenant_rejected
  2. test_crm_addon_subscription_coexists_with_core_subscription_same_tenant
  3. test_core_subscription_allowed_for_different_tenants
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.user import User
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]


@pytest.fixture()
def tenant(pg_session):
    t = Tenant(
        name=f"PG Plan Tenant {uuid.uuid4().hex[:8]}",
        schema_name=f"pg_plan_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    pg_session.add(t)
    pg_session.commit()
    pg_session.refresh(t)
    return t


@pytest.fixture()
def user(pg_session, tenant):
    u = User(
        email=f"pg_plan_{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        first_name="PG",
        last_name="Tester",
        current_tenant_id=tenant.id,
    )
    pg_session.add(u)
    pg_session.commit()
    pg_session.refresh(u)
    return u


@pytest.fixture()
def core_plan(pg_session):
    p = Plan(
        name=f"studio_{uuid.uuid4().hex[:8]}",
        display_name="Studio",
        price_monthly=9900,
        crm_type=None,
        included_minutes=100,
        monthly_credits=Decimal("50.00"),
        is_active=True,
    )
    pg_session.add(p)
    pg_session.commit()
    pg_session.refresh(p)
    return p


@pytest.fixture()
def crm_plan(pg_session):
    p = Plan(
        name=f"monday_{uuid.uuid4().hex[:8]}",
        display_name="Monday CRM Addon",
        price_monthly=1900,
        crm_type="monday",
        is_active=True,
    )
    pg_session.add(p)
    pg_session.commit()
    pg_session.refresh(p)
    return p


def test_second_core_subscription_for_same_tenant_rejected(pg_session, tenant, user, core_plan):
    first = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=core_plan.id,
        crm_type=None,
        status="active",
    )
    pg_session.add(first)
    pg_session.commit()

    second = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=core_plan.id,
        crm_type=None,
        status="active",
    )
    pg_session.add(second)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()


def test_crm_addon_subscription_coexists_with_core_subscription_same_tenant(
    pg_session, tenant, user, core_plan, crm_plan
):
    """A tenant's core (crm_type IS NULL) subscription and a legacy
    user+crm_type CRM-addon row (tenant_id left NULL) must be able to coexist
    without tripping the partial unique index."""
    core_sub = Subscription(
        user_id=user.id,
        tenant_id=tenant.id,
        plan_id=core_plan.id,
        crm_type=None,
        status="active",
    )
    pg_session.add(core_sub)
    pg_session.commit()

    crm_sub = Subscription(
        user_id=user.id,
        tenant_id=None,
        plan_id=crm_plan.id,
        crm_type="monday",
        status="active",
    )
    pg_session.add(crm_sub)
    pg_session.commit()  # must not raise

    assert core_sub.id != crm_sub.id


def test_core_subscription_allowed_for_different_tenants(pg_session, tenant, user, core_plan):
    other_tenant = Tenant(
        name=f"PG Plan Tenant B {uuid.uuid4().hex[:8]}",
        schema_name=f"pg_plan_b_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    pg_session.add(other_tenant)
    pg_session.commit()
    pg_session.refresh(other_tenant)

    sub_a = Subscription(
        user_id=user.id, tenant_id=tenant.id, plan_id=core_plan.id, crm_type=None, status="active"
    )
    sub_b = Subscription(
        user_id=user.id, tenant_id=other_tenant.id, plan_id=core_plan.id, crm_type=None, status="active"
    )
    pg_session.add_all([sub_a, sub_b])
    pg_session.commit()  # must not raise

    assert sub_a.id != sub_b.id
