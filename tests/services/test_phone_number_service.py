"""Studio/Agency `free_phone_numbers` allowance — paid overage, not a hard cap.

Coverage:
  - purchase_phone_number / import_twilio_phone_number, once a tenant is at/over
    its plan's free_phone_numbers allowance, charge a flat $2 (2 credits) overage
    fee and still succeed if the tenant has enough credit balance.
  - If the tenant's credit balance can't cover the $2 overage, the purchase/import
    is rejected (HTTP 402) *before* any Twilio call is made, and no credits move.
  - No charge/cap at all when the plan field is NULL or there's no active plan —
    purchases remain free in that case.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.models.phone_number import PhoneNumber
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.user import User
from app.services.phone_number_service import phone_number_service


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"PhoneCap Tenant {suffix}",
        schema_name=f"phonecap_{suffix}",
        status="active",
        credits=Decimal("0"),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _activate_plan(db, tenant, *, free_phone_numbers):
    suffix = uuid.uuid4().hex[:8]
    plan = Plan(
        name=f"plan_{suffix}",
        display_name="Plan",
        price_monthly=9900,
        crm_type=None,
        free_phone_numbers=free_phone_numbers,
        is_active=True,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    user = User(
        email=f"owner_{suffix}@test.com",
        hashed_password="X",
        first_name="Owner",
        last_name="Test",
        current_tenant_id=tenant.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

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
    return plan


def _existing_number(db, tenant, n=1):
    for i in range(n):
        pn = PhoneNumber(
            phone_number=f"+1555{uuid.uuid4().hex[:7]}",
            tenant_id=tenant.id,
            status="active",
            provider="twilio",
        )
        db.add(pn)
    db.commit()


# ---------------------------------------------------------------------------
# purchase_phone_number
# ---------------------------------------------------------------------------


def test_purchase_phone_number_overage_charged_when_credits_sufficient(db, tenant):
    _activate_plan(db, tenant, free_phone_numbers=1)
    _existing_number(db, tenant, n=1)  # already at the free allowance of 1
    tenant.credits = Decimal("5.00")
    db.commit()

    with patch(
        "app.services.twilio_service.twilio_service.purchase_phone_number",
        return_value={"phone_number": "+15551234567", "sid": "PN123"},
    ) as mock_purchase:
        pn = phone_number_service.purchase_phone_number(
            db, phone_number="+15551234567", tenant_id=tenant.id
        )

    mock_purchase.assert_called_once()
    assert pn.phone_number == "+15551234567"

    db.refresh(tenant)
    assert tenant.credits == Decimal("3.00")


def test_purchase_phone_number_overage_rejected_when_credits_insufficient_twilio_never_called(db, tenant):
    _activate_plan(db, tenant, free_phone_numbers=1)
    _existing_number(db, tenant, n=1)  # already at the free allowance of 1
    tenant.credits = Decimal("1.00")  # not enough to cover the $2 overage
    db.commit()

    with patch("app.services.twilio_service.twilio_service.purchase_phone_number") as mock_purchase:
        with pytest.raises(HTTPException) as exc_info:
            phone_number_service.purchase_phone_number(
                db, phone_number="+15551234567", tenant_id=tenant.id
            )

    assert exc_info.value.status_code == 402
    assert "Insufficient balance" in exc_info.value.detail
    mock_purchase.assert_not_called()

    db.refresh(tenant)
    assert tenant.credits == Decimal("1.00")  # untouched


def test_purchase_phone_number_within_free_allowance_no_charge(db, tenant):
    _activate_plan(db, tenant, free_phone_numbers=3)
    _existing_number(db, tenant, n=1)  # still within the free allowance of 3
    tenant.credits = Decimal("0")
    db.commit()

    with patch(
        "app.services.twilio_service.twilio_service.purchase_phone_number",
        return_value={"phone_number": "+15551234568", "sid": "PN124"},
    ) as mock_purchase:
        pn = phone_number_service.purchase_phone_number(
            db, phone_number="+15551234568", tenant_id=tenant.id
        )

    mock_purchase.assert_called_once()
    assert pn.phone_number == "+15551234568"

    db.refresh(tenant)
    assert tenant.credits == Decimal("0")  # no overage charge


def test_purchase_phone_number_no_cap_when_plan_field_null(db, tenant):
    _activate_plan(db, tenant, free_phone_numbers=None)
    _existing_number(db, tenant, n=5)
    tenant.credits = Decimal("0")
    db.commit()

    with patch(
        "app.services.twilio_service.twilio_service.purchase_phone_number",
        return_value={"phone_number": "+15551234569", "sid": "PN125"},
    ) as mock_purchase:
        pn = phone_number_service.purchase_phone_number(
            db, phone_number="+15551234569", tenant_id=tenant.id
        )

    mock_purchase.assert_called_once()
    assert pn.phone_number == "+15551234569"

    db.refresh(tenant)
    assert tenant.credits == Decimal("0")  # unlimited plan — never charged


def test_purchase_phone_number_no_cap_when_no_active_plan(db, tenant):
    # No subscription at all for this tenant.
    _existing_number(db, tenant, n=5)

    with patch(
        "app.services.twilio_service.twilio_service.purchase_phone_number",
        return_value={"phone_number": "+15559876543", "sid": "PN456"},
    ) as mock_purchase:
        pn = phone_number_service.purchase_phone_number(
            db, phone_number="+15559876543", tenant_id=tenant.id
        )

    mock_purchase.assert_called_once()
    assert pn.phone_number == "+15559876543"


# ---------------------------------------------------------------------------
# import_twilio_phone_number
# ---------------------------------------------------------------------------


def test_import_twilio_phone_number_overage_charged_when_credits_sufficient(db, tenant):
    _activate_plan(db, tenant, free_phone_numbers=2)
    _existing_number(db, tenant, n=2)  # already at the free allowance of 2
    tenant.credits = Decimal("10.00")
    db.commit()

    with patch("app.services.twilio_service.twilio_service.get_client_with_credentials") as mock_get_client:
        mock_client = mock_get_client.return_value
        mock_number = type(
            "OwnedNumber", (), {"sid": "PNIMPORT1", "capabilities": {"voice": True}}
        )()
        mock_client.incoming_phone_numbers.list.return_value = [mock_number]

        with patch(
            "app.services.twilio_service.twilio_service.update_number_configuration_with_credentials"
        ):
            pn = phone_number_service.import_twilio_phone_number(
                db,
                phone_number="+15551112222",
                label=None,
                tenant_id=tenant.id,
                twilio_account_sid="ACxxx",
                twilio_auth_token="authxxx",
            )

    assert pn.phone_number == "+15551112222"
    db.refresh(tenant)
    assert tenant.credits == Decimal("8.00")


def test_import_twilio_phone_number_overage_rejected_when_credits_insufficient_twilio_never_called(db, tenant):
    _activate_plan(db, tenant, free_phone_numbers=2)
    _existing_number(db, tenant, n=2)  # already at the free allowance of 2
    tenant.credits = Decimal("0.50")  # not enough to cover the $2 overage
    db.commit()

    with patch("app.services.twilio_service.twilio_service.get_client_with_credentials") as mock_client:
        with pytest.raises(HTTPException) as exc_info:
            phone_number_service.import_twilio_phone_number(
                db,
                phone_number="+15551112222",
                label=None,
                tenant_id=tenant.id,
                twilio_account_sid="ACxxx",
                twilio_auth_token="authxxx",
            )

    assert exc_info.value.status_code == 402
    assert "Insufficient balance" in exc_info.value.detail
    mock_client.assert_not_called()

    db.refresh(tenant)
    assert tenant.credits == Decimal("0.50")  # untouched
