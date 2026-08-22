"""Minimum-purchase validation for POST /tenants/start-credit-checkout-session."""
import uuid
from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.api_v1.endpoints.tenant import start_credit_checkout_session
from app.models.tenant import Tenant
from app.models.user import User


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Credit Checkout Tenant {suffix}",
        schema_name=f"credit_checkout_{suffix}",
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
        id=uuid.uuid4(),
        email=f"checkout_{suffix}@example.com",
        first_name="Test",
        last_name="User",
        hashed_password="x",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def test_amount_below_minimum_rejected(db, tenant, user):
    with pytest.raises(HTTPException) as exc_info:
        start_credit_checkout_session(
            amount=4.99,
            current_user=user,
            admin_user=user,
            db=db,
        )
    assert exc_info.value.status_code == 400
    assert "Minimum credit purchase is $5" in exc_info.value.detail


def test_amount_at_minimum_boundary_passes_validation(db, tenant, user):
    # Stub Stripe so we only exercise the $5 minimum check, not real network calls.
    class _FakeSession:
        id = "cs_test_min"
        url = "https://checkout.stripe.test/cs_test_min"

    with patch("stripe.checkout.Session.create", return_value=_FakeSession()), patch(
        "app.services.stripe_service.StripeService.create_customer",
        return_value="cus_test_min",
    ):
        result = start_credit_checkout_session(
            amount=5.0,
            current_user=user,
            admin_user=user,
            db=db,
        )
    assert result is not None
