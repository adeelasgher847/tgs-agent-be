"""Stripe checkout fulfillment idempotency (DB-backed, not in-memory)."""
import uuid
from decimal import Decimal
from unittest.mock import patch

import pytest

from app.models.stripe_checkout_fulfillment import StripeCheckoutFulfillment
from app.models.tenant import Tenant
from app.services.billing_service import BillingService


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Billing Test Tenant {suffix}",
        schema_name=f"billing_test_{suffix}",
        status="active",
        credits=Decimal("100"),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


class _FakeStripeSession:
    payment_status = "paid"

    def __init__(self, tenant_id: uuid.UUID, session_id: str):
        self._data = {
            "id": session_id,
            "payment_status": "paid",
            "amount_total": 1000,
            "metadata": {
                "tenant_id": str(tenant_id),
                "purchase_type": "credit_purchase",
            },
        }

    def __getitem__(self, key: str):
        return self._data[key]


def _paid_credit_session(tenant_id: uuid.UUID, session_id: str = "cs_test_123"):
    return _FakeStripeSession(tenant_id, session_id)


@patch("stripe.checkout.Session.retrieve")
def test_sync_payment_credits_once(mock_retrieve, db, tenant):
    mock_retrieve.return_value = _paid_credit_session(tenant.id)

    first = BillingService.sync_payment_status(db, "cs_test_123", "evt_1")
    second = BillingService.sync_payment_status(db, "cs_test_123", "evt_2")

    assert first["status"] == "success"
    assert first["credits_added"] == 100
    assert second["status"] == "already_processed"

    db.refresh(tenant)
    assert tenant.credits == Decimal("200")

    rows = db.query(StripeCheckoutFulfillment).filter(
        StripeCheckoutFulfillment.checkout_session_id == "cs_test_123"
    ).all()
    assert len(rows) == 1


@patch("stripe.checkout.Session.retrieve")
def test_claim_checkout_session_race_safe(mock_retrieve, db, tenant):
    mock_retrieve.return_value = _paid_credit_session(tenant.id, "cs_race_1")

    assert BillingService._claim_checkout_session(db, "cs_race_1", "evt_a") is True
    db.commit()

    assert BillingService._claim_checkout_session(db, "cs_race_1", "evt_b") is False
