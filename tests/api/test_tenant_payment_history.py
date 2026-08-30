"""
Regression coverage for GET /tenants/payment-history
(app/api/api_v1/endpoints/tenant.py::get_payment_history).

Two real production bugs, both from stripe-python 15.x's StripeObject no
longer behaving like the older dict-subclassing versions this endpoint was
apparently written against:

1. `payment_intent.last_payment_error.get("message", ...)` -- StripeObject
   no longer implements dict-style `.get()`; accessing `.get` falls through
   to `__getattr__`, which treats "get" as a data-key lookup, fails, and
   raises AttributeError. Fixed to `getattr(..., "message", ...)`.
2. `invoice.amount_total` -- Invoice's top-level "total after discounts and
   taxes" field is `.total`; `amount_total` only exists nested inside
   `invoice.shipping_cost` (a sub-object), not on the Invoice itself. This
   was a copy-paste from the Checkout Session block above (which DOES have
   a top-level `amount_total`) and raised AttributeError on every real
   invoice. Fixed to `invoice.total`.

The stand-ins below deliberately do NOT support dict-style `.get()` and do
NOT define `amount_total`, mirroring real stripe-python StripeObject
behavior closely enough to have caught both bugs (a MagicMock would not,
since it auto-creates any attribute access).
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.api.api_v1.endpoints.tenant import get_payment_history
from app.models.tenant import Tenant
from app.models.user import User


class _StripeObjectStub:
    """Mimics real stripe-python StripeObject: attribute access for a
    declared field works, but there is no dict-style `.get()` method and
    undeclared attributes raise AttributeError (not silently return None
    like MagicMock would)."""

    def __init__(self, **fields):
        self._fields = fields

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Payment History Tenant {suffix}",
        schema_name=f"payment_hist_{suffix}",
        status="active",
        credits=Decimal(0),
        stripe_customer_id="cus_test_123",
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
        email=f"payment_hist_{suffix}@example.com",
        first_name="Test",
        last_name="User",
        hashed_password="x",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _fake_checkout_sessions(payment_status="paid", with_failed_intent=False):
    session = _StripeObjectStub(
        id="cs_test_1",
        payment_status=payment_status,
        amount_total=9900,
        currency="usd",
        created=1735689600,
        payment_intent="pi_test_1" if with_failed_intent else None,
        subscription=None,
    )
    return SimpleNamespace(data=[session])


def _fake_invoices(total_cents=9900, attempt_count=0, status="paid"):
    invoice = _StripeObjectStub(
        id="in_test_1",
        status=status,
        total=total_cents,
        currency="usd",
        created=1735689600,
        payment_intent=None,
        subscription=None,
        invoice_pdf="https://invoice.stripe.test/in_test_1.pdf",
        period_start=1735689600,
        period_end=1738368000,
        attempt_count=attempt_count,
    )
    return SimpleNamespace(data=[invoice])


class TestPaymentHistoryStripeObjectCompatibility:
    def test_invoice_uses_total_not_amount_total(self, db, tenant, user):
        """Regression: invoice.amount_total does not exist on a real
        Invoice -- this must not raise, and must read the correct total."""
        with patch(
            "stripe.checkout.Session.list",
            return_value=_fake_checkout_sessions(),
        ), patch(
            "stripe.Invoice.list",
            return_value=_fake_invoices(total_cents=15000),
        ):
            result = get_payment_history(current_user=user, db=db)

        invoice_entries = [
            e for e in result.data["payment_history"] if e["type"] == "invoice"
        ]
        assert len(invoice_entries) == 1
        assert invoice_entries[0]["amount_total"] == 150.0
        assert invoice_entries[0]["amount_total_cents"] == 15000

    def test_failed_checkout_session_reads_failure_reason_via_getattr(
        self, db, tenant, user
    ):
        """Regression: last_payment_error.get(...) raises AttributeError on
        a real StripeObject -- this must not raise, and must read the
        failure message correctly via getattr."""
        last_payment_error = _StripeObjectStub(message="Your card was declined.")
        payment_intent = _StripeObjectStub(last_payment_error=last_payment_error)

        with patch(
            "stripe.checkout.Session.list",
            return_value=_fake_checkout_sessions(
                payment_status="unpaid", with_failed_intent=True
            ),
        ), patch(
            "stripe.PaymentIntent.retrieve", return_value=payment_intent
        ), patch(
            "stripe.Invoice.list", return_value=SimpleNamespace(data=[])
        ):
            result = get_payment_history(current_user=user, db=db)

        session_entries = [
            e for e in result.data["payment_history"] if e["type"] == "checkout_session"
        ]
        assert len(session_entries) == 1
        assert session_entries[0]["failure_reason"] == "Your card was declined."

    def test_successful_payment_history_has_no_stripe_errors_logged(
        self, db, tenant, user, caplog
    ):
        """End-to-end guardrail: a normal successful payment/invoice fetch
        must produce zero 'Error getting checkout sessions'/'Error getting
        invoices' log lines -- both bugs previously fired on every single
        real invoice and on any failed checkout session."""
        with patch(
            "stripe.checkout.Session.list",
            return_value=_fake_checkout_sessions(payment_status="paid"),
        ), patch(
            "stripe.Invoice.list",
            return_value=_fake_invoices(status="paid"),
        ):
            result = get_payment_history(current_user=user, db=db)

        assert "Error getting checkout sessions" not in caplog.text
        assert "Error getting invoices" not in caplog.text
        assert len(result.data["payment_history"]) == 2
        assert result.data["summary"]["total_payments"] == 2

    def test_no_stripe_customer_returns_empty_history(self, db, user, tenant):
        tenant.stripe_customer_id = None
        db.commit()

        result = get_payment_history(current_user=user, db=db)

        assert result.data["payment_history"] == []
