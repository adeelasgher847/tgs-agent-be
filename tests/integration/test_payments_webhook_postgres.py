"""
Payments webhook integration tests against real PostgreSQL.

Requires TEST_DATABASE_URL. Verifies that Stripe webhook events correctly update
PaymentRecord rows in real PostgreSQL — exercising UUID primary keys, Integer
numeric columns, nullable FK columns (call_id), and status-string transitions
that SQLite's loose type system cannot reliably catch.

Coverage:
  1.  test_succeeded_webhook_updates_status_and_card_details
  2.  test_failed_webhook_updates_status_to_failed
  3.  test_missing_stripe_signature_returns_403
  4.  test_invalid_stripe_signature_returns_403
  5.  test_unconfigured_webhook_secret_returns_500
  6.  test_record_null_call_id_persisted_correctly
  7.  test_list_payment_records_paginated
  8.  test_get_payment_record_by_pi_id
  9.  test_get_unknown_pi_id_returns_404
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.models.payment_record import PaymentRecord
from app.models.tenant import Tenant
from app.services.api_key_service import create_api_key
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]

_FAKE_WEBHOOK_SECRET = "whsec_test_pg_integration_secret_for_pytest"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def auth_workspace(pg_session):
    """Tenant + real API key committed in the Postgres test schema."""
    tenant = Tenant(
        name=f"PayPG-{uuid.uuid4().hex[:8]}",
        schema_name=f"pay_pg_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    pg_session.add(tenant)
    pg_session.commit()
    pg_session.refresh(tenant)

    _record, raw_key = create_api_key(
        pg_session, tenant_id=tenant.id, name="pay-integration-test"
    )
    return tenant, raw_key


def _auth_headers(raw_key: str, tenant_id: uuid.UUID) -> dict[str, str]:
    return {"x-api-key": raw_key, "x-workspace-id": str(tenant_id)}


def _make_succeeded_event(pi_id: str) -> dict:
    return {
        "type": "payment_intent.succeeded",
        "data": {
            "object": {
                "id": pi_id,
                "object": "payment_intent",
                "amount": 5000,
                "currency": "usd",
                "status": "succeeded",
                "charges": {
                    "data": [
                        {
                            "id": f"ch_{uuid.uuid4().hex[:16]}",
                            "payment_method_details": {
                                "card": {"brand": "visa", "last4": "4242"},
                                "type": "card",
                            },
                        }
                    ]
                },
            }
        },
    }


def _make_failed_event(pi_id: str) -> dict:
    return {
        "type": "payment_intent.payment_failed",
        "data": {
            "object": {
                "id": pi_id,
                "object": "payment_intent",
                "amount": 5000,
                "currency": "usd",
                "status": "requires_payment_method",
            }
        },
    }


@pytest.fixture()
def pending_payment(pg_session, auth_workspace):
    """A committed pending PaymentRecord visible to pg_client requests."""
    tenant, _ = auth_workspace
    pi_id = f"pi_test_pg_{uuid.uuid4().hex[:16]}"
    record = PaymentRecord(
        workspace_id=tenant.id,
        call_id=None,
        payment_intent_id=pi_id,
        amount_cents=5000,
        currency="usd",
        description="Integration test payment",
        status="pending",
    )
    pg_session.add(record)
    pg_session.commit()
    pg_session.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Webhook endpoint — no API key required (excluded from ApiKeyMiddleware)
# ---------------------------------------------------------------------------


class TestStripeWebhookPostgres:
    """Tests that the stripe-webhook endpoint correctly mutates PG rows."""

    def test_succeeded_webhook_updates_status_and_card_details(
        self, pg_client, pg_session, pending_payment
    ):
        pi_id = pending_payment.payment_intent_id
        event = _make_succeeded_event(pi_id)

        with patch.object(settings, "STRIPE_INCALL_WEBHOOK_SECRET", _FAKE_WEBHOOK_SECRET):
            with patch(
                "app.services.stripe_service.StripeService.construct_payment_webhook_event",
                return_value=event,
            ):
                resp = pg_client.post(
                    "/api/v1/payments/stripe-webhook",
                    content=b'{"mocked": true}',
                    headers={"stripe-signature": "t=1,v1=mocksig"},
                )

        assert resp.status_code == 204, resp.text

        # Query fresh — the pg_client session committed the update
        fresh = pg_session.execute(
            select(PaymentRecord).where(PaymentRecord.payment_intent_id == pi_id)
        ).scalar_one()
        assert fresh.status == "succeeded"
        assert fresh.card_last4 == "4242"
        assert fresh.card_brand == "visa"

    def test_failed_webhook_updates_status_to_failed(
        self, pg_client, pg_session, auth_workspace
    ):
        tenant, _ = auth_workspace
        pi_id = f"pi_test_pg_fail_{uuid.uuid4().hex[:12]}"
        pg_session.add(
            PaymentRecord(
                workspace_id=tenant.id,
                payment_intent_id=pi_id,
                amount_cents=2000,
                currency="usd",
                status="pending",
            )
        )
        pg_session.commit()

        event = _make_failed_event(pi_id)

        with patch.object(settings, "STRIPE_INCALL_WEBHOOK_SECRET", _FAKE_WEBHOOK_SECRET):
            with patch(
                "app.services.stripe_service.StripeService.construct_payment_webhook_event",
                return_value=event,
            ):
                resp = pg_client.post(
                    "/api/v1/payments/stripe-webhook",
                    content=b'{"mocked": true}',
                    headers={"stripe-signature": "t=1,v1=mocksig"},
                )

        assert resp.status_code == 204, resp.text

        fresh = pg_session.execute(
            select(PaymentRecord).where(PaymentRecord.payment_intent_id == pi_id)
        ).scalar_one()
        assert fresh.status == "failed"
        assert fresh.card_last4 is None  # no card details on failure

    def test_missing_stripe_signature_returns_403(self, pg_client):
        # 403 is returned before the webhook-secret check, so no need to mock it
        resp = pg_client.post(
            "/api/v1/payments/stripe-webhook",
            content=b'{"type": "payment_intent.succeeded"}',
        )
        assert resp.status_code == 403

    def test_invalid_stripe_signature_returns_403(self, pg_client):
        # Real stripe.Webhook.construct_event verifies HMAC locally — no network call.
        # An invalid signature against our _FAKE_WEBHOOK_SECRET must yield 403.
        with patch.object(settings, "STRIPE_INCALL_WEBHOOK_SECRET", _FAKE_WEBHOOK_SECRET):
            resp = pg_client.post(
                "/api/v1/payments/stripe-webhook",
                content=b'{"type": "payment_intent.succeeded"}',
                headers={"stripe-signature": "t=1000,v1=invalidsignature_definitely_wrong"},
            )
        assert resp.status_code == 403

    def test_unconfigured_webhook_secret_returns_500(self, pg_client):
        with patch.object(settings, "STRIPE_INCALL_WEBHOOK_SECRET", ""):
            resp = pg_client.post(
                "/api/v1/payments/stripe-webhook",
                content=b'{"type": "payment_intent.succeeded"}',
                headers={"stripe-signature": "t=1,v1=anysig"},
            )
        assert resp.status_code == 500

    def test_record_null_call_id_persisted_correctly(
        self, pg_client, pg_session, auth_workspace
    ):
        """Nullable UUID FK (call_id=None) must be stored as SQL NULL, not empty string."""
        tenant, _ = auth_workspace
        pi_id = f"pi_test_pg_null_{uuid.uuid4().hex[:12]}"
        pg_session.add(
            PaymentRecord(
                workspace_id=tenant.id,
                call_id=None,
                payment_intent_id=pi_id,
                amount_cents=100,
                currency="usd",
                status="pending",
            )
        )
        pg_session.commit()

        fresh = pg_session.execute(
            select(PaymentRecord).where(PaymentRecord.payment_intent_id == pi_id)
        ).scalar_one()
        assert fresh.call_id is None
        assert fresh.workspace_id == tenant.id
        assert isinstance(fresh.amount_cents, int)
        assert fresh.currency == "usd"

    def test_unknown_payment_intent_in_webhook_logs_warning_and_returns_204(
        self, pg_client
    ):
        """When the PI id in the event has no matching row the handler logs a warning
        but still returns 204 — Stripe must not retry on a missing record."""
        pi_id = f"pi_ghost_{uuid.uuid4().hex[:16]}"
        event = _make_succeeded_event(pi_id)

        with patch.object(settings, "STRIPE_INCALL_WEBHOOK_SECRET", _FAKE_WEBHOOK_SECRET):
            with patch(
                "app.services.stripe_service.StripeService.construct_payment_webhook_event",
                return_value=event,
            ):
                resp = pg_client.post(
                    "/api/v1/payments/stripe-webhook",
                    content=b'{"mocked": true}',
                    headers={"stripe-signature": "t=1,v1=mocksig"},
                )

        # Service returns False when no row found; router swallows and returns 204.
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# Query endpoints — require API key auth via pg_client
# ---------------------------------------------------------------------------


class TestPaymentQueryPostgres:
    """Tests list and detail endpoints against real PG rows."""

    def test_list_payment_records_paginated(
        self, pg_client, pg_session, auth_workspace
    ):
        tenant, raw_key = auth_workspace
        pi_ids = []
        for _ in range(3):
            pi_id = f"pi_list_pg_{uuid.uuid4().hex[:12]}"
            pi_ids.append(pi_id)
            pg_session.add(
                PaymentRecord(
                    workspace_id=tenant.id,
                    payment_intent_id=pi_id,
                    amount_cents=1000,
                    currency="usd",
                    status="succeeded",
                )
            )
        pg_session.commit()

        resp = pg_client.get(
            "/api/v1/payments",
            params={"page": 1, "per_page": 10},
            headers=_auth_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["total"] >= 3
        returned_ids = [item["payment_intent_id"] for item in body["items"]]
        for pi_id in pi_ids:
            assert pi_id in returned_ids

    def test_get_payment_record_by_pi_id(
        self, pg_client, pg_session, auth_workspace
    ):
        tenant, raw_key = auth_workspace
        pi_id = f"pi_get_pg_{uuid.uuid4().hex[:12]}"
        pg_session.add(
            PaymentRecord(
                workspace_id=tenant.id,
                payment_intent_id=pi_id,
                amount_cents=7500,
                currency="gbp",
                status="succeeded",
                card_last4="1234",
                card_brand="mastercard",
            )
        )
        pg_session.commit()

        resp = pg_client.get(
            f"/api/v1/payments/{pi_id}",
            headers=_auth_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["payment_intent_id"] == pi_id
        assert data["amount_cents"] == 7500
        assert data["currency"] == "gbp"
        assert data["card_last4"] == "1234"
        assert data["card_brand"] == "mastercard"

    def test_get_unknown_pi_id_returns_404(self, pg_client, auth_workspace):
        tenant, raw_key = auth_workspace
        resp = pg_client.get(
            "/api/v1/payments/pi_definitely_does_not_exist_zzz",
            headers=_auth_headers(raw_key, tenant.id),
        )
        assert resp.status_code == 404
