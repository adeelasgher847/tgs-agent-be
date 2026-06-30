"""
Tests for in-call Stripe payment endpoints.

Coverage:
  1. test_create_payment_session            — POST /payments/session (success)
  2. test_create_payment_session_missing_fields — POST /payments/session (validation error)
  3. test_stripe_webhook_valid_signature    — POST /payments/stripe-webhook (204 + DB updated)
  4. test_stripe_webhook_invalid_signature  — POST /payments/stripe-webhook (403)
  5. test_stripe_webhook_missing_signature  — POST /payments/stripe-webhook (403, no header)
  6. test_payment_records_updated_on_succeeded_webhook — card_last4, card_brand set
  7. test_payment_records_updated_on_failed_webhook    — status=failed
  8. test_list_payments_empty               — GET /payments (empty list)
  9. test_list_payments                     — GET /payments (records present)
 10. test_get_payment_detail               — GET /payments/{pi_id} (200)
 11. test_get_payment_detail_not_found     — GET /payments/{pi_id} (404)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.payment_record import PaymentRecord

# ---------------------------------------------------------------------------
# Client fixture — reuses the shared SQLite DB from conftest.py
# ---------------------------------------------------------------------------

client = TestClient(app)

# Auth headers that satisfy ApiKeyMiddleware for the test workspace
# (The conftest seeds a tenant but does not seed an API key row; we instead
# make the auth middleware see the workspace through JWT mock.)
# We bypass auth by monkeypatching get_workspace in the router's deps.

_FAKE_WORKSPACE_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_FAKE_PI_ID = "pi_test_3TxS1MOCK000000001"
_FAKE_CLIENT_SECRET = "pi_test_3TxS1MOCK000000001_secret_MockSecret"
_FAKE_WEBHOOK_SECRET = "whsec_test_mock_webhook_secret_for_pytest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stripe_signature(payload: bytes, secret: str, timestamp: int | None = None) -> str:
    """Build a valid Stripe-Signature header for the given payload."""
    ts = timestamp or int(time.time())
    signed_payload = f"{ts}.{payload.decode()}"
    mac = hmac.new(secret.encode(), signed_payload.encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def _make_payment_intent_event(pi_id: str, event_type: str) -> dict:
    """Build a minimal Stripe payment_intent.succeeded / payment_failed event dict."""
    return {
        "id": f"evt_test_{uuid.uuid4().hex[:16]}",
        "type": event_type,
        "data": {
            "object": {
                "id": pi_id,
                "object": "payment_intent",
                "amount": 5000,
                "currency": "usd",
                "status": "succeeded" if "succeeded" in event_type else "requires_payment_method",
                "charges": {
                    "data": [
                        {
                            "id": f"ch_test_{uuid.uuid4().hex[:16]}",
                            "payment_method_details": {
                                "card": {
                                    "brand": "visa",
                                    "last4": "4242",
                                },
                                "type": "card",
                            },
                        }
                    ]
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Workspace dependency override — avoids needing a seeded API key
# ---------------------------------------------------------------------------

from app.core.workspace import Workspace


def _fake_workspace() -> Workspace:
    return Workspace(
        id=_FAKE_WORKSPACE_ID,
        name="Test Workspace",
        status="active",
        is_active=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def override_auth():
    """Mock API key middleware to authorize requests."""
    async def _resolve(_key_hash, workspace_id):
        return {
            "api_key_id": str(uuid.uuid4()),
            "tenant_id": str(workspace_id),
            "key_is_active": True,
            "workspace": {
                "id": str(workspace_id),
                "name": "Test Workspace",
                "schema_name": "test_ws",
                "status": "active",
            },
        }

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield

def _auth_headers(ws_id=None):
    return {
        "x-api-key": "test_key",
        "x-workspace-id": str(ws_id or _FAKE_WORKSPACE_ID)
    }


@pytest.fixture()
def mock_stripe_intent():
    """Return a mock Stripe PaymentIntent object."""
    intent = MagicMock()
    intent.id = f"pi_test_mock_{uuid.uuid4().hex[:16]}"
    intent.client_secret = f"{intent.id}_secret_MockSecret"
    return intent


@pytest.fixture()
def seeded_payment_record(db):
    """Insert a pending PaymentRecord into the test DB and return it."""
    record = PaymentRecord(
        workspace_id=_FAKE_WORKSPACE_ID,
        call_id=None,
        payment_intent_id=f"pi_test_seed_{uuid.uuid4().hex[:16]}",
        amount_cents=5000,
        currency="usd",
        description="Test payment",
        status="pending",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ===========================================================================
# 1. POST /payments/session — success
# ===========================================================================

def test_create_payment_session(db, mock_stripe_intent):
    """Creates a PaymentIntent in Stripe test mode and persists a PaymentRecord."""
    with patch(
        "app.services.stripe_service.stripe.PaymentIntent.create",
        return_value=mock_stripe_intent,
    ):
        resp = client.post(
            "/api/v1/payments/session",
            headers=_auth_headers(),
            json={
                "amount_cents": 5000,
                "currency": "usd",
                "description": "Consultation fee",
            },
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    assert data["payment_intent_id"] == mock_stripe_intent.id
    assert data["client_secret"] == mock_stripe_intent.client_secret
    assert mock_stripe_intent.id in data["payment_url"]
    assert "client_secret" in data["payment_url"]
    assert "Ask them to complete the payment" in data["agent_context"]

    # DB record should be created
    record = (
        db.query(PaymentRecord)
        .filter(PaymentRecord.payment_intent_id == mock_stripe_intent.id)
        .first()
    )
    assert record is not None
    assert record.status == "pending"
    assert record.amount_cents == 5000
    assert record.currency == "usd"
    assert str(record.workspace_id) == str(_FAKE_WORKSPACE_ID)


# ===========================================================================
# 2. POST /payments/session — validation error (amount ≤ 0)
# ===========================================================================

def test_create_payment_session_invalid_amount(db):
    """Should return 400 when amount_cents is not positive."""
    resp = client.post(
        "/api/v1/payments/session",
        headers=_auth_headers(),
        json={"amount_cents": 0, "currency": "usd", "description": ""},
    )
    assert resp.status_code == 400


# ===========================================================================
# 3. POST /payments/stripe-webhook — valid signature → 204 + DB updated
# ===========================================================================

def test_stripe_webhook_valid_signature(db, seeded_payment_record):
    """Valid Stripe-Signature must yield 204 and update the PaymentRecord status."""
    event = _make_payment_intent_event(seeded_payment_record.payment_intent_id, "payment_intent.succeeded")
    payload = json.dumps(event).encode()
    sig = _make_stripe_signature(payload, _FAKE_WEBHOOK_SECRET)

    with patch(
        "app.services.stripe_service.stripe.Webhook.construct_event",
        return_value=event,
    ), patch.object(
        # Patch the setting so the router picks up our test secret
        type(app.state if hasattr(app, "state") else object()),
        "STRIPE_INCALL_WEBHOOK_SECRET",
        _FAKE_WEBHOOK_SECRET,
        create=True,
    ), patch(
        "app.core.config.settings.STRIPE_INCALL_WEBHOOK_SECRET",
        _FAKE_WEBHOOK_SECRET,
    ):
        resp = client.post(
            "/api/v1/payments/stripe-webhook",
            content=payload,
            headers={"stripe-signature": sig, "content-type": "application/json"},
        )

    assert resp.status_code == 204, resp.text

    db.expire(seeded_payment_record)
    db.refresh(seeded_payment_record)
    assert seeded_payment_record.status == "succeeded"


# ===========================================================================
# 4. POST /payments/stripe-webhook — invalid signature → 403
# ===========================================================================

def test_stripe_webhook_invalid_signature(db):
    """Garbage Stripe-Signature must be rejected with 403."""
    import stripe as stripe_lib

    payload = b'{"type":"payment_intent.succeeded","data":{"object":{"id":"pi_fake"}}}'

    with patch(
        "app.services.stripe_service.stripe.Webhook.construct_event",
        side_effect=stripe_lib.SignatureVerificationError(
            "No signatures found matching the expected signature for payload",
            "bad-header",
        ),
    ), patch(
        "app.core.config.settings.STRIPE_INCALL_WEBHOOK_SECRET",
        _FAKE_WEBHOOK_SECRET,
    ):
        resp = client.post(
            "/api/v1/payments/stripe-webhook",
            content=payload,
            headers={
                "stripe-signature": "t=1000,v1=invalidsig",
                "content-type": "application/json",
            },
        )

    assert resp.status_code == 403, resp.text


# ===========================================================================
# 5. POST /payments/stripe-webhook — missing Stripe-Signature header → 403
# ===========================================================================

def test_stripe_webhook_missing_signature(db):
    """Request without Stripe-Signature must be rejected with 403."""
    with patch(
        "app.core.config.settings.STRIPE_INCALL_WEBHOOK_SECRET",
        _FAKE_WEBHOOK_SECRET,
    ):
        resp = client.post(
            "/api/v1/payments/stripe-webhook",
            content=b'{}',
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 403, resp.text


# ===========================================================================
# 6. payment_intent.succeeded → card_last4 + card_brand stored
# ===========================================================================

def test_payment_records_updated_on_succeeded_webhook(db, seeded_payment_record):
    """Webhook handler must persist card_last4='4242' and card_brand='visa'."""
    event = _make_payment_intent_event(seeded_payment_record.payment_intent_id, "payment_intent.succeeded")

    with patch(
        "app.services.stripe_service.stripe.Webhook.construct_event",
        return_value=event,
    ), patch(
        "app.core.config.settings.STRIPE_INCALL_WEBHOOK_SECRET",
        _FAKE_WEBHOOK_SECRET,
    ):
        payload = json.dumps(event).encode()
        sig = _make_stripe_signature(payload, _FAKE_WEBHOOK_SECRET)
        resp = client.post(
            "/api/v1/payments/stripe-webhook",
            content=payload,
            headers={"stripe-signature": sig, "content-type": "application/json"},
        )

    assert resp.status_code == 204

    db.expire(seeded_payment_record)
    db.refresh(seeded_payment_record)
    assert seeded_payment_record.card_last4 == "4242"
    assert seeded_payment_record.card_brand == "visa"
    assert seeded_payment_record.status == "succeeded"


# ===========================================================================
# 7. payment_intent.payment_failed → status = "failed"
# ===========================================================================

def test_payment_records_updated_on_failed_webhook(db, seeded_payment_record):
    """Webhook handler must set status='failed' on payment_intent.payment_failed."""
    event = _make_payment_intent_event(seeded_payment_record.payment_intent_id, "payment_intent.payment_failed")

    with patch(
        "app.services.stripe_service.stripe.Webhook.construct_event",
        return_value=event,
    ), patch(
        "app.core.config.settings.STRIPE_INCALL_WEBHOOK_SECRET",
        _FAKE_WEBHOOK_SECRET,
    ):
        payload = json.dumps(event).encode()
        sig = _make_stripe_signature(payload, _FAKE_WEBHOOK_SECRET)
        resp = client.post(
            "/api/v1/payments/stripe-webhook",
            content=payload,
            headers={"stripe-signature": sig, "content-type": "application/json"},
        )

    assert resp.status_code == 204

    db.expire(seeded_payment_record)
    db.refresh(seeded_payment_record)
    assert seeded_payment_record.status == "failed"


# ===========================================================================
# 8. GET /payments — empty list
# ===========================================================================

def test_list_payments_empty(db):
    """Workspace with no payment records should return empty paginated list."""
    empty_ws_id = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    resp = client.get("/api/v1/payments", headers=_auth_headers(empty_ws_id))

    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["total"] == 0
    assert body["items"] == []


# ===========================================================================
# 9. GET /payments — returns records
# ===========================================================================

def test_list_payments(db, seeded_payment_record):
    """Should return the seeded PaymentRecord in the list."""
    resp = client.get("/api/v1/payments", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["total"] >= 1
    pi_ids = [item["payment_intent_id"] for item in body["items"]]
    assert seeded_payment_record.payment_intent_id in pi_ids


# ===========================================================================
# 10. GET /payments/{payment_intent_id} — 200
# ===========================================================================

def test_get_payment_detail(db, seeded_payment_record):
    """Should return the PaymentRecord for a known payment_intent_id."""
    resp = client.get(f"/api/v1/payments/{seeded_payment_record.payment_intent_id}", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["payment_intent_id"] == seeded_payment_record.payment_intent_id
    assert data["amount_cents"] == 5000
    assert data["currency"] == "usd"
    assert data["status"] == "pending"


# ===========================================================================
# 11. GET /payments/{payment_intent_id} — 404 for unknown PI
# ===========================================================================

def test_get_payment_detail_not_found(db):
    """Should return 404 for a PaymentIntent ID that does not exist."""
    resp = client.get("/api/v1/payments/pi_nonexistent_abc123", headers=_auth_headers())
    assert resp.status_code == 404


# ===========================================================================
# 12. POST /payments/session — dynamic context injection
# ===========================================================================

def test_create_payment_session_injects_metadata(db, mock_stripe_intent):
    """Creating a payment session must inject the payment page url as a prompt addendum in the CallSession metadata."""
    from app.models.call_session import CallSession
    from datetime import datetime

    # Seed a CallSession first
    call_session = CallSession(
        user_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        tenant_id=_FAKE_WORKSPACE_ID,
        start_time=datetime.utcnow(),
        status="active",
        call_type="inbound",
        call_metadata={}
    )
    db.add(call_session)
    db.commit()
    db.refresh(call_session)

    with patch(
        "app.services.stripe_service.stripe.PaymentIntent.create",
        return_value=mock_stripe_intent,
    ):
        resp = client.post(
            "/api/v1/payments/session",
            headers=_auth_headers(),
            json={
                "call_id": str(call_session.id),
                "amount_cents": 5000,
                "currency": "usd",
                "description": "Consultation fee",
            },
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]

    # Retrieve and verify CallSession metadata
    db.expire(call_session)
    db.refresh(call_session)
    metadata = call_session.call_metadata or {}
    vdc = metadata.get("voice_dynamic_context") or {}
    assert "system_prompt_addendum" in vdc
    assert data["payment_url"] in vdc["system_prompt_addendum"]
    assert "Ask them to complete the payment" in vdc["system_prompt_addendum"]

    # Cleanup seeded call_session
    db.delete(call_session)
    db.commit()
