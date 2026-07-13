# Stripe Test Cards — In-Call Payment QA

Use these cards when testing in-call payments against the Stripe **test mode** environment (`sk_test_...` key).

> ⚠️ Test cards only work with a `sk_test_` key. Never use real card numbers in test mode.

---

## Standard Test Cards

| Card Number | Scenario | CVC | Expiry |
|---|---|---|---|
| `4242 4242 4242 4242` | ✅ Payment succeeds | Any 3 digits | Any future date |
| `4000 0000 0000 0002` | ❌ Card declined (`card_declined`) | Any 3 digits | Any future date |
| `4000 0025 0000 3155` | 🔐 3D Secure required (redirect flow) | Any 3 digits | Any future date |
| `4000 0000 0000 9995` | ❌ Insufficient funds (`insufficient_funds`) | Any 3 digits | Any future date |
| `4000 0000 0000 0069` | ❌ Expired card (`expired_card`) | Any 3 digits | Any past date |
| `4000 0000 0000 0127` | ❌ Incorrect CVC (`incorrect_cvc`) | `999` | Any future date |

---

## Testing the Payment Flow

### Step 1 — Create a payment session

```bash
curl -X POST https://your-api.com/api/v1/payments/session \
  -H "x-api-key: YOUR_API_KEY" \
  -H "x-workspace-id: YOUR_WORKSPACE_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "00000000-0000-0000-0000-000000000001",
    "amount_cents": 5000,
    "currency": "usd",
    "description": "Consultation fee"
  }'
```

Expected response:
```json
{
  "data": {
    "payment_intent_id": "pi_test_...",
    "client_secret": "pi_test_..._secret_...",
    "payment_url": "https://pay.yourdomain.com/pay/pi_test_...?client_secret=...",
    "agent_context": "A payment page has been sent to the caller at ... Ask them to complete the payment."
  }
}
```

### Step 2 — Trigger the success webhook (Stripe CLI)

```bash
# Forward Stripe webhook events to your local server
stripe listen --forward-to localhost:8000/api/v1/payments/stripe-webhook

# Trigger a payment_intent.succeeded event
stripe trigger payment_intent.succeeded
```

### Step 3 — Verify record updated

```bash
curl https://your-api.com/api/v1/payments/pi_test_... \
  -H "x-api-key: YOUR_API_KEY" \
  -H "x-workspace-id: YOUR_WORKSPACE_ID"
```

Expected:
```json
{
  "data": {
    "status": "succeeded",
    "card_last4": "4242",
    "card_brand": "visa"
  }
}
```

---

## Webhook Event Reference

| Event | Expected behaviour |
|---|---|
| `payment_intent.succeeded` | `status` → `succeeded`; `card_last4`, `card_brand` populated |
| `payment_intent.payment_failed` | `status` → `failed` |

---

## Stripe CLI Quick Reference

```bash
# Install
brew install stripe/stripe-cli/stripe   # macOS
# or download from https://github.com/stripe/stripe-cli/releases

# Login
stripe login

# Listen & forward to local
stripe listen --forward-to localhost:8000/api/v1/payments/stripe-webhook

# Manually trigger events
stripe trigger payment_intent.succeeded
stripe trigger payment_intent.payment_failed

# Replay a specific event by ID
stripe events resend evt_...
```

---

## Environment Variables Required

```dotenv
STRIPE_SECRET_KEY=sk_test_...            # Test mode secret key
STRIPE_INCALL_WEBHOOK_SECRET=whsec_...   # From Stripe Dashboard → Webhooks → in-call endpoint
PAYMENT_PAGE_BASE_URL=https://pay.yourdomain.com
```

> **Sprint 6 note**: `STRIPE_SECRET_KEY` must be rotated from `sk_test_` to `sk_live_` only after the Sprint 6 security review. Do not promote to production until that review is complete.
