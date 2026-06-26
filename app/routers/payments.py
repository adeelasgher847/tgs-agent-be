"""
In-call Stripe payment endpoints.

Routes:
  POST   /api/v1/payments/session          — create PaymentIntent + PaymentRecord
  POST   /api/v1/payments/stripe-webhook   — Stripe webhook (no API-key auth; sig verified)
  GET    /api/v1/payments                  — list payment records (paginated)
  GET    /api/v1/payments/{pi_id}          — single payment record detail

Security notes:
  - The /stripe-webhook endpoint is excluded from ApiKeyMiddleware via _SKIP_PREFIXES.
    Authentication is performed by verifying the Stripe-Signature header with
    stripe.webhooks.construct_event() using STRIPE_INCALL_WEBHOOK_SECRET.
  - The Stripe secret key (sk_test_ / sk_live_) is NEVER returned to the browser.
    The client_secret returned in /session is intentionally safe to expose —
    it can only be used to confirm a specific PaymentIntent, not to read account data.
"""
from __future__ import annotations

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_workspace
from app.core.config import settings
from app.core.logger import logger
from app.core.workspace import Workspace
from app.schemas.base import SuccessResponse
from app.schemas.payment import (
    CreatePaymentSessionRequest,
    CreatePaymentSessionResponse,
    PaginatedPaymentResponse,
    PaymentRecordOut,
)
from app.services.payment_service import PaymentService
from app.services.stripe_service import StripeService
from app.utils.response import create_success_response

router = APIRouter()

# ---------------------------------------------------------------------------
# POST /session — create PaymentIntent
# ---------------------------------------------------------------------------

@router.post(
    "/session",
    response_model=SuccessResponse[CreatePaymentSessionResponse],
    summary="Create an in-call payment session",
    description=(
        "Creates a Stripe PaymentIntent and returns a payment URL the agent can "
        "share with the caller. The `agent_context` field contains the ready-made "
        "sentence to inject into the agent's prompt."
    ),
)
def create_payment_session(
    body: CreatePaymentSessionRequest,
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_workspace),
):
    try:
        response, _agent_ctx = PaymentService.create_payment_session(
            db=db,
            workspace_id=workspace.id,
            call_id=body.call_id,
            amount_cents=body.amount_cents,
            currency=body.currency,
            description=body.description,
        )
    except Exception as exc:
        logger.error("create_payment_session failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe error: {exc}",
        )
    return create_success_response(
        response,
        "Payment session created successfully",
        status_code=status.HTTP_200_OK,
    )


# ---------------------------------------------------------------------------
# POST /stripe-webhook — receive Stripe events
# NOTE: This path MUST be listed in _SKIP_PREFIXES in api_key_middleware.py
#       because Stripe sends requests without our x-api-key header.
# ---------------------------------------------------------------------------

@router.post(
    "/stripe-webhook",
    summary="Stripe webhook for in-call payment events",
    include_in_schema=True,
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    description=(
        "Receives `payment_intent.succeeded` and `payment_intent.payment_failed` "
        "events from Stripe. Signature is verified with STRIPE_INCALL_WEBHOOK_SECRET. "
        "Returns 204 on success, 403 on invalid signature."
    ),
)
async def stripe_payment_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not sig_header:
        logger.warning("stripe-webhook: missing Stripe-Signature header")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing Stripe-Signature header",
        )

    webhook_secret = settings.STRIPE_INCALL_WEBHOOK_SECRET
    if not webhook_secret:
        # Fallback warning — should always be configured in staging/production
        logger.error(
            "STRIPE_INCALL_WEBHOOK_SECRET is not set; cannot verify webhook signature"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret not configured",
        )

    try:
        event = StripeService.construct_payment_webhook_event(
            payload=payload,
            sig_header=sig_header,
            webhook_secret=webhook_secret,
        )
    except (ValueError, stripe.SignatureVerificationError) as exc:
        logger.warning("stripe-webhook: signature verification failed — %s", exc)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Stripe webhook signature",
        )
    except Exception as exc:
        logger.error("stripe-webhook: unexpected error during verification — %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook processing error",
        )

    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    logger.info("stripe-webhook: received event type=%s", event_type)

    try:
        event_data = event if isinstance(event, dict) else event.to_dict()

        if event_type == "payment_intent.succeeded":
            PaymentService.handle_payment_intent_succeeded(db, event_data)

        elif event_type == "payment_intent.payment_failed":
            PaymentService.handle_payment_intent_failed(db, event_data)

        else:
            logger.info("stripe-webhook: unhandled event type=%s (ignored)", event_type)

    except Exception as exc:
        # Do not return 5xx — Stripe would retry. Log and return 204 anyway.
        logger.error(
            "stripe-webhook: error processing event type=%s — %s",
            event_type,
            exc,
            exc_info=True,
        )

    # Return 204 No Content — Stripe considers any 2xx a success
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# GET / — list payment records (paginated)
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=SuccessResponse[PaginatedPaymentResponse],
    summary="List payment records for the workspace",
)
def list_payment_records(
    page: int = 1,
    per_page: int = 20,
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_workspace),
):
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 20

    result = PaymentService.get_payments(
        db=db,
        workspace_id=workspace.id,
        page=page,
        per_page=per_page,
    )
    return create_success_response(result, "Payment records retrieved")


# ---------------------------------------------------------------------------
# GET /{payment_intent_id} — single payment record
# ---------------------------------------------------------------------------

@router.get(
    "/{payment_intent_id}",
    response_model=SuccessResponse[PaymentRecordOut],
    summary="Get a single payment record by PaymentIntent ID",
)
def get_payment_record(
    payment_intent_id: str,
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_workspace),
):
    record = PaymentService.get_payment(
        db=db,
        workspace_id=workspace.id,
        payment_intent_id=payment_intent_id,
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Payment record not found for payment_intent_id={payment_intent_id}",
        )
    return create_success_response(
        PaymentRecordOut.from_orm_model(record),
        "Payment record retrieved",
    )
