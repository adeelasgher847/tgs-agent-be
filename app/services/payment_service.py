"""
PaymentService — business logic for in-call Stripe payment processing.

Responsibilities:
  - Create a PaymentIntent + persist a PaymentRecord (status=pending)
  - Handle payment_intent.succeeded webhook: update status + card details
  - Handle payment_intent.payment_failed webhook: update status
  - Paginated listing of payment records for a workspace
  - Single record lookup
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.payment_record import PaymentRecord
from app.schemas.payment import (
    CreatePaymentSessionResponse,
    PaginatedPaymentResponse,
    PaymentRecordOut,
)
from app.services.stripe_service import StripeService


class PaymentService:

    # ------------------------------------------------------------------
    # Create payment session
    # ------------------------------------------------------------------

    @staticmethod
    def create_payment_session(
        db: Session,
        workspace_id: uuid.UUID,
        call_id: Optional[uuid.UUID],
        amount_cents: int,
        currency: str,
        description: str,
    ) -> Tuple[CreatePaymentSessionResponse, str]:
        """
        Create a Stripe PaymentIntent and persist a PaymentRecord.

        Returns:
            (CreatePaymentSessionResponse, agent_context_string)
        """
        stripe_metadata = {
            "workspace_id": str(workspace_id),
            "call_id": str(call_id) if call_id else "",
        }

        # Create the PaymentIntent in Stripe (test mode: sk_test_ key)
        intent = StripeService.create_payment_intent(
            amount_cents=amount_cents,
            currency=currency,
            description=description,
            metadata=stripe_metadata,
        )

        payment_intent_id: str = intent.id
        client_secret: str = intent.client_secret

        # Build the payment URL — no separate hosted page; just a deep-link
        # the caller opens that contains the client_secret.
        payment_url = (
            f"{settings.PAYMENT_PAGE_BASE_URL.rstrip('/')}"
            f"/pay/{payment_intent_id}"
            f"?client_secret={client_secret}"
        )

        # Persist a pending PaymentRecord
        record = PaymentRecord(
            workspace_id=workspace_id,
            call_id=call_id,
            payment_intent_id=payment_intent_id,
            amount_cents=amount_cents,
            currency=currency.lower(),
            description=description,
            status="pending",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "PaymentRecord created: pi=%s workspace=%s call=%s amount=%d %s",
            payment_intent_id,
            workspace_id,
            call_id,
            amount_cents,
            currency,
        )

        # Agent context injection sentence
        agent_context = (
            f"A payment page has been sent to the caller at {payment_url}. "
            "Ask them to complete the payment."
        )

        # Inject context into CallSession's metadata if call_id is provided
        if call_id:
            try:
                from app.models.call_session import CallSession
                call_session = db.query(CallSession).filter(CallSession.id == call_id).first()
                if call_session:
                    current_metadata = dict(call_session.call_metadata or {})
                    vdc = dict(current_metadata.get("voice_dynamic_context") or {})
                    vdc["system_prompt_addendum"] = agent_context
                    current_metadata["voice_dynamic_context"] = vdc
                    call_session.call_metadata = current_metadata
                    db.add(call_session)
                    db.commit()
                    logger.info("Injected payment agent_context into CallSession %s metadata", call_id)
            except Exception as e:
                logger.error("Failed to inject payment context into CallSession: %s", e, exc_info=True)

        response = CreatePaymentSessionResponse(
            payment_intent_id=payment_intent_id,
            client_secret=client_secret,
            payment_url=payment_url,
            agent_context=agent_context,
        )
        return response, agent_context

    # ------------------------------------------------------------------
    # Webhook handlers
    # ------------------------------------------------------------------

    @staticmethod
    def handle_payment_intent_succeeded(db: Session, event_data: dict) -> bool:
        """
        Handle payment_intent.succeeded webhook event.

        Updates status → "succeeded" and stores card_last4 / card_brand
        from the latest charge attached to the PaymentIntent.
        """
        pi_obj = event_data.get("data", {}).get("object", {})
        payment_intent_id = _get_field(pi_obj, "id")
        if not payment_intent_id:
            logger.warning("payment_intent.succeeded: missing pi id in event")
            return False

        record = (
            db.query(PaymentRecord)
            .filter(PaymentRecord.payment_intent_id == payment_intent_id)
            .first()
        )
        if not record:
            logger.warning(
                "payment_intent.succeeded: no PaymentRecord for pi=%s", payment_intent_id
            )
            return False

        # Extract card details from charges.data[0].payment_method_details.card
        card_last4 = None
        card_brand = None
        try:
            charges = _get_field(pi_obj, "charges") or {}
            charges_data = _get_field(charges, "data") or []
            if charges_data:
                charge = charges_data[0]
                pmd = _get_field(charge, "payment_method_details") or {}
                card_info = _get_field(pmd, "card") or {}
                card_last4 = _get_field(card_info, "last4")
                card_brand = _get_field(card_info, "brand")
        except Exception as exc:
            logger.warning("Could not extract card details from event: %s", exc)

        record.status = "succeeded"
        if card_last4:
            record.card_last4 = str(card_last4)
        if card_brand:
            record.card_brand = str(card_brand)

        db.commit()
        db.refresh(record)

        logger.info(
            "PaymentRecord succeeded: pi=%s last4=%s brand=%s",
            payment_intent_id,
            card_last4,
            card_brand,
        )
        return True

    @staticmethod
    def handle_payment_intent_failed(db: Session, event_data: dict) -> bool:
        """
        Handle payment_intent.payment_failed webhook event.

        Updates status → "failed".
        """
        pi_obj = event_data.get("data", {}).get("object", {})
        payment_intent_id = _get_field(pi_obj, "id")
        if not payment_intent_id:
            logger.warning("payment_intent.payment_failed: missing pi id in event")
            return False

        record = (
            db.query(PaymentRecord)
            .filter(PaymentRecord.payment_intent_id == payment_intent_id)
            .first()
        )
        if not record:
            logger.warning(
                "payment_intent.payment_failed: no PaymentRecord for pi=%s",
                payment_intent_id,
            )
            return False

        record.status = "failed"
        db.commit()
        db.refresh(record)

        logger.info("PaymentRecord failed: pi=%s", payment_intent_id)
        return True

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_payments(
        db: Session,
        workspace_id: uuid.UUID,
        page: int = 1,
        per_page: int = 20,
    ) -> PaginatedPaymentResponse:
        """Return a paginated list of payment records for a workspace."""
        offset = (page - 1) * per_page

        query = (
            db.query(PaymentRecord)
            .filter(PaymentRecord.workspace_id == workspace_id)
            .order_by(PaymentRecord.created_at.desc())
        )
        total = query.count()
        records = query.offset(offset).limit(per_page).all()

        pages = max(1, (total + per_page - 1) // per_page)
        return PaginatedPaymentResponse(
            items=[PaymentRecordOut.from_orm_model(r) for r in records],
            total=total,
            page=page,
            per_page=per_page,
            pages=pages,
        )

    @staticmethod
    def get_payment(
        db: Session,
        workspace_id: uuid.UUID,
        payment_intent_id: str,
    ) -> Optional[PaymentRecord]:
        """Return a single PaymentRecord for the given workspace + PI id."""
        return (
            db.query(PaymentRecord)
            .filter(
                PaymentRecord.workspace_id == workspace_id,
                PaymentRecord.payment_intent_id == payment_intent_id,
            )
            .first()
        )


# ---------------------------------------------------------------------------
# Internal helper: read a field from a Stripe object (StripeObject or dict)
# ---------------------------------------------------------------------------

def _get_field(obj, key: str):
    """Read a field from a Stripe StripeObject or plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
