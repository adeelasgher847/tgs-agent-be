from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.api.deps import get_db
from app.services.stripe_service import StripeService
from app.core.logger import logger

router = APIRouter()
IGNORED_EVENT_TYPES = {
    "charge.updated",
    "payment_intent.created",
}


def _get_event_value(event, key: str):
    """Read value from Stripe Event regardless of object/dict shape."""
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _extract_checkout_session(event):
    """Extract checkout session object from Stripe event payload safely."""
    data = _get_event_value(event, "data")
    if isinstance(data, dict):
        return data.get("object") or {}
    if data is None:
        return {}
    # Stripe SDK returns StripeObject with attributes.
    return getattr(data, "object", None) or {}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events. Plan purchase updates subscription only (no credits)."""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing stripe-signature header"
        )
    
    try:
        event = StripeService.construct_webhook_event(payload, sig_header)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    try:
        event_type = _get_event_value(event, "type")
        if not event_type:
            return {"status": "ignored", "reason": "missing_event_type"}
        
        if event_type == 'checkout.session.completed':
            logger.info("STRIPE WEBHOOK: checkout.session.completed")
            # StripeObject supports [] not dict.get(); .get looks up key "get" and raises.
            session = event["data"]["object"]
            session_id = session["id"]
            if not session_id:
                return {"status": "ignored", "reason": "no_session_id"}

            from app.services.billing_service import BillingService
            result = BillingService.sync_payment_status(db, session_id)
            
            if result:
                return {**result, "message": "Webhook processed successfully"}
            return {"status": "failed", "reason": "sync_failed"}
        
        if event_type in IGNORED_EVENT_TYPES:
            logger.info(f"Ignored event type: {event_type}")
            return {"status": "ignored", "event_type": event_type}

        logger.info(f"Unhandled event type: {event_type}")
        return {"status": "success", "event_type": event_type}
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
