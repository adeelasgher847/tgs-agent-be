from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.api.deps import get_db, is_session_already_credited, mark_session_credited
from app.models.tenant import Tenant
from app.models.plan import Plan
from app.services.stripe_service import StripeService
import uuid
from app.core.logger import logger

router = APIRouter()

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events"""
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
    
    # Handle the event
    try:
        event_type = event['type']
        
        if event_type == 'checkout.session.completed':
            # 🎉 Payment Webhook Received - Process
            logger.info(f"📞 STRIPE WEBHOOK RECEIVED: checkout.session.completed")
            
            session = event.get('data', {}).get('object', {}) or {}
            session_id = session.get('id')
            
            if not session_id:
                return {"status": "ignored", "reason": "no_session_id"}

            from app.services.billing_service import BillingService
            result = BillingService.sync_payment_status(db, session_id)
            
            if result:
                return {**result, "message": "Webhook processed successfully"}
            else:
                return {"status": "failed", "reason": "sync_failed"}
        else:
            logger.info(f"Unhandled event type: {event_type}")
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
