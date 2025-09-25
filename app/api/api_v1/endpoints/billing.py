from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.api.deps import get_db
from app.services.stripe_service import StripeService

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
            StripeService.handle_checkout_completed(event, db)
        elif event_type == 'invoice.paid':
            StripeService.handle_invoice_paid(event, db)
        elif event_type == 'invoice.payment_failed':
            StripeService.handle_invoice_payment_failed(event, db)
        elif event_type == 'customer.subscription.updated':
            StripeService.handle_subscription_updated(event, db)
        elif event_type == 'customer.subscription.deleted':
            StripeService.handle_subscription_deleted(event, db)
        elif event_type == 'customer.updated':
            StripeService.handle_customer_updated(event, db)
        elif event_type == 'payment_method.attached':
            StripeService.handle_payment_method_attached(event, db)
        elif event_type == 'payment_method.detached':
            StripeService.handle_payment_method_detached(event, db)
        elif event_type == 'invoice.created':
            StripeService.handle_invoice_created(event, db)
        elif event_type == 'invoice.finalized':
            StripeService.handle_invoice_finalized(event, db)
        elif event_type == 'customer.subscription.trial_will_end':
            StripeService.handle_customer_subscription_trial_will_end(event, db)
        elif event_type == 'customer.subscription.created':
            StripeService.handle_customer_subscription_created(event, db)
        else:
            print(f"Unhandled event type: {event_type}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error handling webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
