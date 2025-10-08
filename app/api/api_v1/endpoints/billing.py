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
            # Credits are handled via verify-payment endpoint to ensure idempotency.
            pass
        else:
            print(f"Unhandled event type: {event_type}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error handling webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
