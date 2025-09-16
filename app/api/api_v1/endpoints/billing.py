from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.subscription import SubscriptionOut, SubscriptionWithUsage
from app.schemas.plan import PlanOut
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.models.tenant import Tenant
from app.api.deps import get_db, get_current_user_jwt, require_admin
from app.services.stripe_service import StripeService
from app.services.billing_service import BillingService
from app.core.config import settings
from app.utils.response import create_success_response
import uuid
import json
from datetime import datetime

router = APIRouter()

@router.post("/checkout")
def create_checkout_session(
    plan_id: str,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Create Stripe checkout session for subscription"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    tenant_id = str(current_user.current_tenant_id)
    success_url = f"{settings.FRONTEND_URL}/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{settings.FRONTEND_URL}/billing/cancel"
    
    try:
        result = StripeService.create_checkout_session(
            tenant_id=tenant_id,
            plan_id=plan_id,
            success_url=success_url,
            cancel_url=cancel_url,
            db=db  # Add this line
        )
        
        return create_success_response(result, "Checkout session created successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/portal")
def create_portal_session(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Create Stripe customer portal session"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    # Get subscription to find customer ID
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Stripe customer found for this tenant"
        )
    
    return_url = f"{settings.FRONTEND_URL}/billing"
    
    try:
        result = StripeService.create_portal_session(
            customer_id=subscription.stripe_customer_id,
            return_url=return_url
        )
        
        return create_success_response(result, "Portal session created successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/subscription", response_model=SuccessResponse[SubscriptionWithUsage])
def get_subscription_status(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get current subscription status and usage"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    status_data = BillingService.get_tenant_subscription_status(db, current_user.current_tenant_id)
    
    # Convert to response format
    subscription_data = status_data['subscription']
    plan_data = status_data['plan']
    usage_data = status_data['usage']
    
    subscription_out = SubscriptionWithUsage(
        id=subscription_data['id'],
        tenant_id=current_user.current_tenant_id,
        plan_id=plan_data['id'],
        status=subscription_data['status'],
        current_period_start=subscription_data['current_period_start'],
        current_period_end=subscription_data['current_period_end'],
        cancel_at_period_end=subscription_data['cancel_at_period_end'],
        stripe_subscription_id=subscription_data['stripe_subscription_id'],
        stripe_customer_id=subscription_data['stripe_customer_id'],
        canceled_at=None,  # Add if needed
        created_at=datetime.now(),  # Add if needed
        updated_at=None,  # Add if needed
        plan=PlanOut(
            id=plan_data['id'],
            name=plan_data['name'],
            display_name=plan_data['display_name'],
            description=plan_data['description'],
            price_monthly=plan_data['price_monthly'],
            price_yearly=plan_data['price_yearly'],
            agent_limit=plan_data['agent_limit'],
            monthly_calls_limit=plan_data['monthly_calls_limit'],
            is_active=True,
            stripe_price_id=None,  # Add if needed
            created_at=datetime.now(),  # Add if needed
            updated_at=None  # Add if needed
        ),
        current_usage=usage_data
    )
    
    return create_success_response(subscription_out, "Subscription status retrieved successfully")

@router.get("/usage")
def get_usage_status(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get current usage status"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    usage = BillingService.get_current_usage(db, current_user.current_tenant_id)
    limits = BillingService.enforce_limits(db, current_user.current_tenant_id)
    
    return create_success_response({
        'usage': usage,
        'limits': limits
    }, "Usage status retrieved successfully")

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
        if event['type'] == 'checkout.session.completed':
            StripeService.handle_checkout_completed(event, db)
        elif event['type'] == 'invoice.paid':
            StripeService.handle_invoice_paid(event, db)
        elif event['type'] == 'invoice.payment_failed':
            StripeService.handle_invoice_payment_failed(event, db)
        elif event['type'] == 'customer.subscription.updated':
            StripeService.handle_subscription_updated(event, db)
        elif event['type'] == 'customer.subscription.deleted':
            StripeService.handle_subscription_deleted(event, db)
        else:
            print(f"Unhandled event type: {event['type']}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error handling webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )

@router.post("/downgrade")
def downgrade_to_free(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Manually downgrade tenant to free plan (admin only)"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    BillingService.downgrade_to_free_plan(db, current_user.current_tenant_id)
    
    return create_success_response(
        {"message": "Successfully downgraded to free plan"},
        "Downgrade completed successfully"
    )
