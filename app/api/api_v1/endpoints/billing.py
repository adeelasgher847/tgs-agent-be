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

@router.get("/invoices")
def get_invoices(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """Get invoice history for the tenant"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_customer_id:
        return create_success_response([], "No invoices found")
    
    try:
        invoices = StripeService.get_invoices(subscription.stripe_customer_id, limit)
        return create_success_response(invoices, "Invoices retrieved successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/upcoming-invoice")
def get_upcoming_invoice(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get upcoming invoice for the tenant"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Stripe customer found"
        )
    
    try:
        invoice = StripeService.get_upcoming_invoice(subscription.stripe_customer_id)
        return create_success_response(invoice, "Upcoming invoice retrieved successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/payment-methods")
def get_payment_methods(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get payment methods for the tenant"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_customer_id:
        return create_success_response([], "No payment methods found")
    
    try:
        payment_methods = StripeService.get_customer_payment_methods(subscription.stripe_customer_id)
        return create_success_response(payment_methods, "Payment methods retrieved successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/cancel-subscription")
def cancel_subscription(
    at_period_end: bool = True,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Cancel the current subscription"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_subscription_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription found"
        )
    
    try:
        result = StripeService.cancel_subscription(subscription.stripe_subscription_id, at_period_end)
        
        # Update local subscription status
        if at_period_end:
            subscription.cancel_at_period_end = True
        else:
            subscription.status = 'canceled'
            subscription.canceled_at = datetime.now()
        
        db.commit()
        
        return create_success_response(result, "Subscription canceled successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/pause-subscription")
def pause_subscription(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Pause the current subscription"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_subscription_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription found"
        )
    
    try:
        result = StripeService.pause_subscription(subscription.stripe_subscription_id)
        return create_success_response(result, "Subscription paused successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/resume-subscription")
def resume_subscription(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Resume a paused subscription"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    subscription = BillingService.get_or_create_subscription(db, current_user.current_tenant_id)
    
    if not subscription.stripe_subscription_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription found"
        )
    
    try:
        result = StripeService.resume_subscription(subscription.stripe_subscription_id)
        return create_success_response(result, "Subscription resumed successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/analytics")
def get_usage_analytics(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get usage analytics and trends"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    analytics = BillingService.get_usage_analytics(db, current_user.current_tenant_id)
    return create_success_response(analytics, "Usage analytics retrieved successfully")

@router.get("/summary")
def get_billing_summary(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get comprehensive billing summary"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    summary = BillingService.get_billing_summary(db, current_user.current_tenant_id)
    return create_success_response(summary, "Billing summary retrieved successfully")

@router.get("/history")
def get_usage_history(
    months: int = 12,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get usage history for the past N months"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    history = BillingService.get_usage_history(db, current_user.current_tenant_id, months)
    return create_success_response(history, "Usage history retrieved successfully")

@router.post("/sync-usage")
def sync_usage_with_stripe(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Sync usage data with Stripe for metered billing"""
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    try:
        BillingService.sync_usage_with_stripe(db, current_user.current_tenant_id)
        return create_success_response(
            {"message": "Usage synced with Stripe successfully"},
            "Usage sync completed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
