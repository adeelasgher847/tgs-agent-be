from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.models.tenant import Tenant
from app.models.plan import Plan
from app.api.deps import get_db, get_current_user_jwt, require_admin_or_owner
from app.services.stripe_service import StripeService
from app.utils.response import create_success_response
import stripe
from app.core.config import settings

router = APIRouter()

@router.post("/webhook")
def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events - DISABLED"""
    # Webhook is disabled in favor of direct API calls
    return {"status": "webhook_disabled", "message": "Use /confirm-payment endpoint instead"}

@router.post("/confirm-payment")
def confirm_payment(
    session_id: str,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Confirm payment using Stripe API and add credits to tenant.
    This replaces webhook dependency with direct API calls.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    # Set Stripe API key
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    try:
        # Retrieve checkout session from Stripe
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Verify payment status
        if session.payment_status != 'paid':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payment not completed or failed"
            )
        
        # Get metadata
        metadata = session.get("metadata", {})
        tenant_id = metadata.get("tenant_id")
        plan_id = metadata.get("plan_id")
        amount = float(metadata.get("amount", 0))
        
        # Get the tenant who made the payment (from session metadata)
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            print(f"BILLING DEBUG: Payment tenant not found: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_id} not found"
            )
        
        print(f"BILLING DEBUG: Found payment tenant - ID: {tenant.id}")
        print(f"BILLING DEBUG: Payment tenant credits: {tenant.credits}")
        print(f"BILLING DEBUG: Payment tenant status: {tenant.status}")
        
        # Get plan details
        plan = None
        if plan_id:
            plan = db.query(Plan).filter(Plan.id == plan_id).first()
        
        # Calculate credits based on payment amount: $1 = 10 credits
        credits_to_add = int(amount * 10)
        
        # Add credits to tenant
        old_credits = tenant.credits or 0
        print(f"BILLING DEBUG: Old credits: {old_credits}")
        print(f"BILLING DEBUG: Credits to add: {credits_to_add}")
        
        tenant.credits = old_credits + credits_to_add
        tenant.status = 'active'
        
        print(f"BILLING DEBUG: New credits before commit: {tenant.credits}")
        print(f"BILLING DEBUG: New status before commit: {tenant.status}")
        
        # Update subscription ID if available
        if session.subscription:
            print(f"BILLING DEBUG: Updating subscription ID: {session.subscription}")
            tenant.stripe_subscription_id = session.subscription
        
        print(f"BILLING DEBUG: About to commit to database...")
        db.commit()
        print(f"BILLING DEBUG: Database commit successful")
        
        db.refresh(tenant)
        print(f"BILLING DEBUG: After refresh - credits: {tenant.credits}, status: {tenant.status}")
        
        # Get subscription information
        subscription_info = None
        if session.subscription:
            try:
                subscription = stripe.Subscription.retrieve(session.subscription)
                subscription_info = {
                    "subscription_id": subscription.id,
                    "status": subscription.status,
                    "current_period_start": subscription.current_period_start,
                    "current_period_end": subscription.current_period_end,
                    "plan_id": subscription.items.data[0].price.id if subscription.items.data else None
                }
            except:
                subscription_info = {"subscription_id": session.subscription, "status": "unknown"}
        
        # Prepare response
        response_data = {
            "tenant_id": str(tenant.id),
            "credits_added": credits_to_add,
            "total_credits": tenant.credits,
            "payment_amount": amount,
            "plan_name": plan.display_name if plan else "Credit Purchase",
            "subscription_id": session.subscription,
            "subscription_info": subscription_info,
            "status": "success"
        }
        
        return create_success_response(response_data, "Payment confirmed and credits added successfully")
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error confirming payment: {str(e)}"
        )

@router.get("/payment-status/{session_id}")
def get_payment_status(
    session_id: str,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Check payment status without adding credits.
    Useful for checking if payment is completed before confirming.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    # Set Stripe API key
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    try:
        # Retrieve checkout session from Stripe
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Get metadata
        metadata = session.get("metadata", {})
        tenant_id = metadata.get("tenant_id")
        plan_id = metadata.get("plan_id")
        amount = float(metadata.get("amount", 0))
        
        # Verify this session belongs to current tenant
        if tenant_id != str(current_user.current_tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This payment session does not belong to your tenant"
            )
        
        # Get plan details
        plan = None
        if plan_id:
            plan = db.query(Plan).filter(Plan.id == plan_id).first()
        
        # Calculate credits that would be added
        credits_to_add = int(amount * 10)
        
        response_data = {
            "session_id": session_id,
            "payment_status": session.payment_status,
            "amount_total": session.amount_total,
            "currency": session.currency,
            "credits_to_add": credits_to_add,
            "plan_name": plan.display_name if plan else "Credit Purchase",
            "amount": amount
        }
        
        return create_success_response(response_data, "Payment status retrieved successfully")
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking payment status: {str(e)}"
        )
