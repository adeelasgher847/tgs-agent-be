from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.tenant import TenantCreate, TenantCreateResponse, TenantOut
from app.schemas.auth import SwitchTenantRequest, TokenResponse, RoleInfo
from app.schemas.base import SuccessResponse
from app.models.tenant import Tenant
from app.models.user import User
from app.models.role import Role
from app.api.deps import get_db, get_current_user_jwt, require_admin, require_member_or_admin
from app.core.security import create_user_token
from app.utils.response import create_success_response
import re
from app.core.config import settings
from app.models.user import user_tenant_association

from sqlalchemy import update
router = APIRouter()

def generate_schema_name(tenant_name: str) -> str:
    """Generate a schema name from tenant name"""
    # Convert to lowercase, replace spaces/special chars with underscores
    schema_name = re.sub(r'[^a-zA-Z0-9]', '_', tenant_name.lower())
    # Remove multiple underscores and trailing/leading underscores
    schema_name = re.sub(r'_+', '_', schema_name).strip('_')
    return f"{schema_name}_schema"

@router.post("/create", response_model=SuccessResponse[TenantCreateResponse])
def create_tenant(tenant_in: TenantCreate, current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """
    Create a new tenant organization and associate the creator as its admin.
    
    Requirements:
    - Tenant name must be unique
    - Creator user is auto-linked to the tenant with role "admin"
    - Sets the new tenant as user's current tenant
    - Creates Stripe customer and links it to the tenant
    - Returns tenant_id and tenant details with updated token
    """
    # Check if tenant name already exists for this user
    existing_tenant = db.query(Tenant).join(user_tenant_association).filter(
        Tenant.name == tenant_in.name,
        user_tenant_association.c.user_id == current_user.id
    ).first()
    if existing_tenant:
        raise HTTPException(status_code=400, detail="You already have a tenant with this name")
    
    # Generate unique schema name
    schema_name = generate_schema_name(tenant_in.name)
    
    # Ensure schema name is unique
    existing_schema = db.query(Tenant).filter(Tenant.schema_name == schema_name).first()
    counter = 1
    original_schema = schema_name
    while existing_schema:
        schema_name = f"{original_schema}_{counter}"
        existing_schema = db.query(Tenant).filter(Tenant.schema_name == schema_name).first()
        counter += 1
    
    # Create new tenant with pending_payment status
    db_tenant = Tenant(
        name=tenant_in.name,
        schema_name=schema_name,
        status="pending_payment"
    )
    
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    
    # Create Stripe customer and link it to the tenant
    from app.services.stripe_service import StripeService
    try:
        stripe_customer_id = StripeService.create_customer(
            tenant=db_tenant,
            email=current_user.email,
            user=current_user
        )
        db_tenant.stripe_customer_id = stripe_customer_id
        db.commit()
    except Exception as e:
        # If Stripe customer creation fails, delete the tenant
        db.delete(db_tenant)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create Stripe customer: {str(e)}"
        )
    
    # Get admin role by name
    admin_role = db.query(Role).filter(Role.name == settings.ADMIN_ROLE).first()
    if not admin_role:
        raise HTTPException(
            status_code=400, 
            detail="Admin role not found. Please contact administrator."
        )

    # Add user to tenant's users list (many-to-many association)
    current_user.tenants.append(db_tenant)
    
    # Commit the association first
    db.commit()
    
    # Update the role_id in the association table
    stmt = update(user_tenant_association).where(
        (user_tenant_association.c.user_id == current_user.id) &
        (user_tenant_association.c.tenant_id == db_tenant.id)
    ).values(role_id=admin_role.id)
    
    db.execute(stmt)
    
    # Set the new tenant as user's current tenant
    # current_user.current_tenant_id = db_tenant.id
    
    db.commit()
    db.refresh(current_user)
    
    # Convert SQLAlchemy model to Pydantic model
    tenant_out = TenantOut.model_validate(db_tenant)
    
    tenant_response = TenantCreateResponse(
        tenant_id=db_tenant.id,
        tenant=tenant_out
    )
    
    return create_success_response(tenant_response, "Tenant created successfully", status.HTTP_201_CREATED)


@router.post("/switch", response_model=SuccessResponse[TokenResponse])
def switch_tenant(
    switch_data: SwitchTenantRequest,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Switch to a different tenant and return new JWT token with role information.
    Also updates the user's current_tenant_id in the database.
    """
    # Get user's tenant IDs from database
    user_tenant_ids = [tenant.id for tenant in current_user.tenants]
    
    # Check if user has access to the requested tenant
    if switch_data.tenant_id not in user_tenant_ids:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied to this tenant"
        )
    
    # Update user's current_tenant_id in the database
    current_user.current_tenant_id = switch_data.tenant_id
    db.commit()
    db.refresh(current_user)
    
    # One-time credit sync for the switched tenant
    try:
        from app.models.tenant import Tenant
        from app.models.subscription import Subscription
        
        # Get the tenant and check if credits need to be synced
        tenant = db.query(Tenant).filter(Tenant.id == switch_data.tenant_id).first()
        if tenant and tenant.subscription:
            subscription = tenant.subscription
            
            # Check if this is a one-time sync (only if credits haven't been updated for this subscription)
            if (not subscription.credits_updated and 
                subscription.status == "active" and 
                subscription.plan and 
                subscription.plan.credits and 
                subscription.plan.credits > 0):
                
                # Update credits based on plan
                old_balance = tenant.credit_balance
                tenant.credit_balance = subscription.plan.credits
                subscription.credits_updated = True  # Mark as updated
                db.commit()
                
                print(f"✅ One-time credit sync for tenant {switch_data.tenant_id}: {old_balance} → {tenant.credit_balance} credits (credits_updated=True)")
            else:
                if subscription.credits_updated:
                    print(f"ℹ️ Credit sync skipped for tenant {switch_data.tenant_id}: credits already updated for this subscription")
                else:
                    print(f"ℹ️ Credit sync skipped for tenant {switch_data.tenant_id}: status={subscription.status if subscription else 'no subscription'}")
    except Exception as e:
        print(f"⚠️ Credit sync failed for tenant {switch_data.tenant_id}: {str(e)}")
        # Don't fail the tenant switch if credit sync fails
    
    # Get role information for the switched tenant
    role_info = None
    current_role = None
    from app.services.role_service import get_user_role_in_tenant
    role = get_user_role_in_tenant(db, current_user.id, switch_data.tenant_id)
    if role:
        role_info = RoleInfo(
            id=role.id,
            name=role.name,
            description=role.description
        )
        current_role = role.name
    
    # Create new token with updated tenant and role
    access_token = create_user_token(
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=switch_data.tenant_id,
        role=current_role
    )

    # Create refresh token (valid 7 days)
    from app.core.security import create_refresh_token_value, refresh_token_expires_at
    from app.models.refresh_token import RefreshToken
    
    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=current_user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False
    )
    db.add(rt)
    db.commit()
    
    token_response = TokenResponse(
        access_token=access_token,
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=switch_data.tenant_id,
        tenant_ids=user_tenant_ids,
        role=role_info,
        refresh_token=rt_value
    )
    
    return create_success_response(token_response, "Tenant switched successfully")

@router.post("/start-checkout")
def start_checkout_session(
    stripe_price_id: str,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Start Stripe checkout session for tenant subscription.
    Stripe customer ID is automatically fetched from tenant record.
    Tenant ID is fetched from current user's JWT token.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    tenant_id = str(current_user.current_tenant_id)
    
    # Validate tenant exists and has Stripe customer ID
    tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    if not tenant.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Stripe customer found for this tenant. Please create tenant first."
        )
    
    # Create checkout session directly with Stripe
    import stripe
    from app.core.config import settings
    
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    success_url = f"{settings.FRONTEND_URL}/payment/success?tenant_id={tenant_id}"
    cancel_url = f"{settings.FRONTEND_URL}/payment/cancel?tenant_id={tenant_id}"
    
    try:
        checkout_session = stripe.checkout.Session.create(
            customer=tenant.stripe_customer_id,
            success_url=success_url,
            cancel_url=cancel_url,
            mode="subscription",
            line_items=[{
                "price": stripe_price_id,
                "quantity": 1
            }],
            metadata={
                "tenant_id": tenant_id,
                "stripe_customer_id": tenant.stripe_customer_id
            }
        )
        
        # Create or update subscription record
        from app.models.subscription import Subscription
        from app.models.plan import Plan
        
        # Get plan by stripe_price_id
        plan = db.query(Plan).filter(Plan.stripe_price_id == stripe_price_id).first()
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found for the given stripe_price_id"
            )
        
        # Check if subscription already exists
        subscription = db.query(Subscription).filter(
            Subscription.tenant_id == current_user.current_tenant_id
        ).first()
        
        if subscription:
            # Update existing subscription
            subscription.stripe_customer_id = tenant.stripe_customer_id
            subscription.plan_id = plan.id
            subscription.status = "pending_payment"
            subscription.stripe_session_id = checkout_session.id
        else:
            # Create new subscription
            subscription = Subscription(
                tenant_id=current_user.current_tenant_id,
                plan_id=plan.id,
                stripe_customer_id=tenant.stripe_customer_id,
                status="pending_payment",
                stripe_session_id=checkout_session.id
            )
            db.add(subscription)
        
        db.commit()
        
        return create_success_response({
            "session_id": checkout_session.id,
            "url": checkout_session.url,
            "subscription_id": str(subscription.id)
        }, "Checkout session created successfully")
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/verify-payment/{session_id}")
def verify_payment(
    session_id: str,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Verify payment status using Stripe checkout session ID.
    Returns payment details and subscription information.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    # Retrieve checkout session from Stripe
    import stripe
    from app.core.config import settings
    
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Verify this session belongs to the current tenant
        if session.metadata.get("tenant_id") != str(current_user.current_tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This payment session does not belong to your tenant"
            )
        
        # Get subscription details if payment was successful
        subscription_info = None
        if session.payment_status == "paid" and session.subscription:
            try:
                stripe_subscription = stripe.Subscription.retrieve(session.subscription)
                subscription_info = {
                    "stripe_subscription_id": stripe_subscription.id,
                    "status": stripe_subscription.status,
                    "current_period_start": stripe_subscription.current_period_start,
                    "current_period_end": stripe_subscription.current_period_end,
                    "cancel_at_period_end": stripe_subscription.cancel_at_period_end
                }
            except Exception as e:
                print(f"Error retrieving subscription: {str(e)}")
        
        return create_success_response({
            "session_id": session.id,
            "payment_status": session.payment_status,
            "subscription_id": session.subscription,
            "customer_id": session.customer,
            "amount_total": session.amount_total,
            "currency": session.currency,
            "payment_intent": session.payment_intent,
            "subscription_details": subscription_info,
            "metadata": session.metadata
        }, "Payment verification completed")
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/verify-last-payment")
def verify_last_payment(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Verify the last payment for the current tenant.
    Automatically gets the latest session ID from subscription table.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    # Get the latest subscription for the tenant
    from app.models.subscription import Subscription
    subscription = db.query(Subscription).filter(
        Subscription.tenant_id == current_user.current_tenant_id
    ).first()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription found for this tenant"
        )
    
    if not subscription.stripe_session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No payment session found for this tenant"
        )
    
    # Use the existing verify_payment function with the session ID
    return verify_payment(subscription.stripe_session_id, current_user, db)

@router.get("/payment-history")
def get_payment_history(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Get complete payment history for the current tenant.
    Returns all payment attempts, failures, refunds, and invoices.
    """
    try:
        if not current_user.current_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No tenant selected"
            )
        
        # Get tenant and subscription
        tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        if not tenant.stripe_customer_id:
            return create_success_response({
                "payment_history": [],
                "summary": {
                    "total_payments": 0,
                    "successful_payments": 0,
                    "failed_payments": 0,
                    "total_amount": 0,
                    "currency": "usd"
                }
            }, "No payment history found")
        
        # Import Stripe
        import stripe
        from app.core.config import settings
        
        stripe.api_key = settings.STRIPE_SECRET_KEY
        
        payment_history = []
        
        # 1. Get all checkout sessions (payment attempts)
        try:
            checkout_sessions = stripe.checkout.Session.list(
                customer=tenant.stripe_customer_id,
                limit=100
            )
            
            for session in checkout_sessions.data:
                amount_dollars = session.amount_total / 100 if session.amount_total else 0
                payment_entry = {
                    "type": "checkout_session",
                    "id": session.id,
                    "status": session.payment_status,
                    "amount_total": amount_dollars,
                    "amount_total_cents": session.amount_total,
                    "amount_formatted": f"${amount_dollars:.2f}",  # Format as USD
                    "currency": session.currency,
                    "created": session.created,
                    "payment_intent": session.payment_intent,
                    "subscription_id": session.subscription,
                    "success": session.payment_status == "paid",
                    "failure_reason": None
                }
                
                # Get failure reason if payment failed
                if session.payment_status == "unpaid" and session.payment_intent:
                    try:
                        payment_intent = stripe.PaymentIntent.retrieve(session.payment_intent)
                        if payment_intent.last_payment_error:
                            payment_entry["failure_reason"] = payment_intent.last_payment_error.get("message", "Payment failed")
                    except:
                        pass
                
                payment_history.append(payment_entry)
        except Exception as e:
            print(f"Error getting checkout sessions: {str(e)}")
        
        # 2. Get all invoices
        try:
            invoices = stripe.Invoice.list(
                customer=tenant.stripe_customer_id,
                limit=100
            )
            
            for invoice in invoices.data:
                amount_dollars = invoice.amount_total / 100 if invoice.amount_total else 0
                payment_entry = {
                    "type": "invoice",
                    "id": invoice.id,
                    "status": invoice.status,
                    "amount_total": amount_dollars,
                    "amount_total_cents": invoice.amount_total,
                    "amount_formatted": f"${amount_dollars:.2f}",  # Format as USD
                    "currency": invoice.currency,
                    "created": invoice.created,
                    "payment_intent": invoice.payment_intent,
                    "subscription_id": invoice.subscription,
                    "success": invoice.status == "paid",
                    "failure_reason": None,
                    "invoice_url": invoice.invoice_pdf,
                    "period_start": invoice.period_start,
                    "period_end": invoice.period_end
                }
                
                # Get failure reason if invoice failed
                if invoice.status == "open" and invoice.attempt_count > 0:
                    payment_entry["failure_reason"] = "Invoice payment failed after multiple attempts"
                
                payment_history.append(payment_entry)
        except Exception as e:
            print(f"Error getting invoices: {str(e)}")
        
        # Sort by creation date (newest first)
        payment_history.sort(key=lambda x: x["created"], reverse=True)
        
        # Calculate summary statistics
        successful_payments = [p for p in payment_history if p["success"] and p["type"] != "refund"]
        failed_payments = [p for p in payment_history if not p["success"] and p["type"] != "refund"]
        total_amount = sum(p["amount_total"] for p in payment_history if p["type"] != "refund")  # Already converted to dollars
        total_amount_cents = sum(p["amount_total_cents"] for p in payment_history if p["type"] != "refund")
        
        summary = {
            "total_payments": len(payment_history),
            "successful_payments": len(successful_payments),
            "failed_payments": len(failed_payments),
            "total_amount": total_amount,  # In dollars
            "total_amount_cents": total_amount_cents,  # In cents
            "total_amount_formatted": f"${total_amount:.2f}",  # Format as USD
            "currency": "usd" if payment_history else "usd",
            "refunds_count": len([p for p in payment_history if p["type"] == "refund"]),
            "last_payment_date": payment_history[0]["created"] if payment_history else None
        }
        
        return create_success_response({
            "payment_history": payment_history,
            "summary": summary
        }, "Payment history retrieved successfully")
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@router.post("/sync-credits", response_model=SuccessResponse[dict])
def sync_tenant_credits(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    One-time sync of tenant credits based on their current subscription.
    This will only update credits if the tenant has 0 credits and an active subscription.
    """
    try:
        from app.models.tenant import Tenant
        from app.models.subscription import Subscription
        
        # Get the tenant ID from the current user
        tenant_id = current_user.current_tenant_id
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No tenant associated with current user"
            )
        
        # Get the tenant and check if credits need to be synced
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        sync_result = {
            "tenant_id": str(tenant_id),
            "tenant_name": tenant.name,
            "before_sync": {
                "credit_balance": tenant.credit_balance,
                "subscription_status": None,
                "plan_name": "No Plan",
                "plan_credits": 0
            },
            "after_sync": {
                "credit_balance": tenant.credit_balance,
                "subscription_status": None,
                "plan_name": "No Plan",
                "plan_credits": 0
            },
            "action_taken": "No action needed",
            "success": False
        }
        
        if tenant.subscription:
            subscription = tenant.subscription
            sync_result["before_sync"]["subscription_status"] = subscription.status
            sync_result["after_sync"]["subscription_status"] = subscription.status
            
            if subscription.plan:
                plan = subscription.plan
                plan_credits = plan.credits or 0
                sync_result["before_sync"]["plan_name"] = plan.display_name
                sync_result["before_sync"]["plan_credits"] = plan_credits
                sync_result["after_sync"]["plan_name"] = plan.display_name
                sync_result["after_sync"]["plan_credits"] = plan_credits
                
                # One-time sync logic: only update if credits haven't been updated for this subscription
                if (not subscription.credits_updated and 
                    subscription.status == "active" and 
                    plan_credits > 0):
                    
                    # Update credits based on plan
                    tenant.credit_balance = plan_credits
                    subscription.credits_updated = True  # Mark as updated
                    db.commit()
                    
                    sync_result["after_sync"]["credit_balance"] = plan_credits
                    sync_result["action_taken"] = f"Updated credits to {plan_credits} based on active plan (credits_updated=True)"
                    sync_result["success"] = True
                    
                    print(f"✅ One-time credit sync: {tenant_id} → {plan_credits} credits (credits_updated=True)")
                else:
                    if subscription.credits_updated:
                        sync_result["action_taken"] = f"Credits already updated for this subscription (credits_updated=True)"
                    elif subscription.status != "active":
                        sync_result["action_taken"] = f"Subscription not active (status: {subscription.status})"
                    else:
                        sync_result["action_taken"] = "Plan has no credits to sync"
                    sync_result["success"] = True
            else:
                sync_result["action_taken"] = "No plan associated with subscription"
        else:
            sync_result["action_taken"] = "No subscription found"
        
        return create_success_response(
            sync_result,
            f"Credit sync completed for tenant {tenant_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error syncing credits: {str(e)}"
        )