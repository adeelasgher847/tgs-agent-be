from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.tenant import TenantCreate, TenantCreateResponse, TenantOut
from app.schemas.auth import SwitchTenantRequest, TokenResponse, RoleInfo
from app.schemas.base import SuccessResponse
from app.models.tenant import Tenant
from app.models.user import User
from app.models.role import Role
from app.api.deps import get_db, get_current_user_jwt, require_admin, require_member_or_admin, require_admin_or_owner, is_session_already_credited, mark_session_credited
from app.core.security import create_user_token, create_refresh_token_value, refresh_token_expires_at
from app.utils.response import create_success_response
import re
from app.core.config import settings
from app.models.user import user_tenant_association
from app.models.refresh_token import RefreshToken
from app.models.plan import Plan
import stripe
import uuid

from sqlalchemy import update
router = APIRouter()

def generate_schema_name(tenant_name: str) -> str:
    """Generate a schema name from tenant name"""
    # Convert to lowercase, replace spaces/special chars with underscores
    schema_name = re.sub(r'[^a-zA-Z0-9]', '_', tenant_name.lower())
    # Remove multiple underscores and trailing/leading underscores
    schema_name = re.sub(r'_+', '_', schema_name).strip('_')
    return f"{schema_name}_schema"

@router.post("/create", response_model=SuccessResponse[TokenResponse])
def create_tenant(tenant_in: TenantCreate, current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """
    Create a new tenant organization and associate the creator as its admin.
    
    Requirements:
    - Tenant name must be unique
    - Creator user is auto-linked to the tenant with role "admin"
    - Sets the new tenant as user's current tenant
    - Returns tenant_id and tenant details with updated token
    """
    # Trim whitespace from tenant name
    tenant_in.name = " ".join(tenant_in.name.split())
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
        status="pending_payment",
        credits=50
    )
    
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    
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
    current_user.current_tenant_id = db_tenant.id
    
    db.commit()
    db.refresh(current_user)
    
    # Get role information for the new tenant
    role_info = None
    current_role = None
    if admin_role:
        role_info = RoleInfo(
            id=admin_role.id,
            name=admin_role.name,
            description=admin_role.description
        )
        current_role = admin_role.name
    
    # Create new token with updated tenant and role
    access_token = create_user_token(
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=db_tenant.id,
        role=current_role
    )

    # Create refresh token (valid 7 days)
    
    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=current_user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False
    )
    db.add(rt)
    db.commit()
    
    # Get user's updated tenant IDs
    user_tenant_ids = [tenant.id for tenant in current_user.tenants]
    
    token_response = TokenResponse(
        access_token=access_token,
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=db_tenant.id,
        tenant_ids=user_tenant_ids,
        role=role_info,
        refresh_token=rt_value
    )
    
    return create_success_response(token_response, "Tenant created successfully", status.HTTP_201_CREATED)


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
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Start a one-time Stripe checkout for a plan purchase.
    Credits will be granted after verification: $1 = 10 credits.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    tenant_id = str(current_user.current_tenant_id)
    tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    # Create Stripe customer if not exists
    if not tenant.stripe_customer_id:
        from app.services.stripe_service import StripeService
        stripe_customer_id = StripeService.create_customer(
            tenant=tenant,
            email=current_user.email,
            user=current_user
        )
        tenant.stripe_customer_id = stripe_customer_id
        db.commit()
    else:
        stripe_customer_id = tenant.stripe_customer_id
    # Create checkout session directly with Stripe (one-time payment)
    import stripe
    from app.core.config import settings
    
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    success_url = f"{settings.FRONTEND_URL}/payment/success?tenant_id={tenant_id}&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{settings.FRONTEND_URL}/payment/cancel?tenant_id={tenant_id}"
    
    try:
        # Lookup plan to compute amount and embed metadata
        from app.models.plan import Plan
        plan = db.query(Plan).filter(Plan.stripe_price_id == stripe_price_id).first()
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found for the given stripe_price_id"
            )
        raw_amount = int(plan.price_monthly or 0)
        amount_cents = raw_amount if raw_amount >= 50 else raw_amount * 100
        if amount_cents < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plan amount is not configured"
            )
        amount_dollars = amount_cents / 100.0

        checkout_session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            success_url=success_url,
            cancel_url=cancel_url,
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"Plan Purchase - {plan.display_name}"},
                    "unit_amount": amount_cents
                },
                "quantity": 1
            }],
            metadata={
                "tenant_id": tenant_id,
                "purchase_type": "plan_purchase",
                "plan_id": str(plan.id),
                "amount": str(amount_dollars)
            }
        )

        return create_success_response({
            "session_id": checkout_session.id,
            "url": checkout_session.url
        }, "Checkout session created successfully")
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/start-credit-checkout-session")
def start_credit_checkout_session(
    amount: float,  # Amount in dollars
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Start Stripe checkout session for one-time credit purchase (pay as you go).
    $1 = 10 credits. User can buy any amount of credits.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    import stripe
    from app.core.config import settings
    stripe.api_key = settings.STRIPE_SECRET_KEY
    # Create Stripe customer if not exists
    if not tenant.stripe_customer_id:
        from app.services.stripe_service import StripeService
        stripe_customer_id = StripeService.create_customer(
            tenant=tenant,
            email=current_user.email,
            user=current_user
        )
        tenant.stripe_customer_id = stripe_customer_id
        db.commit()
    else:
        stripe_customer_id = tenant.stripe_customer_id
    # Create checkout session (one-time payment)
    amount_cents = int(amount * 100)
    success_url = f"{settings.FRONTEND_URL}/payment/success?tenant_id={tenant.id}&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{settings.FRONTEND_URL}/payment/cancel?tenant_id={tenant.id}"
    try:
        checkout_session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            success_url=success_url,
            cancel_url=cancel_url,
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": "Credits Purchase"},
                    "unit_amount": amount_cents
                },
                "quantity": 1
            }],
            metadata={
                "tenant_id": str(tenant.id),
                "purchase_type": "credit_purchase",
                "amount": str(amount)
            }
        )
        return create_success_response({
            "session_id": checkout_session.id,
            "url": checkout_session.url
        }, "Credit checkout session created successfully")
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/credits")
def get_tenant_credits(current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """
    Get current credits for the current user's active tenant.
    """
    tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
    print(tenant)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return create_success_response({"tenant_id": tenant.id, "credits": tenant.credits, "status": tenant.status}, "Tenant credits fetched successfully")

@router.get("/verify-payment/{session_id}")
def verify_payment(
    session_id: str,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Verify payment status using Stripe checkout session ID.
    If paid, add credits to tenant exactly once per session.
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
        
        # Get tenant ID from session metadata (the tenant who made the payment)
        session_tenant_id = session.metadata.get("tenant_id")
        print(f"VERIFY DEBUG: Session tenant ID: {session_tenant_id}")
        
        # Get the tenant who made the payment (not necessarily current user's tenant)
        tenant = db.query(Tenant).filter(Tenant.id == session_tenant_id).first()
        if not tenant:
            print(f"VERIFY DEBUG: Payment tenant not found: {session_tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {session_tenant_id} not found"
            )
        
        print(f"VERIFY DEBUG: Found payment tenant - ID: {tenant.id}")
        print(f"VERIFY DEBUG: Payment tenant credits: {tenant.credits}")
        print(f"VERIFY DEBUG: Payment tenant status: {tenant.status}")
        
        # Calculate credits that would be added
        amount_dollars = session.amount_total / 100 if session.amount_total else 0
        credits_to_add = int(amount_dollars * 10)
        
        print(f"VERIFY DEBUG: Payment status: {session.payment_status}")
        print(f"VERIFY DEBUG: Amount dollars: {amount_dollars}")
        print(f"VERIFY DEBUG: Credits to add: {credits_to_add}")
        print(f"VERIFY DEBUG: Current tenant credits: {tenant.credits or 0}")
        
        # If payment is paid, actually update the credits
        if session.payment_status == 'paid':
            print(f"VERIFY DEBUG: Payment is paid, updating credits...")
            old_credits = tenant.credits or 0
            tenant.credits = old_credits + credits_to_add
            tenant.status = 'active'
            
            print(f"VERIFY DEBUG: Old credits: {old_credits}")
            print(f"VERIFY DEBUG: New credits: {tenant.credits}")
            print(f"VERIFY DEBUG: New status: {tenant.status}")
            
            # Update subscription ID if available
            if session.subscription:
                print(f"VERIFY DEBUG: Updating subscription ID: {session.subscription}")
                tenant.stripe_subscription_id = session.subscription
            
            print(f"VERIFY DEBUG: About to commit to database...")
            db.commit()
            print(f"VERIFY DEBUG: Database commit successful")
            
            db.refresh(tenant)
            print(f"VERIFY DEBUG: After refresh - credits: {tenant.credits}, status: {tenant.status}")
        else:
            print(f"VERIFY DEBUG: Payment not paid, skipping credit update")
        
        # Get subscription information if available
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
        
        return create_success_response({
            "payment_status": session.payment_status,
            "customer_id": session.customer,
            "amount_total": session.amount_total,
            "amount_dollars": amount_dollars,
            "currency": session.currency,
            "payment_intent": session.payment_intent,
            "metadata": session.metadata,
            "subscription_id": session.subscription,
            "subscription_info": subscription_info,
            "credits_to_add": credits_to_add,
            "current_tenant_credits": tenant.credits or 0,
            "total_credits_after_payment": (tenant.credits or 0) + credits_to_add,
            "credits_updated": session.payment_status == 'paid'
        }, "Payment verification and credit update completed")
        
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
        
        # Calculate total credits from successful payments
        total_credits_earned = sum(int(p["amount_total"] * 10) for p in successful_payments)
        current_tenant_credits = tenant.credits or 0
        
        # Get subscription information
        subscription_info = None
        if tenant.stripe_subscription_id:
            try:
                subscription = stripe.Subscription.retrieve(tenant.stripe_subscription_id)
                subscription_info = {
                    "subscription_id": subscription.id,
                    "status": subscription.status,
                    "current_period_start": subscription.current_period_start,
                    "current_period_end": subscription.current_period_end,
                    "plan_id": subscription.items.data[0].price.id if subscription.items.data else None
                }
            except:
                subscription_info = {"subscription_id": tenant.stripe_subscription_id, "status": "unknown"}
        
        summary = {
            "total_payments": len(payment_history),
            "successful_payments": len(successful_payments),
            "failed_payments": len(failed_payments),
            "total_amount": total_amount,  # In dollars
            "total_amount_cents": total_amount_cents,  # In cents
            "total_amount_formatted": f"${total_amount:.2f}",  # Format as USD
            "currency": "usd" if payment_history else "usd",
            "refunds_count": len([p for p in payment_history if p["type"] == "refund"]),
            "last_payment_date": payment_history[0]["created"] if payment_history else None,
            "total_credits_earned": total_credits_earned,
            "current_tenant_credits": current_tenant_credits,
            "subscription_info": subscription_info
        }
        
        return create_success_response({
            "payment_history": payment_history,
            "summary": summary
        }, "Payment history retrieved successfully")
        
    except stripe.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@router.post("/confirm-payment-simple")
def confirm_payment_simple(
    session_id: str,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Simple payment confirmation using Stripe API.
    No webhook dependency - direct API call to confirm payment and add credits.
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
            print(f"CONFIRM DEBUG: Payment tenant not found: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_id} not found"
            )
        
        print(f"CONFIRM DEBUG: Found payment tenant - ID: {tenant.id}")
        print(f"CONFIRM DEBUG: Payment tenant credits: {tenant.credits}")
        print(f"CONFIRM DEBUG: Payment tenant status: {tenant.status}")
        
        # Get plan details
        plan = None
        if plan_id:
            plan = db.query(Plan).filter(Plan.id == plan_id).first()
        
        # Calculate credits based on payment amount: $1 = 10 credits
        credits_to_add = int(amount * 10)
        
        # Add credits to tenant
        old_credits = tenant.credits or 0
        print(f"DEBUG: Old credits: {old_credits}")
        print(f"DEBUG: Credits to add: {credits_to_add}")
        
        tenant.credits = old_credits + credits_to_add
        tenant.status = 'active'
        
        print(f"DEBUG: New credits before commit: {tenant.credits}")
        print(f"DEBUG: New status before commit: {tenant.status}")
        
        # Update subscription ID if available
        if session.subscription:
            print(f"DEBUG: Updating subscription ID: {session.subscription}")
            tenant.stripe_subscription_id = session.subscription
        
        print(f"DEBUG: About to commit to database...")
        db.commit()
        print(f"DEBUG: Database commit successful")
        
        db.refresh(tenant)
        print(f"DEBUG: After refresh - credits: {tenant.credits}, status: {tenant.status}")
        
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

@router.post("/test-credit-update")
def test_credit_update(
    credits_to_add: int = 100,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Test endpoint to manually add credits and update subscription ID.
    This helps debug the credit update issue.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    try:
        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Store old values for comparison
        old_credits = tenant.credits or 0
        old_subscription_id = tenant.stripe_subscription_id
        
        # Update credits
        tenant.credits = old_credits + credits_to_add
        tenant.status = 'active'
        
        # Update subscription ID (test value)
        import uuid
        test_subscription_id = f"sub_test_{uuid.uuid4().hex[:8]}"
        tenant.stripe_subscription_id = test_subscription_id
        
        # Commit changes
        db.commit()
        db.refresh(tenant)
        
        # Prepare response
        response_data = {
            "tenant_id": str(tenant.id),
            "old_credits": old_credits,
            "new_credits": tenant.credits,
            "credits_added": credits_to_add,
            "old_subscription_id": old_subscription_id,
            "new_subscription_id": tenant.stripe_subscription_id,
            "status": "success"
        }
        
        return create_success_response(response_data, "Test credit update successful")
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in test credit update: {str(e)}"
        )

@router.get("/tenant-status")
def get_tenant_status(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Get current tenant status including credits and subscription ID.
    This helps debug the tenant update issue.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    try:
        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        print (tenant.stripe_subscription_id    )
        print (tenant.stripe_customer_id)
        print (tenant.credits)
        print (tenant.status)
        print (tenant.name)
        print (tenant.id)
        print (tenant.created_at)
        print (tenant.updated_at)
        print (tenant.schema_name)
        # Prepare response
        response_data = {
            "tenant_id": str(tenant.id),
            "tenant_name": tenant.name,
            "status": tenant.status,
            "credits": tenant.credits,
            "stripe_customer_id": tenant.stripe_customer_id,
            "stripe_subscription_id": tenant.stripe_subscription_id,
            "created_at": tenant.created_at.isoformat() if tenant.created_at else None
        }
        
        return create_success_response(response_data, "Tenant status retrieved successfully")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting tenant status: {str(e)}"
        )

@router.post("/manual-update-credits")
def manual_update_credits(
    credits_to_add: int,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Manually update tenant credits and status.
    This bypasses all payment checks and directly updates the database.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    try:
        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Store old values
        old_credits = tenant.credits or 0
        old_status = tenant.status
        
        # Update credits and status
        print(f"MANUAL DEBUG: Old credits: {old_credits}")
        print(f"MANUAL DEBUG: Credits to add: {credits_to_add}")
        
        tenant.credits = old_credits + credits_to_add
        tenant.status = 'active'
        
        print(f"MANUAL DEBUG: New credits before commit: {tenant.credits}")
        print(f"MANUAL DEBUG: New status before commit: {tenant.status}")
        
        # Force commit
        print(f"MANUAL DEBUG: About to commit to database...")
        db.commit()
        print(f"MANUAL DEBUG: Database commit successful")
        
        db.refresh(tenant)
        print(f"MANUAL DEBUG: After refresh - credits: {tenant.credits}, status: {tenant.status}")
        
        # Verify the update
        updated_tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        print(f"MANUAL DEBUG: Verification query - credits: {updated_tenant.credits}, status: {updated_tenant.status}")
        
        response_data = {
            "tenant_id": str(tenant.id),
            "old_credits": old_credits,
            "new_credits": updated_tenant.credits,
            "old_status": old_status,
            "new_status": updated_tenant.status,
            "credits_added": credits_to_add,
            "database_updated": True
        }
        
        return create_success_response(response_data, "Manual credit update successful")
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in manual credit update: {str(e)}"
        )

@router.get("/debug-database")
def debug_database(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Debug endpoint to check database state directly.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    try:
        print(f"DEBUG DB: Current user tenant ID: {current_user.current_tenant_id}")
        
        # Direct database query
        tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        if not tenant:
            print(f"DEBUG DB: Tenant not found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        print(f"DEBUG DB: Tenant found - ID: {tenant.id}")
        print(f"DEBUG DB: Tenant credits: {tenant.credits}")
        print(f"DEBUG DB: Tenant status: {tenant.status}")
        print(f"DEBUG DB: Tenant subscription ID: {tenant.stripe_subscription_id}")
        
        # Try to update and see what happens
        print(f"DEBUG DB: Attempting to update credits...")
        old_credits = tenant.credits or 0
        tenant.credits = old_credits + 1
        print(f"DEBUG DB: Set credits to: {tenant.credits}")
        
        print(f"DEBUG DB: About to commit...")
        db.commit()
        print(f"DEBUG DB: Commit successful")
        
        # Refresh and check
        db.refresh(tenant)
        print(f"DEBUG DB: After refresh - credits: {tenant.credits}")
        
        response_data = {
            "tenant_id": str(tenant.id),
            "credits": tenant.credits,
            "status": tenant.status,
            "subscription_id": tenant.stripe_subscription_id,
            "database_working": True
        }
        
        return create_success_response(response_data, "Database debug successful")
        
    except Exception as e:
        print(f"DEBUG DB ERROR: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database debug error: {str(e)}"
        )

@router.get("/check-credits")
def check_credits(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Simple endpoint to check current credits without any updates.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    
    try:
        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        print(f"CHECK CREDITS: Tenant ID: {tenant.id}")
        print(f"CHECK CREDITS: Current credits: {tenant.credits}")
        print(f"CHECK CREDITS: Current status: {tenant.status}")
        print(f"CHECK CREDITS: Subscription ID: {tenant.stripe_subscription_id}")
        
        response_data = {
            "tenant_id": str(tenant.id),
            "credits": tenant.credits,
            "status": tenant.status,
            "subscription_id": tenant.stripe_subscription_id,
            "timestamp": "current"
        }
        
        return create_success_response(response_data, "Current credits retrieved")
        
    except Exception as e:
        print(f"CHECK CREDITS ERROR: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking credits: {str(e)}"
        )

@router.get("/debug-user-tenant")
def debug_user_tenant(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Debug endpoint to check user-tenant mapping and all tenants.
    """
    try:
        print(f"USER DEBUG: Current user ID: {current_user.id}")
        print(f"USER DEBUG: Current user email: {current_user.email}")
        print(f"USER DEBUG: Current user tenant ID: {current_user.current_tenant_id}")
        
        # Get current user's tenant
        current_tenant = None
        if current_user.current_tenant_id:
            current_tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
            if current_tenant:
                print(f"USER DEBUG: Current tenant found - ID: {current_tenant.id}")
                print(f"USER DEBUG: Current tenant credits: {current_tenant.credits}")
                print(f"USER DEBUG: Current tenant status: {current_tenant.status}")
            else:
                print(f"USER DEBUG: Current tenant not found in database")
        
        # Get all tenants for this user
        user_tenants = db.query(Tenant).join(
            user_tenant_association
        ).filter(
            user_tenant_association.c.user_id == current_user.id
        ).all()
        
        print(f"USER DEBUG: User has {len(user_tenants)} tenants")
        for i, tenant in enumerate(user_tenants):
            print(f"USER DEBUG: Tenant {i+1} - ID: {tenant.id}, Credits: {tenant.credits}, Status: {tenant.status}")
        
        # Get all tenants in database (for debugging)
        all_tenants = db.query(Tenant).all()
        print(f"USER DEBUG: Total tenants in database: {len(all_tenants)}")
        for i, tenant in enumerate(all_tenants):
            print(f"USER DEBUG: DB Tenant {i+1} - ID: {tenant.id}, Credits: {tenant.credits}, Status: {tenant.status}")
        
        response_data = {
            "user_id": str(current_user.id),
            "user_email": current_user.email,
            "current_tenant_id": str(current_user.current_tenant_id) if current_user.current_tenant_id else None,
            "current_tenant": {
                "id": str(current_tenant.id) if current_tenant else None,
                "name": current_tenant.name if current_tenant else None,
                "credits": current_tenant.credits if current_tenant else None,
                "status": current_tenant.status if current_tenant else None
            } if current_tenant else None,
            "user_tenants": [
                {
                    "id": str(tenant.id),
                    "name": tenant.name,
                    "credits": tenant.credits,
                    "status": tenant.status
                } for tenant in user_tenants
            ],
            "all_tenants_count": len(all_tenants)
        }
        
        return create_success_response(response_data, "User-tenant debug completed")
        
    except Exception as e:
        print(f"USER DEBUG ERROR: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User-tenant debug error: {str(e)}"
        )

@router.post("/update-credits-for-tenant")
def update_credits_for_tenant(
    tenant_id: str,
    credits_to_add: int,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Update credits for a specific tenant ID.
    This helps when payment was made for a different tenant.
    """
    try:
        print(f"TENANT UPDATE: Updating credits for tenant: {tenant_id}")
        print(f"TENANT UPDATE: Credits to add: {credits_to_add}")
        
        # Get the specific tenant
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            print(f"TENANT UPDATE: Tenant not found: {tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_id} not found"
            )
        
        print(f"TENANT UPDATE: Found tenant - ID: {tenant.id}")
        print(f"TENANT UPDATE: Current credits: {tenant.credits}")
        print(f"TENANT UPDATE: Current status: {tenant.status}")
        
        # Update credits
        old_credits = tenant.credits or 0
        tenant.credits = old_credits + credits_to_add
        tenant.status = 'active'
        
        print(f"TENANT UPDATE: Old credits: {old_credits}")
        print(f"TENANT UPDATE: New credits: {tenant.credits}")
        print(f"TENANT UPDATE: New status: {tenant.status}")
        
        # Commit changes
        db.commit()
        print(f"TENANT UPDATE: Database commit successful")
        
        # Refresh and verify
        db.refresh(tenant)
        print(f"TENANT UPDATE: After refresh - credits: {tenant.credits}, status: {tenant.status}")
        
        response_data = {
            "tenant_id": str(tenant.id),
            "tenant_name": tenant.name,
            "old_credits": old_credits,
            "new_credits": tenant.credits,
            "credits_added": credits_to_add,
            "status": tenant.status,
            "updated": True
        }
        
        return create_success_response(response_data, f"Credits updated for tenant {tenant.name}")
        
    except Exception as e:
        print(f"TENANT UPDATE ERROR: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating tenant credits: {str(e)}"
        )

@router.get("/test-simple")
def test_simple():
    """
    Simple test endpoint to check if server is working.
    """
    return {"status": "success", "message": "Server is working correctly"}