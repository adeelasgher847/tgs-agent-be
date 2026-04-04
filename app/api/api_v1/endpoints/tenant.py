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

from sqlalchemy import update
from app.core.logger import logger
from app.services.stripe_service import StripeService

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
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Ensure credits are returned as an integer (no floating/decimal value)
    credits_int = int(tenant.credits or 0)

    return create_success_response(
        {"tenant_id": tenant.id, "credits": credits_int, "status": tenant.status},
        "Tenant credits fetched successfully"
    )

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
        
        # Verify this session belongs to the current tenant
        meta = StripeService.stripe_metadata_as_dict(session["metadata"])
        if meta.get("tenant_id") != str(current_user.current_tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This payment session does not belong to your tenant"
            )
        
        return create_success_response({
            "payment_status": session.payment_status,
            "customer_id": session.customer,
            "amount_total": session.amount_total,
            "currency": session.currency,
            "payment_intent": session.payment_intent,
            "metadata": meta
        }, "Payment verification fetched")
        
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
    
    # Get the latest subscription for the user (with a session ID)
    from app.models.subscription import Subscription
    subscription = (
        db.query(Subscription)
        .filter(
            Subscription.user_id == current_user.id,
            Subscription.stripe_session_id.isnot(None),
        )
        .order_by(Subscription.updated_at.desc())
        .first()
    )
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription with payment session found for this user"
        )
    
    if not subscription.stripe_session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No payment session found for this user"
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
            logger.error(f"Error getting checkout sessions: {str(e)}", exc_info=True)
        
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
            logger.error(f"Error getting invoices: {str(e)}", exc_info=True)
        
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
