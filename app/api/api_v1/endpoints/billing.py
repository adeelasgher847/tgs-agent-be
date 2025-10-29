from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.api.deps import get_db, is_session_already_credited, mark_session_credited
from app.models.tenant import Tenant
from app.models.plan import Plan
from app.services.stripe_service import StripeService
import uuid

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
            # Process credits directly on webhook with idempotency
            session = event.get('data', {}).get('object', {}) or {}
            session_id = session.get('id')
            if session_id and is_session_already_credited(session_id):
                return {"status": "already_processed"}

            metadata = session.get('metadata') or {}
            tenant_id_str = metadata.get('tenant_id')
            if not tenant_id_str:
                # Missing tenant context; ignore gracefully
                return {"status": "ignored", "reason": "no_tenant_id"}

            try:
                tenant_uuid = uuid.UUID(tenant_id_str)
            except Exception:
                return {"status": "ignored", "reason": "invalid_tenant_id"}

            tenant = db.query(Tenant).filter(Tenant.id == tenant_uuid).first()
            if not tenant:
                return {"status": "ignored", "reason": "tenant_not_found"}

            purchase_type = metadata.get('purchase_type') or 'credit_purchase'
            credits_to_add = 0

            if purchase_type == 'credit_purchase':
                # $1 = 10 credits mapping
                amount_total_cents = session.get('amount_total') or 0
                try:
                    credits_to_add = int((float(amount_total_cents) / 100.0) * 10)
                except Exception:
                    credits_to_add = 0
            elif purchase_type == 'plan_purchase':
                plan_id_str = metadata.get('plan_id')
                try:
                    plan_uuid = uuid.UUID(plan_id_str) if plan_id_str else None
                except Exception:
                    plan_uuid = None
                plan_credits = None
                if plan_uuid:
                    plan = db.query(Plan).filter(Plan.id == plan_uuid).first()
                    # Prefer plan.credits if schema supports it
                    if plan and isinstance(getattr(plan, 'credits', None), int) and plan.credits > 0:
                        plan_credits = int(plan.credits)
                if plan_credits:
                    credits_to_add = plan_credits
                else:
                    # Fallback to $1 = 10 credits based on amount_total
                    amount_total_cents = session.get('amount_total') or 0
                    try:
                        credits_to_add = int((float(amount_total_cents) / 100.0) * 10)
                    except Exception:
                        credits_to_add = 0

            if credits_to_add and credits_to_add > 0:
                tenant.credits = (tenant.credits or 0) + credits_to_add
                # Mark tenant active on successful purchase
                if getattr(tenant, 'status', None) in (None, 'pending_payment', 'inactive'):
                    tenant.status = 'active'
                db.commit()
                if session_id:
                    mark_session_credited(session_id)
                return {"status": "success", "credits_added": credits_to_add}
            else:
                return {"status": "ignored", "reason": "no_credits_computed"}
        else:
            print(f"Unhandled event type: {event_type}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error handling webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
