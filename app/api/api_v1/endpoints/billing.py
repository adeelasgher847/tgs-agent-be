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
            # 🎉 Payment Webhook Received - Process Credits
            print("=" * 60)
            print(f"📞 STRIPE WEBHOOK RECEIVED: checkout.session.completed")
            print("=" * 60)
            
            # Process credits directly on webhook with idempotency
            session = event.get('data', {}).get('object', {}) or {}
            session_id = session.get('id')
            amount_total_cents = session.get('amount_total') or 0
            amount_dollars = float(amount_total_cents) / 100.0
            
            print(f"   Session ID: {session_id}")
            print(f"   Payment Amount: ${amount_dollars:.2f}")
            
            if session_id and is_session_already_credited(session_id):
                print(f"⚠️ Session {session_id} already processed - skipping")
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
                # Get payment amount from session
                amount_total_cents = session.get('amount_total') or 0
                amount_dollars = float(amount_total_cents) / 100.0
                
                # 🎯 $2 Plan Payment → Add 20 Credits
                if amount_dollars == 2.0:
                    credits_to_add = 20
                    print(f"✅ $2 Plan Payment Detected - Adding 20 credits to tenant {tenant_uuid}")
                else:
                    # Fallback: Check plan.credits if it exists (for other plans)
                    plan_id_str = metadata.get('plan_id')
                    try:
                        plan_uuid = uuid.UUID(plan_id_str) if plan_id_str else None
                    except Exception:
                        plan_uuid = None
                    if plan_uuid:
                        plan = db.query(Plan).filter(Plan.id == plan_uuid).first()
                        if plan and getattr(plan, 'credits', None) is not None:
                            credits_to_add = int(plan.credits)
                        else:
                            print(f"⚠️ Plan {plan_uuid} has no credits field - no credits added")
                    else:
                        print(f"⚠️ No plan_id in metadata for plan purchase - no credits added")

            if credits_to_add and credits_to_add > 0:
                old_credits = tenant.credits or 0
                tenant.credits = old_credits + credits_to_add
                new_credits = tenant.credits
                
                # Mark tenant active on successful purchase
                if getattr(tenant, 'status', None) in (None, 'pending_payment', 'inactive'):
                    tenant.status = 'active'
                
                db.commit()
                db.refresh(tenant)
                
                # 🎉 Payment Success Notification
                print("=" * 60)
                print(f"✅ PAYMENT SUCCESS - Credits Added!")
                print(f"   Tenant ID: {tenant_uuid}")
                print(f"   Purchase Type: {purchase_type}")
                print(f"   Credits Added: {credits_to_add}")
                print(f"   Previous Credits: {old_credits}")
                print(f"   New Credits Balance: {new_credits}")
                print(f"   Tenant Status: {tenant.status}")
                print("=" * 60)
                
                if session_id:
                    mark_session_credited(session_id)
                
                return {
                    "status": "success", 
                    "credits_added": credits_to_add,
                    "previous_credits": old_credits,
                    "new_credits_balance": new_credits,
                    "tenant_status": tenant.status,
                    "message": f"Payment successful! Added {credits_to_add} credits to account."
                }
            else:
                print(f"⚠️ No credits to add - purchase_type: {purchase_type}, amount: {session.get('amount_total', 0)}")
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
