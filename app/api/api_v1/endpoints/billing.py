from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.api.deps import get_db, is_session_already_credited, mark_session_credited
from app.models.tenant import Tenant
from app.models.plan import Plan
from app.services.stripe_service import StripeService
import uuid
from datetime import datetime

router = APIRouter()

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events"""
    print("🔥🔥🔥 STRIPE WEBHOOK CALLED! 🔥🔥🔥")
    print(f"Timestamp: {datetime.now()}")
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    print(f"Request headers: {dict(request.headers)}")
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    print(f"Payload length: {len(payload)}")
    print(f"Signature header: {sig_header}")
    
    if not sig_header:
        print("❌ Missing stripe-signature header")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing stripe-signature header"
        )
    
    try:
        event = StripeService.construct_webhook_event(payload, sig_header)
        print(f"✅ Event constructed successfully")
        print(f"Event type: {event.get('type')}")
        print(f"Event ID: {event.get('id')}")
    except Exception as e:
        print(f"❌ Error constructing event: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Handle the event
    try:
        event_type = event['type']
        print(f"📋 Processing event type: {event_type}")
        
        if event_type == 'checkout.session.completed':
            print("🎯 Processing checkout.session.completed event")
            # Process credits directly on webhook with idempotency
            session = event.get('data', {}).get('object', {}) or {}
            session_id = session.get('id')
            print(f"📋 Session ID: {session_id}")
            
            metadata = session.get('metadata') or {}
            print(f"📋 Metadata: {metadata}")
            
            tenant_id_str = metadata.get('tenant_id')
            print(f"📋 Tenant ID: {tenant_id_str}")
            
            if session_id and is_session_already_credited(session_id):
                print(f"⚠️ Session {session_id} already processed")
                return {"status": "already_processed"}

            if not tenant_id_str:
                print("❌ Missing tenant_id in metadata")
                # Missing tenant context; ignore gracefully
                return {"status": "ignored", "reason": "no_tenant_id"}

            try:
                tenant_uuid = uuid.UUID(tenant_id_str)
                print(f"✅ Valid tenant UUID: {tenant_uuid}")
            except Exception as e:
                print(f"❌ Invalid tenant UUID: {tenant_id_str}, error: {e}")
                return {"status": "ignored", "reason": "invalid_tenant_id"}

            tenant = db.query(Tenant).filter(Tenant.id == tenant_uuid).first()
            if not tenant:
                print(f"❌ Tenant not found: {tenant_uuid}")
                return {"status": "ignored", "reason": "tenant_not_found"}
            
            print(f"✅ Tenant found: {tenant.id}, current credits: {tenant.credits}")

            purchase_type = metadata.get('purchase_type') or 'credit_purchase'
            print(f"📋 Purchase type: {purchase_type}")
            credits_to_add = 0

            if purchase_type == 'credit_purchase':
                # $1 = 10 credits mapping
                amount_total_cents = session.get('amount_total') or 0
                print(f"📋 Amount total cents: {amount_total_cents}")
                try:
                    credits_to_add = int((float(amount_total_cents) / 100.0) * 10)
                    print(f"✅ Credits to add: {credits_to_add}")
                except Exception as e:
                    print(f"❌ Error calculating credits: {e}")
                    credits_to_add = 0
            elif purchase_type == 'plan_purchase':
                plan_id_str = metadata.get('plan_id')
                print(f"📋 Plan ID: {plan_id_str}")
                try:
                    plan_uuid = uuid.UUID(plan_id_str) if plan_id_str else None
                except Exception:
                    plan_uuid = None
                if plan_uuid:
                    plan = db.query(Plan).filter(Plan.id == plan_uuid).first()
                    if plan and getattr(plan, 'credits', None) is not None:
                        credits_to_add = int(plan.credits)
                        print(f"✅ Plan credits to add: {credits_to_add}")

            print(f"📋 Final credits to add: {credits_to_add}")
            if credits_to_add and credits_to_add > 0:
                old_credits = tenant.credits or 0
                tenant.credits = old_credits + credits_to_add
                print(f"✅ Updated credits: {old_credits} + {credits_to_add} = {tenant.credits}")
                
                # Mark tenant active on successful purchase
                if getattr(tenant, 'status', None) in (None, 'pending_payment', 'inactive'):
                    tenant.status = 'active'
                    print(f"✅ Updated tenant status to: active")
                
                db.commit()
                print(f"✅ Database committed")
                
                if session_id:
                    mark_session_credited(session_id)
                    print(f"✅ Session marked as credited: {session_id}")
                
                print(f"🎉 SUCCESS: Added {credits_to_add} credits to tenant {tenant.id}")
                return {"status": "success", "credits_added": credits_to_add}
            else:
                print(f"❌ No credits computed or credits_to_add is 0")
                return {"status": "ignored", "reason": "no_credits_computed"}
        else:
            print(f"⚠️ Unhandled event type: {event_type}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error handling webhook: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
