from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from app.models.tenant import Tenant
from app.models.subscription import Subscription
from app.models.plan import Plan
from app.models.usage_record import UsageRecord
from app.models.agent import Agent
from app.core.config import settings
from app.services.stripe_service import StripeService
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timedelta
import uuid

class BillingService:
    
    @staticmethod
    def get_or_create_subscription(db: Session, user_id: uuid.UUID) -> Subscription:
        """Get existing subscription or create a free one"""
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.crm_type == None  # Default usage
        ).first()
        
        if not subscription:
            # Create free plan subscription
            free_plan = db.query(Plan).filter(Plan.name == "free").first()
            if not free_plan:
                # Create default free plan if it doesn't exist
                free_plan = Plan(
                    name="free",
                    display_name="Free Plan",
                    description="Free tier with limited features",
                    price_monthly=0,
                    is_active=True
                )
                db.add(free_plan)
                db.commit()
                db.refresh(free_plan)
            
            subscription = Subscription(
                user_id=user_id,
                plan_id=free_plan.id,
                status="active"
            )
            db.add(subscription)
            db.commit()
            db.refresh(subscription)
        
        return subscription
    
    @staticmethod
    def increment_agent_usage(db: Session, user_id: uuid.UUID) -> None:
        """Increment agent creation count for current month"""
        pass

    @staticmethod
    def sync_payment_status(db: Session, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Verify Stripe session and update subscription/credits.
        Returns result dict or None if session not paid/valid.
        """
        from app.api.deps import is_session_already_credited, mark_session_credited
        import stripe
        from app.core.config import settings
        from app.core.logger import logger
        from app.models.tenant import Tenant
        from app.models.plan import Plan

        if is_session_already_credited(session_id):
            logger.info(f"ℹ️ Session {session_id} already credited.")
            return {"status": "already_processed"}

        stripe.api_key = settings.STRIPE_SECRET_KEY
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status != "paid":
                logger.warning(f"⚠️ Session {session_id} not paid yet (status: {session.payment_status})")
                return None

            metadata = session.get('metadata') or {}
            user_id_str = metadata.get('user_id')
            tenant_id_str = metadata.get('tenant_id')
            plan_id_str = metadata.get('plan_id')
            purchase_type = metadata.get('purchase_type') or 'credit_purchase'
            crm_type = metadata.get('crm_type')

            if not tenant_id_str:
                logger.warning(f"⚠️ Session {session_id} missing tenant ID in metadata")
                return None

            tenant_id = uuid.UUID(tenant_id_str)
            user_id = uuid.UUID(user_id_str) if user_id_str else None
            plan_id = uuid.UUID(plan_id_str) if plan_id_str else None

            # 🔄 Update subscription if it's a plan purchase
            if purchase_type == 'plan_purchase' and user_id and plan_id:
                BillingService.update_subscription(
                    db=db,
                    user_id=user_id,
                    plan_id=plan_id,
                    status="active",
                    stripe_customer_id=session.get('customer'),
                    stripe_session_id=session_id,
                    crm_type=crm_type
                )
                logger.info(f"✅ Subscription updated for user {user_id} via sync")

            # 💰 Add credits
            credits_to_add = 0
            amount_total_cents = session.get('amount_total') or 0
            amount_dollars = float(amount_total_cents) / 100.0

            if purchase_type == 'credit_purchase':
                # $1 = 10 credits
                credits_to_add = int(amount_dollars * 10)
            elif purchase_type == 'plan_purchase':
                if amount_dollars == 2.0:
                    credits_to_add = 20
                elif plan_id:
                    plan = db.query(Plan).filter(Plan.id == plan_id).first()
                    if plan and hasattr(plan, 'credits') and plan.credits:
                        credits_to_add = plan.credits
            
            if credits_to_add > 0:
                tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
                if tenant:
                    tenant.credits = (tenant.credits or 0) + credits_to_add
                    tenant.status = 'active'
                    db.commit()
                    logger.info(f"✅ Added {credits_to_add} credits to tenant {tenant_id} (Type: {purchase_type})")

            mark_session_credited(session_id)
            return {
                "status": "success",
                "credits_added": credits_to_add,
                "crm_type": crm_type,
                "purchase_type": purchase_type
            }

        except Exception as e:
            logger.error(f"❌ Error syncing payment status for session {session_id}: {str(e)}")
            return None

    @staticmethod
    def has_active_paid_subscription(db: Session, user_id: uuid.UUID) -> bool:
        """Check if user has an active paid subscription"""
        subscription = db.query(Subscription).join(Plan).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active",
            Plan.price_monthly > 0
        ).first()
        
        return subscription is not None

    @staticmethod
    def update_subscription(
        db: Session, 
        user_id: uuid.UUID, 
        plan_id: uuid.UUID, 
        status: str = "active",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        stripe_session_id: Optional[str] = None,
        crm_type: Optional[str] = None
    ) -> Subscription:
        """Update or create user subscription"""
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.crm_type == crm_type
        ).first()
        
        if not subscription:
            subscription = Subscription(user_id=user_id)
            db.add(subscription)
        
        subscription.plan_id = plan_id
        subscription.status = status
        if stripe_subscription_id:
            subscription.stripe_subscription_id = stripe_subscription_id
        if stripe_customer_id:
            subscription.stripe_customer_id = stripe_customer_id
        if stripe_session_id:
            subscription.stripe_session_id = stripe_session_id
        if crm_type:
            subscription.crm_type = crm_type
        
        subscription.updated_at = datetime.now()
        
        db.commit()
        db.refresh(subscription)
        return subscription

    @staticmethod
    def has_crm_access(db: Session, user_id: uuid.UUID, crm_type: str) -> bool:
        """Check if user has access to a specific CRM type based on their subscription"""
        subscription = db.query(Subscription).join(Plan).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active",
            Subscription.crm_type == crm_type,
            Plan.price_monthly > 0
        ).first()
        
        return subscription is not None
