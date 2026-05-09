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
from datetime import datetime, date, timedelta, timezone
import uuid

class BillingService:
    
    # Default subscription period in days (e.g. 1 month)
    DEFAULT_PERIOD_DAYS = 30

    @staticmethod
    def get_or_create_subscription(db: Session, user_id: uuid.UUID, crm_type: Optional[str] = None) -> Subscription:
        """Get existing subscription for user (and optional crm_type) or create a free one for default usage."""
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            (Subscription.crm_type == crm_type) if crm_type is not None else (Subscription.crm_type.is_(None))
        ).first()
        
        if not subscription:
            free_plan = db.query(Plan).filter(Plan.name == "free").first()
            if not free_plan:
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
                status="active",
                crm_type=crm_type
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
        Verify Stripe session and update subscription. No credits added for plan_purchase.
        Returns result dict or None if session not paid/valid.
        """
        from app.api.deps import is_session_already_credited, mark_session_credited
        from app.core.config import settings
        from app.core.logger import logger

        if is_session_already_credited(session_id):
            logger.info(f"Session {session_id} already processed.")
            return {"status": "already_processed"}

        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status != "paid":
                logger.warning(f"Session {session_id} not paid yet (status: {session.payment_status})")
                return None

            metadata = StripeService.stripe_metadata_as_dict(session["metadata"])
            user_id_str = metadata.get('user_id')
            tenant_id_str = metadata.get('tenant_id')
            plan_id_str = metadata.get('plan_id')
            purchase_type = metadata.get('purchase_type') or 'credit_purchase'
            crm_type = metadata.get('crm_type')

            if not tenant_id_str:
                logger.warning(f"Session {session_id} missing tenant_id in metadata")
                return None

            tenant_id = uuid.UUID(tenant_id_str)
            user_id = uuid.UUID(user_id_str) if user_id_str else None
            plan_id = uuid.UUID(plan_id_str) if plan_id_str else None

            # Plan purchase: update subscription only, NO credits
            if purchase_type == 'plan_purchase' and user_id and plan_id:
                # If crm_type missing/empty in metadata, get from plan so we create correct subscription row
                if not crm_type and plan_id:
                    plan_row = db.query(Plan).filter(Plan.id == plan_id).first()
                    if plan_row and plan_row.crm_type:
                        crm_type = plan_row.crm_type
                stripe_sub_id = session["subscription"]  # set when mode=subscription
                period_start, period_end = None, None
                if stripe_sub_id:
                    try:
                        sub = stripe.Subscription.retrieve(stripe_sub_id)
                        if sub.current_period_start:
                            period_start = datetime.fromtimestamp(sub.current_period_start, tz=timezone.utc)
                        if sub.current_period_end:
                            period_end = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
                    except Exception:
                        pass
                BillingService.update_subscription(
                    db=db,
                    user_id=user_id,
                    plan_id=plan_id,
                    status="active",
                    stripe_subscription_id=stripe_sub_id,
                    stripe_customer_id=session["customer"],
                    stripe_session_id=session_id,
                    crm_type=crm_type,
                    current_period_start=period_start,
                    current_period_end=period_end
                )
                logger.info(f"Subscription updated for user {user_id} (crm_type={crm_type}) via sync - no credits added")
                mark_session_credited(session_id)
                return {
                    "status": "success",
                    "credits_added": 0,
                    "crm_type": crm_type,
                    "purchase_type": purchase_type,
                    "message": "Plan subscription updated. No credits added for plan purchase."
                }

            # Credit purchase: add credits to tenant
            credits_to_add = 0
            amount_total_cents = session["amount_total"] or 0
            amount_dollars = float(amount_total_cents) / 100.0
            if purchase_type == 'credit_purchase':
                credits_to_add = int(amount_dollars * 10)

            if credits_to_add > 0:
                tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
                if tenant:
                    tenant.credits = (tenant.credits or 0) + credits_to_add
                    tenant.status = 'active'
                    db.commit()
                    logger.info(f"Added {credits_to_add} credits to tenant {tenant_id} (credit_purchase)")

            mark_session_credited(session_id)
            return {
                "status": "success",
                "credits_added": credits_to_add,
                "purchase_type": purchase_type
            }
        except Exception as e:
            from app.core.logger import logger
            logger.error(f"Error syncing payment status for session {session_id}: {str(e)}")
            return None

    @staticmethod
    def has_active_paid_subscription(db: Session, user_id: uuid.UUID) -> bool:
        """Check if user has at least one active paid (CRM) subscription with valid period."""
        now = datetime.now(timezone.utc)
        subscription = db.query(Subscription).join(Plan).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active",
            Subscription.crm_type.isnot(None),
            Plan.price_monthly > 0,
            (Subscription.current_period_end.is_(None)) | (Subscription.current_period_end > now)
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
        crm_type: Optional[str] = None,
        current_period_start: Optional[datetime] = None,
        current_period_end: Optional[datetime] = None
    ) -> Subscription:
        """Update or create user subscription for this CRM. Sets current_period_start/end from args or default 30 days."""
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            (Subscription.crm_type == crm_type) if crm_type is not None else Subscription.crm_type.is_(None)
        ).first()

        now = datetime.now(timezone.utc)
        period_start = current_period_start if current_period_start is not None else now
        period_end = current_period_end if current_period_end is not None else (now + timedelta(days=BillingService.DEFAULT_PERIOD_DAYS))

        if not subscription:
            subscription = Subscription(user_id=user_id, crm_type=crm_type)
            db.add(subscription)

        subscription.plan_id = plan_id
        subscription.status = status
        subscription.current_period_start = period_start
        subscription.current_period_end = period_end
        if stripe_subscription_id:
            subscription.stripe_subscription_id = stripe_subscription_id
        if stripe_customer_id:
            subscription.stripe_customer_id = stripe_customer_id
        if stripe_session_id:
            subscription.stripe_session_id = stripe_session_id
        if crm_type is not None:
            subscription.crm_type = crm_type
        subscription.updated_at = now

        db.commit()
        db.refresh(subscription)
        return subscription

    @staticmethod
    def has_crm_access(db: Session, user_id: uuid.UUID, crm_type: str) -> bool:
        """Check if user has active subscription for this CRM type and period has not ended."""
        now = datetime.now(timezone.utc)
        subscription = db.query(Subscription).join(Plan).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active",
            Subscription.crm_type == crm_type,
            Plan.price_monthly > 0,
            (Subscription.current_period_end.is_(None)) | (Subscription.current_period_end > now)
        ).first()
        return subscription is not None
