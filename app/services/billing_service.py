from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.tenant import Tenant
from app.models.stripe_checkout_fulfillment import StripeCheckoutFulfillment
from app.models.subscription import Subscription
from app.models.plan import Plan
from app.models.usage_record import UsageRecord
from app.services.stripe_service import StripeService
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import uuid

class BillingService:

    # Default subscription period in days (e.g. 1 month)
    DEFAULT_PERIOD_DAYS = 30

    @staticmethod
    def get_or_create_subscription(
        db: Session,
        user_id: uuid.UUID,
        crm_type: str | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> Subscription:
        """Get existing subscription (for user+crm_type, or tenant's core subscription) or create a free one.

        When `tenant_id` is provided, look up/create the tenant-scoped core-product
        subscription (`tenant_id` match, `crm_type IS NULL`) instead of the legacy
        `user_id` + `crm_type` row. When `tenant_id` is None, behavior is unchanged.
        """
        if tenant_id is not None:
            subscription = db.query(Subscription).filter(
                Subscription.tenant_id == tenant_id,
                Subscription.crm_type.is_(None),
            ).first()
        else:
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

            if tenant_id is not None:
                subscription = Subscription(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    plan_id=free_plan.id,
                    status="active",
                    crm_type=None,
                )
            else:
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
    def _claim_checkout_session(
        db: Session, session_id: str, stripe_event_id: str | None = None
    ) -> bool:
        """Reserve fulfillment for this session. False if already processed."""
        db.add(
            StripeCheckoutFulfillment(
                checkout_session_id=session_id,
                stripe_event_id=stripe_event_id,
            )
        )
        try:
            db.flush()
            return True
        except IntegrityError:
            db.rollback()
            return False

    @staticmethod
    def sync_payment_status(
        db: Session,
        session_id: str,
        stripe_event_id: str | None = None,
    ) -> Dict[str, Any] | None:
        """
        Verify Stripe session and update subscription. No credits added for plan_purchase.
        Returns result dict or None if session not paid/valid.
        """
        from app.core.config import settings
        from app.core.logger import logger

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

            if not BillingService._claim_checkout_session(db, session_id, stripe_event_id):
                logger.info(f"Session {session_id} already processed.")
                return {"status": "already_processed"}

            # Plan purchase: update subscription (core plans additionally grant monthly credits)
            if purchase_type == 'plan_purchase' and user_id and plan_id:
                plan_row = db.query(Plan).filter(Plan.id == plan_id).first()
                # If crm_type missing/empty in metadata, get from plan so we create correct subscription row
                if not crm_type and plan_row and plan_row.crm_type:
                    crm_type = plan_row.crm_type

                is_core_plan = plan_row is not None and plan_row.crm_type is None

                stripe_sub_id = session["subscription"]  # set when mode=subscription
                period_start, period_end = None, None
                if stripe_sub_id:
                    try:
                        sub = stripe.Subscription.retrieve(stripe_sub_id)
                        if sub.current_period_start:
                            period_start = datetime.fromtimestamp(sub.current_period_start, tz=timezone.utc)
                        if sub.current_period_end:
                            period_end = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
                    except Exception as exc:
                        logger.warning(f"Failed to retrieve Stripe subscription period for {stripe_sub_id}: {exc}")

                if is_core_plan:
                    BillingService.update_subscription(
                        db=db,
                        user_id=user_id,
                        plan_id=plan_id,
                        status="active",
                        stripe_subscription_id=stripe_sub_id,
                        stripe_customer_id=session["customer"],
                        stripe_session_id=session_id,
                        current_period_start=period_start,
                        current_period_end=period_end,
                        tenant_id=tenant_id,
                    )
                    credits_granted = Decimal("0")
                    if plan_row.monthly_credits:
                        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
                        if tenant:
                            credits_granted = Decimal(str(plan_row.monthly_credits))
                            tenant.credits = (tenant.credits or Decimal("0")) + credits_granted
                            tenant.status = 'active'
                            db.commit()
                    logger.info(
                        f"Core plan subscription updated for tenant {tenant_id} (plan={plan_row.name}) "
                        f"via sync - granted {credits_granted} credits"
                    )
                    return {
                        "status": "success",
                        "credits_added": float(credits_granted),
                        "crm_type": None,
                        "purchase_type": purchase_type,
                        "message": "Core plan subscription updated and monthly credits granted."
                    }

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
                    logger.info(f"Added {credits_to_add} credits to tenant {tenant_id} (credit_purchase)")

            db.commit()
            return {
                "status": "success",
                "credits_added": credits_to_add,
                "purchase_type": purchase_type
            }
        except Exception as e:
            from app.core.logger import logger
            logger.error(f"Error syncing payment status for session {session_id}: {str(e)}")
            db.rollback()
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
        stripe_subscription_id: str | None = None,
        stripe_customer_id: str | None = None,
        stripe_session_id: str | None = None,
        crm_type: str | None = None,
        current_period_start: datetime | None = None,
        current_period_end: datetime | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> Subscription:
        """Update or create a subscription. Sets current_period_start/end from args or default 30 days.

        When `tenant_id` is provided, the subscription is looked up/created by
        `tenant_id` + `crm_type IS NULL` (the tenant-scoped core-product subscription)
        instead of by `user_id` + `crm_type`. When `tenant_id` is None, behavior is
        unchanged from before.
        """
        if tenant_id is not None:
            subscription = db.query(Subscription).filter(
                Subscription.tenant_id == tenant_id,
                Subscription.crm_type.is_(None),
            ).first()
        else:
            subscription = db.query(Subscription).filter(
                Subscription.user_id == user_id,
                (Subscription.crm_type == crm_type) if crm_type is not None else Subscription.crm_type.is_(None)
            ).first()

        now = datetime.now(timezone.utc)
        period_start = current_period_start if current_period_start is not None else now
        period_end = current_period_end if current_period_end is not None else (now + timedelta(days=BillingService.DEFAULT_PERIOD_DAYS))

        if not subscription:
            if tenant_id is not None:
                subscription = Subscription(user_id=user_id, tenant_id=tenant_id, crm_type=None)
            else:
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
        if tenant_id is None and crm_type is not None:
            subscription.crm_type = crm_type
        subscription.updated_at = now

        db.commit()
        db.refresh(subscription)
        return subscription

    @staticmethod
    def get_workspace_subscription(db: Session, tenant_id: uuid.UUID) -> Subscription | None:
        """Return the tenant's active core-product (non-CRM) subscription, if any."""
        return db.query(Subscription).filter(
            Subscription.tenant_id == tenant_id,
            Subscription.crm_type.is_(None),
            Subscription.status == "active",
        ).first()

    @staticmethod
    def get_included_minutes_status(
        db: Session, tenant_id: uuid.UUID
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Return (included_minutes, used_this_cycle, remaining) for the tenant's current billing cycle.

        If the tenant has an active core subscription, the cycle window is
        `current_period_start` -> now and `included_minutes` comes from the plan.
        Otherwise (pure pay-as-you-go), included_minutes is 0 and the cycle window
        is calendar-month-to-date (matching the existing `get_workspace_usage` convention).
        """
        from sqlalchemy import func

        subscription = BillingService.get_workspace_subscription(db, tenant_id)

        if subscription is not None:
            plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
            included_minutes = Decimal(str(plan.included_minutes)) if plan and plan.included_minutes is not None else Decimal("0")
            cycle_start = subscription.current_period_start or datetime.now(timezone.utc).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
        else:
            included_minutes = Decimal("0")
            cycle_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        usage_sum = db.query(func.sum(UsageRecord.billable_minutes)).filter(
            UsageRecord.workspace_id == tenant_id,
            UsageRecord.recorded_at >= cycle_start,
        ).scalar() or Decimal("0")
        used_this_cycle = Decimal(str(usage_sum))

        remaining = max(Decimal("0"), included_minutes - used_this_cycle)
        return (included_minutes, used_this_cycle, remaining)

    @staticmethod
    def record_call_minutes(
        db: Session,
        tenant_id: uuid.UUID,
        call_id: uuid.UUID | None,
        minutes: Decimal,
        credits_charged: Decimal = Decimal("0"),
    ) -> UsageRecord:
        """Record a billing usage row for a call: minutes consumed + credits actually charged."""
        record = UsageRecord(
            workspace_id=tenant_id,
            call_id=call_id,
            billable_minutes=minutes,
            credits_charged=credits_charged,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def record_call_usage(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> None:
        """Record a billing event for a connected outbound call.

        Legacy stub kept for existing callers (e.g. batch_call_worker_service.py)
        that don't yet compute real minutes/credits — behaviorally identical to
        before (inserts a 0-minute record). New code should call
        `record_call_minutes` directly with real values.
        """
        BillingService.record_call_minutes(
            db=db,
            tenant_id=tenant_id,
            call_id=None,
            minutes=Decimal("0"),
            credits_charged=Decimal("0"),
        )

    @staticmethod
    def _same_instant(a: datetime | None, b: datetime | None) -> bool:
        """Compare two datetimes for practical equality, tolerant of SQLite's
        tzinfo-dropping round-trip and Stripe's integer-second timestamps."""
        if a is None or b is None:
            return False

        def _norm(dt: datetime) -> datetime:
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        return abs((_norm(a) - _norm(b)).total_seconds()) < 1

    @staticmethod
    def handle_subscription_renewed(
        db: Session,
        stripe_subscription_id: str,
        period_start: datetime,
        period_end: datetime,
        invoice_id: str | None = None,
    ) -> None:
        """Handle a Stripe `invoice.payment_succeeded` renewal: roll the period forward and
        re-grant the plan's monthly credit allotment to the tenant (core plans only).

        Idempotency (two guards, since a tenant's very first invoice for a new core-plan
        subscription fires BOTH `checkout.session.completed` — which already grants
        `monthly_credits` once via `sync_payment_status` — AND `invoice.payment_succeeded`
        for that same period):

        1. Claim on the Stripe invoice id (mirrors `_claim_checkout_session`) so a
           redelivered `invoice.payment_succeeded` webhook event (Stripe is at-least-once)
           is a hard no-op.
        2. If the incoming `period_start` matches what's already recorded on the
           subscription, this is the same period already credited by the checkout-
           completion path (or a prior call to this method) — skip granting credits again.
        """
        from app.core.logger import logger

        subscription = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == stripe_subscription_id
        ).first()
        if not subscription:
            logger.warning(
                "handle_subscription_renewed: no subscription found for stripe_subscription_id=%s",
                stripe_subscription_id,
            )
            return

        if invoice_id and not BillingService._claim_checkout_session(db, f"invoice:{invoice_id}"):
            logger.info(
                "Invoice %s already processed for subscription %s - skipping duplicate renewal.",
                invoice_id, stripe_subscription_id,
            )
            return

        already_recorded_period = BillingService._same_instant(
            subscription.current_period_start, period_start
        )

        subscription.current_period_start = period_start
        subscription.current_period_end = period_end
        subscription.updated_at = datetime.now(timezone.utc)

        if already_recorded_period:
            logger.info(
                "Renewal for subscription %s: period %s already recorded (likely granted "
                "via checkout completion) - skipping duplicate credit grant.",
                stripe_subscription_id, period_start,
            )
            db.commit()
            return

        plan = db.query(Plan).filter(Plan.id == subscription.plan_id).first()
        if plan and plan.monthly_credits and subscription.tenant_id:
            tenant = db.query(Tenant).filter(Tenant.id == subscription.tenant_id).first()
            if tenant:
                tenant.credits = (tenant.credits or Decimal("0")) + Decimal(str(plan.monthly_credits))
                logger.info(
                    "Renewed subscription %s: granted %s credits to tenant %s",
                    stripe_subscription_id, plan.monthly_credits, subscription.tenant_id,
                )

        db.commit()

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
