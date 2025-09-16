from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from app.models.tenant import Tenant
from app.models.subscription import Subscription
from app.models.plan import Plan
from app.models.usage_record import UsageRecord
from app.models.agent import Agent
from app.core.config import settings
from typing import Optional, Dict, Any
from datetime import datetime, date
import uuid

class BillingService:
    
    @staticmethod
    def get_or_create_subscription(db: Session, tenant_id: uuid.UUID) -> Subscription:
        """Get existing subscription or create a free one"""
        subscription = db.query(Subscription).filter(
            Subscription.tenant_id == tenant_id
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
                    price_yearly=0,
                    agent_limit=settings.FREE_PLAN_AGENT_LIMIT,
                    monthly_calls_limit=settings.FREE_PLAN_MONTHLY_CALLS,
                    is_active=True
                )
                db.add(free_plan)
                db.commit()
                db.refresh(free_plan)
            
            subscription = Subscription(
                tenant_id=tenant_id,
                plan_id=free_plan.id,
                status="active"
            )
            db.add(subscription)
            db.commit()
            db.refresh(subscription)
        
        return subscription
    
    @staticmethod
    def get_current_usage(db: Session, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Get current month usage for a tenant"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        current_date = datetime.now()
        
        # Get current month usage record
        usage_record = db.query(UsageRecord).filter(
            and_(
                UsageRecord.subscription_id == subscription.id,
                UsageRecord.month == current_date.month,
                UsageRecord.year == current_date.year
            )
        ).first()
        
        if not usage_record:
            usage_record = UsageRecord(
                subscription_id=subscription.id,
                month=current_date.month,
                year=current_date.year,
                calls_used=0,
                agents_created=0
            )
            db.add(usage_record)
            db.commit()
            db.refresh(usage_record)
        
        # Get actual agent count
        agent_count = db.query(Agent).filter(Agent.tenant_id == tenant_id).count()
        
        return {
            'subscription_id': subscription.id,
            'plan_name': subscription.plan.name,
            'plan_display_name': subscription.plan.display_name,
            'agent_limit': subscription.plan.agent_limit,
            'monthly_calls_limit': subscription.plan.monthly_calls_limit,
            'agents_used': agent_count,
            'calls_used': usage_record.calls_used,
            'agents_created_this_month': usage_record.agents_created,
            'usage_percentage': {
                'agents': (agent_count / subscription.plan.agent_limit * 100) if subscription.plan.agent_limit > 0 else 0,
                'calls': (usage_record.calls_used / subscription.plan.monthly_calls_limit * 100) if subscription.plan.monthly_calls_limit > 0 else 0
            }
        }
    
    @staticmethod
    def check_agent_limit(db: Session, tenant_id: uuid.UUID) -> bool:
        """Check if tenant can create more agents"""
        usage = BillingService.get_current_usage(db, tenant_id)
        return usage['agents_used'] < usage['agent_limit']
    
    @staticmethod
    def check_calls_limit(db: Session, tenant_id: uuid.UUID, additional_calls: int = 1) -> bool:
        """Check if tenant can make more calls"""
        usage = BillingService.get_current_usage(db, tenant_id)
        return (usage['calls_used'] + additional_calls) <= usage['monthly_calls_limit']
    
    @staticmethod
    def increment_agent_usage(db: Session, tenant_id: uuid.UUID) -> None:
        """Increment agent creation count for current month"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        current_date = datetime.now()
        
        usage_record = db.query(UsageRecord).filter(
            and_(
                UsageRecord.subscription_id == subscription.id,
                UsageRecord.month == current_date.month,
                UsageRecord.year == current_date.year
            )
        ).first()
        
        if usage_record:
            usage_record.agents_created += 1
        else:
            usage_record = UsageRecord(
                subscription_id=subscription.id,
                month=current_date.month,
                year=current_date.year,
                calls_used=0,
                agents_created=1
            )
            db.add(usage_record)
        
        db.commit()
    
    @staticmethod
    def increment_calls_usage(db: Session, tenant_id: uuid.UUID, calls_count: int = 1) -> None:
        """Increment calls usage for current month"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        current_date = datetime.now()
        
        usage_record = db.query(UsageRecord).filter(
            and_(
                UsageRecord.subscription_id == subscription.id,
                UsageRecord.month == current_date.month,
                UsageRecord.year == current_date.year
            )
        ).first()
        
        if usage_record:
            usage_record.calls_used += calls_count
        else:
            usage_record = UsageRecord(
                subscription_id=subscription.id,
                month=current_date.month,
                year=current_date.year,
                calls_used=calls_count,
                agents_created=0
            )
            db.add(usage_record)
        
        db.commit()
    
    @staticmethod
    def downgrade_to_free_plan(db: Session, tenant_id: uuid.UUID) -> None:
        """Downgrade tenant to free plan (used for payment failures)"""
        subscription = db.query(Subscription).filter(
            Subscription.tenant_id == tenant_id
        ).first()
        
        if subscription:
            free_plan = db.query(Plan).filter(Plan.name == "free").first()
            if free_plan:
                subscription.plan_id = free_plan.id
                subscription.status = "active"
                subscription.stripe_subscription_id = None
                subscription.stripe_customer_id = None
                db.commit()
    
    @staticmethod
    def enforce_limits(db: Session, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Enforce plan limits and return status"""
        usage = BillingService.get_current_usage(db, tenant_id)
        
        # Check if over limits
        over_agent_limit = usage['agents_used'] > usage['agent_limit']
        over_calls_limit = usage['calls_used'] > usage['monthly_calls_limit']
        
        return {
            'within_limits': not (over_agent_limit or over_calls_limit),
            'over_agent_limit': over_agent_limit,
            'over_calls_limit': over_calls_limit,
            'usage': usage
        }
    
    @staticmethod
    def get_tenant_subscription_status(db: Session, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Get comprehensive subscription status for a tenant"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        usage = BillingService.get_current_usage(db, tenant_id)
        
        return {
            'subscription': {
                'id': subscription.id,
                'status': subscription.status,
                'current_period_start': subscription.current_period_start,
                'current_period_end': subscription.current_period_end,
                'cancel_at_period_end': subscription.cancel_at_period_end,
                'stripe_subscription_id': subscription.stripe_subscription_id,
                'stripe_customer_id': subscription.stripe_customer_id
            },
            'plan': {
                'id': subscription.plan.id,
                'name': subscription.plan.name,
                'display_name': subscription.plan.display_name,
                'description': subscription.plan.description,
                'price_monthly': subscription.plan.price_monthly,
                'price_yearly': subscription.plan.price_yearly,
                'agent_limit': subscription.plan.agent_limit,
                'monthly_calls_limit': subscription.plan.monthly_calls_limit
            },
            'usage': usage,
            'limits_enforcement': BillingService.enforce_limits(db, tenant_id)
        }
