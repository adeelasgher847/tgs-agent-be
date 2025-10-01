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
                'agent_limit': subscription.plan.agent_limit,
                'monthly_calls_limit': subscription.plan.monthly_calls_limit
            },
            'usage': usage,
            'limits_enforcement': BillingService.enforce_limits(db, tenant_id)
        }
    
    @staticmethod
    def get_usage_history(db: Session, tenant_id: uuid.UUID, months: int = 12) -> List[Dict[str, Any]]:
        """Get usage history for the past N months"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        current_date = datetime.now()
        
        usage_history = []
        for i in range(months):
            target_date = current_date - timedelta(days=30 * i)
            month = target_date.month
            year = target_date.year
            
            usage_record = db.query(UsageRecord).filter(
                and_(
                    UsageRecord.subscription_id == subscription.id,
                    UsageRecord.month == month,
                    UsageRecord.year == year
                )
            ).first()
            
            if usage_record:
                usage_history.append({
                    'month': month,
                    'year': year,
                    'calls_used': usage_record.calls_used,
                    'agents_created': usage_record.agents_created,
                    'created_at': usage_record.created_at
                })
            else:
                usage_history.append({
                    'month': month,
                    'year': year,
                    'calls_used': 0,
                    'agents_created': 0,
                    'created_at': None
                })
        
        return usage_history
    
    @staticmethod
    def get_usage_analytics(db: Session, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Get usage analytics and trends"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        current_usage = BillingService.get_current_usage(db, tenant_id)
        usage_history = BillingService.get_usage_history(db, tenant_id, 6)
        
        # Calculate trends
        if len(usage_history) >= 2:
            current_month = usage_history[0]
            previous_month = usage_history[1]
            
            calls_trend = ((current_month['calls_used'] - previous_month['calls_used']) / 
                          max(previous_month['calls_used'], 1)) * 100
            agents_trend = ((current_month['agents_created'] - previous_month['agents_created']) / 
                           max(previous_month['agents_created'], 1)) * 100
        else:
            calls_trend = 0
            agents_trend = 0
        
        # Calculate average usage
        total_calls = sum(record['calls_used'] for record in usage_history)
        total_agents = sum(record['agents_created'] for record in usage_history)
        avg_calls = total_calls / len(usage_history) if usage_history else 0
        avg_agents = total_agents / len(usage_history) if usage_history else 0
        
        return {
            'current_usage': current_usage,
            'usage_history': usage_history,
            'trends': {
                'calls_trend_percentage': calls_trend,
                'agents_trend_percentage': agents_trend
            },
            'averages': {
                'monthly_calls': avg_calls,
                'monthly_agents': avg_agents
            },
            'projections': {
                'estimated_monthly_calls': avg_calls,
                'estimated_monthly_agents': avg_agents
            }
        }
    
    @staticmethod
    def sync_usage_with_stripe(db: Session, tenant_id: uuid.UUID) -> None:
        """Sync usage data with Stripe for metered billing"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        
        if not subscription.stripe_subscription_id:
            return
        
        try:
            # Get current usage
            current_usage = BillingService.get_current_usage(db, tenant_id)
            
            # Get Stripe subscription to find subscription items
            stripe_subscription = StripeService.get_subscription(subscription.stripe_subscription_id)
            
            # Update usage for each subscription item
            for item in stripe_subscription['items']['data']:
                # Assuming we have a metered price for calls
                if 'calls' in item['price']['nickname'].lower():
                    StripeService.create_usage_record(
                        item['id'],
                        current_usage['calls_used']
                    )
                
                # Assuming we have a metered price for agents
                elif 'agents' in item['price']['nickname'].lower():
                    StripeService.create_usage_record(
                        item['id'],
                        current_usage['agents_used']
                    )
        
        except Exception as e:
            print(f"Error syncing usage with Stripe: {str(e)}")
    
    @staticmethod
    def check_and_enforce_limits(db: Session, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Check limits and return enforcement status"""
        limits = BillingService.enforce_limits(db, tenant_id)
        
        if not limits['within_limits']:
            # Log the violation
            print(f"Tenant {tenant_id} exceeded limits: {limits}")
            
            # Optionally downgrade to free plan if over limits
            if limits['over_calls_limit'] and limits['usage']['plan_name'] != 'free':
                print(f"Downgrading tenant {tenant_id} to free plan due to overage")
                BillingService.downgrade_to_free_plan(db, tenant_id)
        
        return limits
    
    @staticmethod
    def get_billing_summary(db: Session, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Get comprehensive billing summary"""
        subscription = BillingService.get_or_create_subscription(db, tenant_id)
        usage_analytics = BillingService.get_usage_analytics(db, tenant_id)
        limits = BillingService.enforce_limits(db, tenant_id)
        
        # Get upcoming invoice if customer exists
        upcoming_invoice = None
        if subscription.stripe_customer_id:
            try:
                upcoming_invoice = StripeService.get_upcoming_invoice(subscription.stripe_customer_id)
            except:
                pass
        
        return {
            'subscription': {
                'id': subscription.id,
                'status': subscription.status,
                'plan_name': subscription.plan.name,
                'plan_display_name': subscription.plan.display_name,
                'current_period_end': subscription.current_period_end,
                'cancel_at_period_end': subscription.cancel_at_period_end
            },
            'usage': usage_analytics['current_usage'],
            'analytics': usage_analytics,
            'limits': limits,
            'upcoming_invoice': upcoming_invoice,
            'billing_status': {
                'is_active': subscription.status == 'active',
                'is_past_due': subscription.status == 'past_due',
                'is_canceled': subscription.status == 'canceled',
                'has_stripe_customer': bool(subscription.stripe_customer_id)
            }
        }
