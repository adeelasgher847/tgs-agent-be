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
            Subscription.user_id == user_id
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
    def has_active_paid_subscription(db: Session, user_id: uuid.UUID) -> bool:
        """Check if user has an active paid subscription"""
        subscription = db.query(Subscription).join(Plan).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active",
            Plan.price_monthly > 0
        ).first()
        
        return subscription is not None
