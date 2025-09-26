"""
Simple Credit Service Module
Handles basic credit operations for tenants
"""

from sqlalchemy.orm import Session
from typing import Optional
import uuid

from app.models.tenant import Tenant
from app.models.plan import Plan
from app.schemas.credit import CreditBalanceResponse

class CreditService:
    """Simple service class for handling credit operations"""
    
    def __init__(self):
        pass
    
    def get_credit_balance(self, db: Session, tenant_id: uuid.UUID) -> CreditBalanceResponse:
        """Get current credit balance for a tenant"""
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError("Tenant not found")
        
        # Get plan credits if tenant has a subscription
        plan_credits = 0
        plan_name = "No Plan"
        
        if tenant.subscription and tenant.subscription.plan:
            plan_credits = tenant.subscription.plan.credits or 0  # Handle None case
            plan_name = tenant.subscription.plan.display_name
        
        return CreditBalanceResponse(
            tenant_id=tenant.id,
            credit_balance=tenant.credit_balance,
            plan_credits=plan_credits,
            plan_name=plan_name
        )
    
    def add_credits(self, db: Session, tenant_id: uuid.UUID, amount: int, description: str = "Credit purchase") -> bool:
        """Add credits to a tenant's balance"""
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError(f"Tenant not found with ID: {tenant_id}")
        
        old_balance = tenant.credit_balance
        new_balance = old_balance + amount
        
        print(f"🔄 Adding credits for tenant {tenant_id}:")
        print(f"   Description: {description}")
        print(f"   Amount to add: {amount}")
        print(f"   Old balance: {old_balance}")
        print(f"   New balance: {new_balance}")
        
        # Add credits to balance
        tenant.credit_balance = new_balance
        
        db.commit()
        print(f"✅ Successfully added {amount} credits to tenant {tenant_id}")
        return True
    
    def use_credits(self, db: Session, tenant_id: uuid.UUID, amount: int, description: str = "Credit usage") -> bool:
        """Use credits from a tenant's balance"""
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError("Tenant not found")
        
        # Check if tenant has enough credits
        if tenant.credit_balance < amount:
            raise ValueError(f"Insufficient credits. Current balance: {tenant.credit_balance}, Required: {amount}")
        
        # Deduct credits from balance
        tenant.credit_balance -= amount
        
        db.commit()
        return True
    
    def calculate_call_cost(self, duration_seconds: int, price_per_minute: float = 0.05) -> int:
        """Calculate credit cost for a call based on duration (1 credit = $1)"""
        duration_minutes = duration_seconds / 60.0
        cost_dollars = duration_minutes * price_per_minute
        # Convert to credits (1 credit = $1) and round up
        return max(1, int(cost_dollars + 0.5))
    
    def get_plan_pricing(self, db: Session, tenant_id: uuid.UUID) -> dict:
        """Get current plan pricing for a tenant"""
        from app.models.tenant import Tenant
        
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError("Tenant not found")
        
        if tenant.subscription and tenant.subscription.plan:
            plan = tenant.subscription.plan
            return {
                "price_per_minute": plan.price_per_minute,
                "plan_name": plan.display_name,
                "plan_id": str(plan.id),
                "has_plan": True
            }
        else:
            return {
                "price_per_minute": 0.05,  # Default fallback
                "plan_name": "No Plan",
                "plan_id": None,
                "has_plan": False
            }
    
    def get_detailed_credit_info(self, db: Session, tenant_id: uuid.UUID) -> dict:
        """Get detailed credit information for debugging"""
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError(f"Tenant not found with ID: {tenant_id}")
        
        info = {
            "tenant_id": str(tenant_id),
            "tenant_name": tenant.name,
            "current_credit_balance": tenant.credit_balance,
            "stripe_customer_id": tenant.stripe_customer_id,
            "stripe_subscription_id": tenant.stripe_subscription_id,
            "tenant_status": tenant.status,
            "subscription": None,
            "plan": None
        }
        
        if tenant.subscription:
            info["subscription"] = {
                "id": str(tenant.subscription.id),
                "status": tenant.subscription.status,
                "stripe_subscription_id": tenant.subscription.stripe_subscription_id,
                "stripe_customer_id": tenant.subscription.stripe_customer_id
            }
            
            if tenant.subscription.plan:
                info["plan"] = {
                    "id": str(tenant.subscription.plan.id),
                    "name": tenant.subscription.plan.name,
                    "display_name": tenant.subscription.plan.display_name,
                    "credits": tenant.subscription.plan.credits,
                    "price_monthly": tenant.subscription.plan.price_monthly,
                    "price_per_minute": tenant.subscription.plan.price_per_minute,
                    "stripe_price_id": tenant.subscription.plan.stripe_price_id
                }
        
        return info
    
    def initialize_tenant_credits(self, db: Session, tenant_id: uuid.UUID) -> bool:
        """Initialize tenant with credits from their plan"""
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError(f"Tenant not found with ID: {tenant_id}")
        
        # If tenant has a subscription with a plan, give them the plan's credits
        if tenant.subscription and tenant.subscription.plan:
            plan_credits = tenant.subscription.plan.credits or 0  # Handle None case
            plan_name = tenant.subscription.plan.display_name
            
            print(f"🔄 Initializing credits for tenant {tenant_id}:")
            print(f"   Plan: {plan_name}")
            print(f"   Plan credits: {plan_credits}")
            print(f"   Current balance: {tenant.credit_balance}")
            
            if plan_credits > 0:
                tenant.credit_balance = plan_credits
                db.commit()
                print(f"✅ Set credit balance to {plan_credits} for tenant {tenant_id}")
                return True
            else:
                print(f"ℹ️ Plan has 0 credits, keeping current balance: {tenant.credit_balance}")
                return False
        else:
            print(f"⚠️ Tenant {tenant_id} has no active subscription or plan")
            return False

# Create service instance
credit_service = CreditService()
