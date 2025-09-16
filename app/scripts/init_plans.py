#!/usr/bin/env python3
"""
Simple script to create Vapi-style plans.
Just run: python app/scripts/init_plans.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.plan import Plan

def create_simple_plans():
    """Create 3 simple plans like Vapi"""
    db: Session = SessionLocal()
    
    try:
        # Check if plans already exist
        if db.query(Plan).count() > 0:
            print("✅ Plans already exist!")
            return
        
        # 1. Free Plan
        free = Plan(
            name="free",
            display_name="Free",
            description="Perfect for testing. Pay $0.05 per minute.",
            price_monthly=0,  # Free
            price_per_minute=0.05,  # $0.05 per minute
            agent_limit=2,
            monthly_calls_limit=0,  # No limit, pay per minute
            included_minutes=0
        )
        
        # 2. Starter Plan (Popular)
        starter = Plan(
            name="starter", 
            display_name="Starter",
            description="Great for small businesses. $10/month + $0.05/minute.",
            price_monthly=10,  # $10
            price_per_minute=0.05,
            agent_limit=1,
            monthly_calls_limit=0,  # No limit, pay per minute
            included_minutes=500,  # 500 free minutes
            is_popular=True
        )
        
        # 3. Pro Plan
        pro = Plan(
            name="pro",
            display_name="Pro", 
            description="For growing businesses. $99/month + $0.05/minute.",
            price_monthly=2,  # $99
            price_per_minute=0.05,
            agent_limit=50,
            monthly_calls_limit=0,  # No limit, pay per minute
            included_minutes=2000  # 2000 free minutes
        )
        
        # Save to database
        db.add(free)
        db.add(starter) 
        db.add(pro)
        db.commit()
        
        print("✅ Created 3 simple plans:")
        print("   📱 Free: $0/month, $0.05/minute, 2 agents")
        print("   🚀 Starter: $10/month, 500 free minutes, 10 agents (Popular)")
        print("   💼 Pro: $99/month, 2000 free minutes, 50 agents")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Creating simple Vapi-style plans...")
    create_simple_plans()
    print("\n✅ Done! Your plans are ready.")
    print("\n💡 Next steps:")
    print("   1. Set up Stripe products in your dashboard")
    print("   2. Add STRIPE_PRICE_ID_PRO to your .env file")
    print("   3. Test the billing endpoints")
