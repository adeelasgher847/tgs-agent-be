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
            print("Plans already exist!")
            return
        
        # 1. Free Plan
        free = Plan(
            name="free",
            display_name="Free",
            description="Perfect for testing.",
            price_monthly=0,  # Free
        )
        
        # 2. Starter Plan (Popular)
        starter = Plan(
            name="starter", 
            display_name="Starter",
            description="Great for small businesses.",
            price_monthly=1000,  # $10.00
            is_popular=True
        )
        
        # 3. Pro Plan
        pro = Plan(
            name="pro",
            display_name="Pro", 
            description="For growing businesses.",
            price_monthly=9900,  # $99.00
        )
        
        # Save to database
        db.add(free)
        db.add(starter) 
        db.add(pro)
        db.commit()
        
        print("Created 3 simple plans:")
        print("   Free: $0/month")
        print("   Starter: $10/month (Popular)")
        print("   Pro: $99/month")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Creating simple Vapi-style plans...")
    create_simple_plans()
    print("\nDone! Your plans are ready.")
    print("\nNext steps:")
    print("   1. Set up Stripe products in your dashboard")
    print("   2. Add STRIPE_PRICE_ID_PRO to your .env file")
    print("   3. Test the billing endpoints")
