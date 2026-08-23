#!/usr/bin/env python3
"""
Seed/upsert the core-product (non-CRM) Studio and Agency Plan rows.

Run: python app/scripts/init_plans.py

These are tenant-level subscription plans (see `app/models/plan.py`'s
Studio/Agency fields: included_minutes, monthly_credits, max_subaccounts,
free_phone_numbers, features). `crm_type` is left NULL for both — they are
NOT one of the three CRM-addon plan stacks.

Set STRIPE_PRICE_ID_STUDIO / STRIPE_PRICE_ID_AGENCY in the environment before
running so the seeded rows have a working `stripe_price_id` for checkout.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
import app.db.base  # noqa: F401 - registers all models so relationship() strings resolve
from app.models.plan import Plan

CORE_PLAN_FEATURES = [
    "WhiteLabel workspaces",
    "Advanced analytics",
    "Priority support",
]


def _upsert_plan(db: Session, *, name: str, display_name: str, description: str,
                  price_monthly_cents: int, included_minutes: int, monthly_credits: str,
                  max_subaccounts: int | None, free_phone_numbers: int, features: list[str],
                  stripe_price_id_env: str, is_popular: bool = False) -> Plan:
    plan = db.query(Plan).filter(Plan.name == name).first()
    if plan is None:
        plan = Plan(name=name)
        db.add(plan)

    plan.display_name = display_name
    plan.description = description
    plan.price_monthly = price_monthly_cents
    plan.crm_type = None
    plan.is_active = True
    plan.is_popular = is_popular
    plan.included_minutes = included_minutes
    plan.monthly_credits = monthly_credits
    plan.max_subaccounts = max_subaccounts
    plan.free_phone_numbers = free_phone_numbers
    plan.features = features
    plan.stripe_price_id = os.environ.get(stripe_price_id_env) or plan.stripe_price_id

    return plan


def seed_core_plans() -> None:
    """Create/update the Studio and Agency core-product Plan rows."""
    db: Session = SessionLocal()

    try:
        studio = _upsert_plan(
            db,
            name="studio",
            display_name="Studio",
            description="100 included call minutes/mo, whitelabel workspaces, up to 3 sub-accounts.",
            price_monthly_cents=9900,
            included_minutes=100,
            monthly_credits="12.00",
            max_subaccounts=3,
            free_phone_numbers=3,
            features=CORE_PLAN_FEATURES,
            stripe_price_id_env="STRIPE_PRICE_ID_STUDIO",
        )

        agency = _upsert_plan(
            db,
            name="agency",
            display_name="Agency",
            description="300 included call minutes/mo, unlimited whitelabel sub-accounts.",
            price_monthly_cents=29900,
            included_minutes=300,
            monthly_credits="36.00",
            max_subaccounts=None,
            free_phone_numbers=10,
            features=CORE_PLAN_FEATURES + ["Unlimited sub-accounts"],
            stripe_price_id_env="STRIPE_PRICE_ID_AGENCY",
            is_popular=True,
        )

        db.commit()

        print("Seeded core plans:")
        print(f"   Studio: $99/month, {studio.included_minutes} included minutes, "
              f"${studio.monthly_credits} credit grant, {studio.max_subaccounts} max sub-accounts")
        print(f"   Agency: $299/month, {agency.included_minutes} included minutes, "
              f"${agency.monthly_credits} credit grant, unlimited sub-accounts")

    except Exception as e:
        print(f"Error seeding core plans: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding Studio/Agency core-product plans...")
    seed_core_plans()
    print("\nDone.")
    print("\nNext steps:")
    print("   1. Set STRIPE_PRICE_ID_STUDIO / STRIPE_PRICE_ID_AGENCY in your .env if not already set")
    print("   2. Re-run this script if you update pricing/features so the DB rows stay in sync")
