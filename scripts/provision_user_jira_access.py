#!/usr/bin/env python3
"""
Provision Jira CRM access for a user to match app billing + select-crm flow.

1) Ensures an active subscription row: user + crm_type=jira + paid plan (Plan.price_monthly > 0).
   This is what BillingService.has_crm_access() checks.

2) Optionally links Jira by creating the per-user Jira project via Jira API
   (ScheduledCallService.get_or_create_board_for_user). This requires valid
   credentials in crmconfig for crm_type=jira.

Usage (from repo root, with .env loaded):
  python scripts/provision_user_jira_access.py --email user@example.com

Pure SQL equivalent for step (1) only — replace UUIDs from your DB:
  INSERT INTO subscription (
    id, user_id, plan_id, crm_type, status,
    current_period_start, current_period_end, cancel_at_period_end, created_at
  )
  SELECT
    gen_random_uuid(),
    u.id,
    p.id,
    'jira',
    'active',
    NOW(), NOW() + INTERVAL '30 days', false, NOW()
  FROM "user" u
  CROSS JOIN plan p
  WHERE u.email = 'user@example.com'
    AND p.crm_type = 'jira' AND p.price_monthly > 0
  ON CONFLICT ON CONSTRAINT uq_user_crm_subscription
  DO UPDATE SET
    plan_id = EXCLUDED.plan_id,
    status = 'active',
    current_period_start = EXCLUDED.current_period_start,
    current_period_end = EXCLUDED.current_period_end;
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import func, text

from app.db.session import SessionLocal
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.user import User
from app.services.scheduled_call_service import ScheduledCallService

DEFAULT_PERIOD_DAYS = 30
CRM = "jira"


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument(
        "--link-jira",
        action="store_true",
        help="Call Jira API to create/link project (needs valid crmconfig credentials)",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        user = db.query(User).filter(
            func.lower(User.email) == args.email.strip().lower()
        ).first()
        if not user:
            raise SystemExit(f"User not found: {args.email}")
        if not user.current_tenant_id:
            raise SystemExit("User has no current_tenant_id set")

        plan = db.query(Plan).filter(Plan.crm_type == CRM, Plan.price_monthly > 0).first()
        if not plan:
            raise SystemExit("No paid Jira plan in DB (plan.crm_type=jira, price_monthly>0)")

        sub = db.query(Subscription).filter(
            Subscription.user_id == user.id,
            Subscription.crm_type == CRM,
        ).first()
        now = datetime.now(timezone.utc)
        period_end = now + timedelta(days=DEFAULT_PERIOD_DAYS)
        if not sub:
            sub = Subscription(
                user_id=user.id,
                plan_id=plan.id,
                crm_type=CRM,
                status="active",
                current_period_start=now,
                current_period_end=period_end,
            )
            db.add(sub)
            print("Created subscription for Jira")
        else:
            sub.plan_id = plan.id
            sub.status = "active"
            sub.current_period_start = now
            sub.current_period_end = period_end
            print("Updated subscription for Jira")
        db.commit()
        db.refresh(sub)

        row = db.execute(
            text(
                """
            SELECT 1 FROM subscription s JOIN plan p ON p.id = s.plan_id
            WHERE s.user_id = :uid AND s.status = 'active' AND s.crm_type = :crm
              AND p.price_monthly > 0
              AND (s.current_period_end IS NULL OR s.current_period_end > NOW() AT TIME ZONE 'utc')
            """
            ),
            {"uid": user.id, "crm": CRM},
        ).first()
        print("has_crm_access(jira) equivalent:", row is not None)

        if not args.link_jira:
            print("Skip --link-jira: no Jira project creation")
            return

        jira_row = db.execute(
            text("SELECT id FROM crmconfig WHERE crm_type = :c"),
            {"c": CRM},
        ).mappings().first()
        if not jira_row:
            raise SystemExit("No jira row in crmconfig")
        crm_config_id = jira_row["id"]

        existing = ScheduledCallService.get_board_for_user(db, user.id, crm_config_id)
        if existing:
            print("Already linked:", existing.crm_container_id)
            return

        board, _ = ScheduledCallService.get_or_create_board_for_user(
            db, user.id, user.current_tenant_id, crm_config_id
        )
        print("Linked Jira project:", board.crm_container_id, board.crm_container_url)
    finally:
        db.close()


if __name__ == "__main__":
    main()
