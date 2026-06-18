"""upgrade_auditlog_enterprise

Upgrades the HIPAA-only auditlog table to a full enterprise audit log:
  - Adds actor_api_key_prefix column
  - Changes old_value / new_value from TEXT to JSONB
  - Changes ip_address from VARCHAR(45) to INET
  - Adds DB-level no_update RULE (silently drops all UPDATEs)
  - Adds a no_delete trigger that blocks DELETEs from application code;
    the retention ARQ job bypasses it by setting the session GUC
    app.bypass_audit_delete = 'true' before its DELETE statement.

Revision ID: 20260617_enterprise_audit
Revises: 5b152b97d6b5
Create Date: 2026-06-17

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import INET, JSONB

revision: str = "20260617_enterprise_audit"
down_revision: Union[str, Sequence[str], None] = "5b152b97d6b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Column additions / type changes ──────────────────────────────────────
    op.add_column(
        "auditlog",
        sa.Column("actor_api_key_prefix", sa.String(8), nullable=True),
    )
    op.alter_column(
        "auditlog",
        "old_value",
        existing_type=sa.Text(),
        type_=JSONB(),
        existing_nullable=True,
        postgresql_using="old_value::jsonb",
    )
    op.alter_column(
        "auditlog",
        "new_value",
        existing_type=sa.Text(),
        type_=JSONB(),
        existing_nullable=True,
        postgresql_using="new_value::jsonb",
    )
    op.alter_column(
        "auditlog",
        "ip_address",
        existing_type=sa.String(45),
        type_=INET(),
        existing_nullable=True,
        postgresql_using="ip_address::inet",
    )
    # Change user_agent from VARCHAR(512) to TEXT
    op.alter_column(
        "auditlog",
        "user_agent",
        existing_type=sa.String(512),
        type_=sa.Text(),
        existing_nullable=True,
    )

    # ── Append-only enforcement ───────────────────────────────────────────────
    # Block all UPDATEs from every role via a PostgreSQL RULE.
    op.execute(
        "CREATE RULE no_update_audit AS ON UPDATE TO auditlog DO INSTEAD NOTHING"
    )

    # Block DELETEs via a trigger function that checks a session GUC.
    # The 90-day retention job sets  SET LOCAL app.bypass_audit_delete = 'true'
    # inside its transaction to bypass this check.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION _prevent_auditlog_delete()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF current_setting('app.bypass_audit_delete', true) = 'true' THEN
                RETURN OLD;
            END IF;
            RETURN NULL;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER no_delete_audit
        BEFORE DELETE ON auditlog
        FOR EACH ROW EXECUTE FUNCTION _prevent_auditlog_delete()
        """
    )

    # ── GIN index on JSONB columns for fast filtering ─────────────────────────
    op.execute(
        "CREATE INDEX ix_auditlog_new_value ON auditlog USING GIN (new_value)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_auditlog_new_value")
    op.execute("DROP TRIGGER IF EXISTS no_delete_audit ON auditlog")
    op.execute("DROP FUNCTION IF EXISTS _prevent_auditlog_delete()")
    op.execute("DROP RULE IF EXISTS no_update_audit ON auditlog")

    op.alter_column(
        "auditlog",
        "user_agent",
        existing_type=sa.Text(),
        type_=sa.String(512),
        existing_nullable=True,
    )
    op.alter_column(
        "auditlog",
        "ip_address",
        existing_type=INET(),
        type_=sa.String(45),
        existing_nullable=True,
        postgresql_using="ip_address::text",
    )
    op.alter_column(
        "auditlog",
        "new_value",
        existing_type=JSONB(),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using="new_value::text",
    )
    op.alter_column(
        "auditlog",
        "old_value",
        existing_type=JSONB(),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using="old_value::text",
    )
    op.drop_column("auditlog", "actor_api_key_prefix")
