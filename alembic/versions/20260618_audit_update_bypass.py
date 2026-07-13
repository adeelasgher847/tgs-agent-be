"""audit_update_bypass

GDPR right-to-erasure requires anonymizing auditlog.user_id and
actor_api_key_prefix when a workspace account is deleted, but the
no_update_audit RULE added in 20260617_enterprise_audit unconditionally
drops every UPDATE — including that one.

Replaces the blanket RULE with a BEFORE UPDATE trigger that:
  - By default, still silently no-ops every UPDATE (same append-only
    behaviour as before).
  - When the session sets  SET LOCAL app.bypass_audit_update = 'true' ,
    forces the row's user_id to NULL and actor_api_key_prefix to
    '[DELETED]' — and *only* those two columns, regardless of what the
    calling UPDATE statement's SET clause actually requested. This caps
    the blast radius of the bypass to the one GDPR anonymization use case;
    no other column (action, old_value, new_value, timestamp, ...) can
    ever be altered through this path, even if the caller's SQL is wrong.

The account-deletion service (app/services/account_deletion_service.py)
is the only caller expected to set this GUC.

Revision ID: 20260618_audit_update_bypass
Revises: 20260617_enterprise_audit
Create Date: 2026-06-18

"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260618_audit_update_bypass"
down_revision: Union[str, Sequence[str], None] = "20260617_enterprise_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP RULE IF EXISTS no_update_audit ON auditlog")

    op.execute(
        """
        CREATE OR REPLACE FUNCTION _restrict_auditlog_update()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF current_setting('app.bypass_audit_update', true) = 'true' THEN
                NEW := OLD;
                NEW.user_id := NULL;
                NEW.actor_api_key_prefix := '[DELETED]';
                RETURN NEW;
            END IF;
            RETURN NULL;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER no_update_audit
        BEFORE UPDATE ON auditlog
        FOR EACH ROW EXECUTE FUNCTION _restrict_auditlog_update()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS no_update_audit ON auditlog")
    op.execute("DROP FUNCTION IF EXISTS _restrict_auditlog_update()")
    op.execute(
        "CREATE RULE no_update_audit AS ON UPDATE TO auditlog DO INSTEAD NOTHING"
    )
