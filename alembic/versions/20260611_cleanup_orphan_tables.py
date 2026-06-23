"""Drop orphaned old-named tables and complete the usage_records rename.

The rename migration (20260611_rename_tables_base_class) was stamped as
applied without executing, leaving every old-named table intact alongside
its new-named counterpart created by subsequent autogenerate runs.

This migration performs the actual cleanup:
  - Drops all old-named orphan tables (CASCADE)
  - Drops the legacy subscription-quota `usagerecord` table
  - Renames `usage_records` → `usagerecord` (billing-audit table)
  - Drops pre-pgvector KB tables (knowledgebasedocument, knowledgebasechunk)
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision: str = "20260611_cleanup_orphan_tables"
down_revision: str = "20260611_add_kb_ids_callflow"
branch_labels = None
depends_on = None

# Old-named tables that now have a correctly-named counterpart in the ORM.
# Safe to drop CASCADE — the new-named tables are the live ones.
_ORPHAN_TABLES = [
    "branding_configs",
    "pricing_configs",
    "knowledge_bases",
    "kb_files",
    "kb_chunks",
    "webhook_endpoints",
    "webhook_deliveries",
    # Pre-pgvector KB tables — superseded by knowledgebase / kbfile / kbchunk
    "knowledgebasedocument",
    "knowledgebasechunk",
]


def upgrade() -> None:
    for tbl in _ORPHAN_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")

    # The old subscription-quota `usagerecord` (subscription_id, month, year,
    # calls_used, agents_created) conflicts with the rename target. Drop it
    # first, then rename the billing-audit `usage_records` → `usagerecord` —
    # but only if 20260611_rename_tables_base_class's own attempt at this
    # same rename hasn't already completed it for real. That migration's
    # docstring notes it "was stamped as applied without executing" in every
    # real deployment, which is why this migration exists at all — but on a
    # fresh install (or any environment where it actually executes), doing
    # this unconditionally would drop the just-renamed `usagerecord` table
    # and have nothing left to rename into its place.
    if inspect(op.get_bind()).has_table("usage_records"):
        op.execute("DROP TABLE IF EXISTS usagerecord CASCADE")
        op.execute("ALTER TABLE usage_records RENAME TO usagerecord")


def downgrade() -> None:
    # Non-reversible: data in the old-named tables was already superseded by the
    # new-named ones before this migration ran. Downgrade is a no-op.
    pass
