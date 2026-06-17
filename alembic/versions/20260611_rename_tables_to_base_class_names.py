"""Rename tables to match SQLAlchemy base-class auto-naming (cls.__name__.lower()).

Also merges the two diverged heads:
  - 166509c9a980  (branding_configs / pricing_configs / usage_records + role table RBAC columns)
  - 20260611_kb_pgvector  (knowledge_bases / kb_files / kb_chunks)

Renames:
  branding_configs  → brandingconfig
  pricing_configs   → pricingconfig
  usagerecord (old subscription-quota table) → DROPPED (no longer used)
  usage_records     → usagerecord
  knowledge_bases   → knowledgebase
  kb_files          → kbfile
  kb_chunks         → kbchunk
"""
from __future__ import annotations

from typing import Sequence, Union
from alembic import op

revision: str = "20260611_rename_tables_base_class"
down_revision: Union[str, Sequence[str]] = ("166509c9a980", "20260611_kb_pgvector")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("branding_configs", "brandingconfig")
    op.rename_table("pricing_configs", "pricingconfig")

    # The legacy subscription-quota table `usagerecord` (columns: subscription_id,
    # month, year, calls_used, agents_created) is no longer used. Drop it so we can
    # rename the billing-audit table `usage_records` → `usagerecord`.
    op.execute("DROP TABLE IF EXISTS usagerecord CASCADE")
    op.rename_table("usage_records", "usagerecord")

    op.rename_table("knowledge_bases", "knowledgebase")
    op.rename_table("kb_files", "kbfile")
    op.rename_table("kb_chunks", "kbchunk")

    # Rename indexes that embed the old table names so they stay consistent.
    op.execute("ALTER INDEX IF EXISTS idx_usage_records_workspace_recorded_at RENAME TO idx_usagerecord_workspace_recorded_at")


def downgrade() -> None:
    op.execute("ALTER INDEX IF EXISTS idx_usagerecord_workspace_recorded_at RENAME TO idx_usage_records_workspace_recorded_at")

    op.rename_table("kbchunk", "kb_chunks")
    op.rename_table("kbfile", "kb_files")
    op.rename_table("knowledgebase", "knowledge_bases")
    op.rename_table("usagerecord", "usage_records")
    op.rename_table("pricingconfig", "pricing_configs")
    op.rename_table("brandingconfig", "branding_configs")
