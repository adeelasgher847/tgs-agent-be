"""Add knowledge_base_ids array column to callflow

Revision ID: 20260611_add_kb_ids_callflow
Revises: 20260611_rename_tables_base_class
Create Date: 2026-06-11 00:00:00.000000
"""

revision = "20260611_add_kb_ids_callflow"
down_revision = "20260611_rename_tables_base_class"
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


def upgrade() -> None:
    op.add_column(
        "callflow",
        sa.Column("knowledge_base_ids", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("callflow", "knowledge_base_ids")
