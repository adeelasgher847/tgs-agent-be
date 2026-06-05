"""add deleted_at to tenant for soft delete

Revision ID: 20260518_tenant_deleted_at
Revises: 20260517_apikey_idx
Create Date: 2026-05-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260518_tenant_deleted_at"
down_revision: Union[str, Sequence[str], None] = "20260517_apikey_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenant",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_tenant_deleted_at",
        "tenant",
        ["deleted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_tenant_deleted_at", table_name="tenant")
    op.drop_column("tenant", "deleted_at")
