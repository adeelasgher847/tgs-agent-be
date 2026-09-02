"""Add soft-delete timestamp to tenant memberships.

Revision ID: 20260902_membership_removed_at
Revises: a2b3c4d5e6f7
Create Date: 2026-09-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision: str = "20260902_membership_removed_at"
down_revision: Union[str, Sequence[str], None] = "a2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if "removed_at" not in {
        column["name"]
        for column in inspector.get_columns("user_tenant_association")
    }:
        op.add_column(
            "user_tenant_association",
            sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "removed_at" in {
        column["name"]
        for column in inspector.get_columns("user_tenant_association")
    }:
        op.drop_column("user_tenant_association", "removed_at")