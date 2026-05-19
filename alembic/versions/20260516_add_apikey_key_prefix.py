"""add_apikey_key_prefix

Revision ID: 20260516_key_prefix
Revises: 01a225a67ca1
Create Date: 2026-05-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260516_key_prefix"
down_revision: Union[str, Sequence[str], None] = "01a225a67ca1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "apikey",
        sa.Column("key_prefix", sa.String(length=32), nullable=False, server_default="••••••••"),
    )
    op.alter_column("apikey", "key_prefix", server_default=None)


def downgrade() -> None:
    op.drop_column("apikey", "key_prefix")
