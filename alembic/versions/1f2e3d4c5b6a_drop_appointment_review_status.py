"""drop appointment review_status column

Revision ID: 1f2e3d4c5b6a
Revises: 9a1b2c3d4e5f
Create Date: 2026-04-10 21:25:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1f2e3d4c5b6a"
down_revision: Union[str, Sequence[str], None] = "9a1b2c3d4e5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("appointment", "review_status")


def downgrade() -> None:
    op.add_column(
        "appointment",
        sa.Column("review_status", sa.String(length=20), nullable=False, server_default="not_reviewed"),
    )
