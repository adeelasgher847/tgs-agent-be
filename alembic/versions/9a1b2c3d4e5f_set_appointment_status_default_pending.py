"""set appointment status default pending

Revision ID: 9a1b2c3d4e5f
Revises: 7c9a1d2e3f40
Create Date: 2026-04-10 21:05:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "9a1b2c3d4e5f"
down_revision: Union[str, Sequence[str], None] = "7c9a1d2e3f40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE appointment
        ALTER COLUMN status SET DEFAULT 'pending';
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE appointment
        ALTER COLUMN status SET DEFAULT 'confirmed';
        """
    )
