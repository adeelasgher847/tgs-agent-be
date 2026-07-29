"""add_status_to_callflow

Revision ID: 85f24740c31a
Revises: 20260721_prompt_soft_del
Create Date: 2026-07-28 23:09:44.798817

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '85f24740c31a'
down_revision: Union[str, Sequence[str], None] = '20260721_prompt_soft_del'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "callflow",
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="active",
        ),
    )
    op.create_check_constraint(
        "ck_callflow_status",
        "callflow",
        "status IN ('active', 'inactive')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_callflow_status", "callflow", type_="check")
    op.drop_column("callflow", "status")
