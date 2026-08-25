"""add_call_screening_action_to_call_flow

Revision ID: b3e81a942cd1
Revises: 9a4d8c12b7f0
Create Date: 2026-08-25 18:03:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3e81a942cd1'
down_revision: Union[str, Sequence[str], None] = '9a4d8c12b7f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'callflow',
        sa.Column(
            'call_screening_action',
            sa.String(length=50),
            server_default='respond',
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'call_screening_action')
