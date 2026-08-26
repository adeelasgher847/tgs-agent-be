"""add_disable_metadata_to_call_flow

Revision ID: c4f92d831e05
Revises: b3e81a942cd1
Create Date: 2026-08-26 08:26:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4f92d831e05'
down_revision: Union[str, Sequence[str], None] = 'b3e81a942cd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'callflow',
        sa.Column(
            'disable_metadata',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'disable_metadata')
