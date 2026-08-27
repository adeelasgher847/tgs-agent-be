"""add_recording_settings_to_call_flow

Revision ID: f9a0b1c2d3e4
Revises: e8b9c0d1e2f3
Create Date: 2026-08-26 13:35:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f9a0b1c2d3e4'
down_revision: Union[str, None] = 'e8b9c0d1e2f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add recording settings columns to callflow table."""
    op.add_column(
        'callflow',
        sa.Column(
            'recording_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('true'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'public_recording_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'faster_inbound_pickup',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'stop_recording_on_transfer',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )


def downgrade() -> None:
    """Downgrade schema: drop recording settings columns from callflow table."""
    op.drop_column('callflow', 'stop_recording_on_transfer')
    op.drop_column('callflow', 'faster_inbound_pickup')
    op.drop_column('callflow', 'public_recording_enabled')
    op.drop_column('callflow', 'recording_enabled')
