"""add_inbound_redirect_settings_to_call_flow

Revision ID: d7a8b9c0e1f2
Revises: f6a2b3c4d5e6
Create Date: 2026-08-26 11:03:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'd7a8b9c0e1f2'
down_revision: Union[str, None] = 'f6a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'callflow',
        sa.Column(
            'redirect_inbound_calls_enabled',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'redirect_forward_phone_number',
            sa.String(length=50),
            nullable=True,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'redirect_conditions',
            JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'redirect_speak_message_enabled',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'redirect_message',
            sa.Text(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'redirect_message')
    op.drop_column('callflow', 'redirect_speak_message_enabled')
    op.drop_column('callflow', 'redirect_conditions')
    op.drop_column('callflow', 'redirect_forward_phone_number')
    op.drop_column('callflow', 'redirect_inbound_calls_enabled')
