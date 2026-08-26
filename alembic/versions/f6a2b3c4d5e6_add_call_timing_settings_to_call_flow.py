"""add_call_timing_settings_to_call_flow

Revision ID: f6a2b3c4d5e6
Revises: e5f12a73b901
Create Date: 2026-08-26 10:17:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'f6a2b3c4d5e6'
down_revision: Union[str, None] = 'e5f12a73b901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'callflow',
        sa.Column(
            'silence_timeout',
            sa.Integer(),
            server_default='10',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'end_call_after_reminder',
            sa.Integer(),
            server_default='10',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'reminder_retries',
            sa.Integer(),
            server_default='1',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'reminder_messages',
            JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'max_call_duration',
            sa.Integer(),
            server_default='1800',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'max_duration_message',
            sa.Text(),
            nullable=True,
            server_default="I appreciate the conversation, but we've reached our time limit for this call.",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'max_duration_message')
    op.drop_column('callflow', 'max_call_duration')
    op.drop_column('callflow', 'reminder_messages')
    op.drop_column('callflow', 'reminder_retries')
    op.drop_column('callflow', 'end_call_after_reminder')
    op.drop_column('callflow', 'silence_timeout')
