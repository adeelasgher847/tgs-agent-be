"""add_ivr_and_dtmf_settings_to_call_flow

Revision ID: e5f12a73b901
Revises: c4f92d831e05
Create Date: 2026-08-26 09:28:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e5f12a73b901'
down_revision: Union[str, Sequence[str], None] = 'c4f92d831e05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_enabled',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_action',
            sa.String(length=50),
            server_default='dial_through',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_navigation_mode',
            sa.String(length=50),
            server_default='let_ai_converse',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_max_attempts',
            sa.Integer(),
            server_default='3',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_keypress_delay',
            sa.Integer(),
            server_default='8',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_priority_list',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_wait_on_hold',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'ivr_max_hold_time',
            sa.Integer(),
            server_default='120',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_enabled',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_button_press_delay',
            sa.Integer(),
            server_default='2',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_allow_caller_interruption',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_max_digits',
            sa.Integer(),
            server_default='50',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_allowed_exceeded_attempts',
            sa.Integer(),
            server_default='10',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_exceeded_action',
            sa.String(length=50),
            server_default='end_call',
            nullable=False,
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'dtmf_end_call_message',
            sa.Text(),
            nullable=True,
            server_default="You've reached the maximum number of inputs allowed for this call.",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'dtmf_end_call_message')
    op.drop_column('callflow', 'dtmf_exceeded_action')
    op.drop_column('callflow', 'dtmf_allowed_exceeded_attempts')
    op.drop_column('callflow', 'dtmf_max_digits')
    op.drop_column('callflow', 'dtmf_allow_caller_interruption')
    op.drop_column('callflow', 'dtmf_button_press_delay')
    op.drop_column('callflow', 'dtmf_enabled')
    op.drop_column('callflow', 'ivr_max_hold_time')
    op.drop_column('callflow', 'ivr_wait_on_hold')
    op.drop_column('callflow', 'ivr_priority_list')
    op.drop_column('callflow', 'ivr_keypress_delay')
    op.drop_column('callflow', 'ivr_max_attempts')
    op.drop_column('callflow', 'ivr_navigation_mode')
    op.drop_column('callflow', 'ivr_action')
    op.drop_column('callflow', 'ivr_enabled')
