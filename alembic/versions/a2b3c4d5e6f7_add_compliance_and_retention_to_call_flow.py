"""add_compliance_and_retention_to_call_flow

Revision ID: a2b3c4d5e6f7
Revises: f9a0b1c2d3e4
Create Date: 2026-08-26 14:15:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2b3c4d5e6f7'
down_revision: Union[str, None] = 'f9a0b1c2d3e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add compliance detection and data retention columns to callflow table."""
    # Compliance & Detection
    op.add_column(
        'callflow',
        sa.Column(
            'compliance_monitoring_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'anti_bot_detection_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'terminate_on_fake_voice',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )

    # Data Retention Policy
    op.add_column(
        'callflow',
        sa.Column(
            'retention_policy_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'retention_transcript_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'retention_transcript_days',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('30'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'retention_summary_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'retention_summary_days',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('30'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'retention_recording_enabled',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )
    op.add_column(
        'callflow',
        sa.Column(
            'retention_recording_days',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('30'),
        ),
    )


def downgrade() -> None:
    """Downgrade schema: drop compliance detection and data retention columns from callflow table."""
    op.drop_column('callflow', 'retention_recording_days')
    op.drop_column('callflow', 'retention_recording_enabled')
    op.drop_column('callflow', 'retention_summary_days')
    op.drop_column('callflow', 'retention_summary_enabled')
    op.drop_column('callflow', 'retention_transcript_days')
    op.drop_column('callflow', 'retention_transcript_enabled')
    op.drop_column('callflow', 'retention_policy_enabled')
    op.drop_column('callflow', 'terminate_on_fake_voice')
    op.drop_column('callflow', 'anti_bot_detection_enabled')
    op.drop_column('callflow', 'compliance_monitoring_enabled')
