"""add_voicemail_settings_to_call_flow

Revision ID: 9a4d8c12b7f0
Revises: 7e68c9adf64e
Create Date: 2026-08-25 17:22:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9a4d8c12b7f0'
down_revision: Union[str, Sequence[str], None] = '7e68c9adf64e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('callflow', sa.Column('voicemail_detection_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('voicemail_action', sa.String(length=50), server_default='hang_up', nullable=False))
    op.add_column('callflow', sa.Column('voicemail_message', sa.Text(), nullable=True))
    op.add_column('callflow', sa.Column('voicemail_advanced_detection_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('voicemail_detection_timeout', sa.Integer(), server_default='5', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'voicemail_detection_timeout')
    op.drop_column('callflow', 'voicemail_advanced_detection_enabled')
    op.drop_column('callflow', 'voicemail_message')
    op.drop_column('callflow', 'voicemail_action')
    op.drop_column('callflow', 'voicemail_detection_enabled')
