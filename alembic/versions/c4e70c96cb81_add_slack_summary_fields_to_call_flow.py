"""add_slack_summary_fields_to_call_flow

Revision ID: c4e70c96cb81
Revises: 20260817_widen_llm_check_oair
Create Date: 2026-08-20 10:50:45.827422

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c4e70c96cb81'
down_revision: Union[str, Sequence[str], None] = '20260817_widen_llm_check_oair'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('callflow', sa.Column('slack_summary_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('slack_channel_id', sa.String(length=50), nullable=True))
    op.add_column('callflow', sa.Column('slack_channel_name', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'slack_channel_name')
    op.drop_column('callflow', 'slack_channel_id')
    op.drop_column('callflow', 'slack_summary_enabled')
