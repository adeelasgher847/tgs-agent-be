"""add_post_call_actions_to_call_flow

Revision ID: bf84a79ee866
Revises: c35f4e3d5e3f
Create Date: 2026-08-11 16:45:12.595601

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bf84a79ee866'
down_revision: Union[str, Sequence[str], None] = 'c35f4e3d5e3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('callflow', sa.Column('email_summary_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('email_summary_recipients', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('callflow', sa.Column('summary_to_business_owner_enabled', sa.Boolean(), server_default='false', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'summary_to_business_owner_enabled')
    op.drop_column('callflow', 'email_summary_recipients')
    op.drop_column('callflow', 'email_summary_enabled')
