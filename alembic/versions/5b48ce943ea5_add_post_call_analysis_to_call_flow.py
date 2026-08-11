"""add_post_call_analysis_to_call_flow

Revision ID: 5b48ce943ea5
Revises: bf84a79ee866
Create Date: 2026-08-11 20:11:02.281510

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5b48ce943ea5'
down_revision: Union[str, Sequence[str], None] = 'bf84a79ee866'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('callflow', sa.Column('post_call_analysis_variables', postgresql.JSONB(astext_type=sa.Text()), server_default="'[]'::jsonb", nullable=False))
    op.add_column('callflow', sa.Column('post_call_analysis_model', sa.String(length=100), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'post_call_analysis_model')
    op.drop_column('callflow', 'post_call_analysis_variables')
