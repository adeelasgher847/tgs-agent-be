"""add_llm_token_budget_daily_to_tenant

Revision ID: 86f5457724a9
Revises: a2b3c4d5e6f7
Create Date: 2026-09-02 10:37:58.789390

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '86f5457724a9'
down_revision: Union[str, Sequence[str], None] = 'a2b3c4d5e6f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tenant', sa.Column('llm_token_budget_daily', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('tenant', 'llm_token_budget_daily')
