"""add_wallet_auto_recharge_and_wallet_sharing

Revision ID: 7ec8c07644d0
Revises: 3ad4a023f421
Create Date: 2026-08-23 10:52:14.369656

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '7ec8c07644d0'
down_revision: Union[str, Sequence[str], None] = '3ad4a023f421'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'walletautorechargeconfig',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('min_balance', sa.Numeric(precision=10, scale=4), server_default='10.00', nullable=False),
        sa.Column('recharge_amount', sa.Numeric(precision=10, scale=4), server_default='20.00', nullable=False),
        sa.Column('stripe_payment_method_id', sa.String(), nullable=True),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['workspace_id'], ['tenant.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('workspace_id'),
    )
    op.add_column('tenant', sa.Column('uses_master_wallet', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('tenant', sa.Column('auto_link_new_workspaces', sa.Boolean(), server_default='false', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('tenant', 'auto_link_new_workspaces')
    op.drop_column('tenant', 'uses_master_wallet')
    op.drop_table('walletautorechargeconfig')
