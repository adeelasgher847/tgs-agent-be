"""add_workspaceintegration_table

Revision ID: f75178a95482
Revises: fdff731d11a7
Create Date: 2026-06-24 20:31:39.546692

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f75178a95482'
down_revision: Union[str, Sequence[str], None] = 'fdff731d11a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'workspaceintegration',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('access_token', sa.Text(), nullable=True),
        sa.Column('refresh_token', sa.Text(), nullable=True),
        sa.Column('token_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['workspace_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('workspace_id', 'provider', name='uq_workspace_integration_provider'),
    )
    op.create_index(op.f('ix_workspaceintegration_id'), 'workspaceintegration', ['id'], unique=False)
    op.create_index(op.f('ix_workspaceintegration_workspace_id'), 'workspaceintegration', ['workspace_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_workspaceintegration_workspace_id'), table_name='workspaceintegration')
    op.drop_index(op.f('ix_workspaceintegration_id'), table_name='workspaceintegration')
    op.drop_table('workspaceintegration')
