"""Add parent_workspace_id to tenant

Revision ID: da61d0d331c1
Revises: 6069015890ad
Create Date: 2026-06-12 21:17:13.158867

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'da61d0d331c1'
down_revision: Union[str, Sequence[str], None] = '6069015890ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tenant', sa.Column('parent_workspace_id', sa.UUID(), nullable=True))
    op.add_column('tenant', sa.Column('workspace_type', sa.String(), server_default='standalone', nullable=False))
    op.create_index('idx_tenants_parent_workspace_id', 'tenant', ['parent_workspace_id'], unique=False)
    op.create_foreign_key(None, 'tenant', 'tenant', ['parent_workspace_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('tenant_parent_workspace_id_fkey', 'tenant', type_='foreignkey')
    op.drop_index('idx_tenants_parent_workspace_id', table_name='tenant')
    op.drop_column('tenant', 'workspace_type')
    op.drop_column('tenant', 'parent_workspace_id')
