"""Add parent_workspace_id to tenant

Revision ID: da61d0d331c1
Revises: 6069015890ad
Create Date: 2026-06-12 21:17:13.158867

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = 'da61d0d331c1'
down_revision: Union[str, Sequence[str], None] = '6069015890ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(conn, table: str, column: str) -> bool:
    return any(c["name"] == column for c in inspect(conn).get_columns(table))


def upgrade() -> None:
    """Upgrade schema."""
    # Guarded: 166509c9a980_v3_sub_accounts_branding_rbac_roles.py adds the
    # same two columns on a sibling branch that merges with this one —
    # whichever runs second would otherwise hit DuplicateColumn.
    conn = op.get_bind()
    if not _has_column(conn, "tenant", "parent_workspace_id"):
        op.add_column('tenant', sa.Column('parent_workspace_id', sa.UUID(), nullable=True))
        op.create_index('idx_tenants_parent_workspace_id', 'tenant', ['parent_workspace_id'], unique=False)
        op.create_foreign_key(None, 'tenant', 'tenant', ['parent_workspace_id'], ['id'], ondelete='SET NULL')
    if not _has_column(conn, "tenant", "workspace_type"):
        op.add_column('tenant', sa.Column('workspace_type', sa.String(), server_default='standalone', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('tenant_parent_workspace_id_fkey', 'tenant', type_='foreignkey')
    op.drop_index('idx_tenants_parent_workspace_id', table_name='tenant')
    op.drop_column('tenant', 'workspace_type')
    op.drop_column('tenant', 'parent_workspace_id')
