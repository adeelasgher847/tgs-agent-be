"""v3_sub_accounts_branding_rbac_roles

Revision ID: 166509c9a980
Revises: 20260602_schema_v2
Create Date: 2026-06-09 16:56:46.534534

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '166509c9a980'
down_revision: Union[str, Sequence[str], None] = '20260602_schema_v2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Modify the existing 'tenant' table (backward-compatible fields)
    op.add_column('tenant', sa.Column('parent_workspace_id', sa.UUID(), sa.ForeignKey('tenant.id', ondelete='SET NULL'), nullable=True))
    op.add_column('tenant', sa.Column('workspace_type', sa.String(), server_default='standalone', nullable=False))
    
    op.create_check_constraint(
        'chk_tenant_workspace_type',
        'tenant',
        "workspace_type IN ('agency', 'sub_account', 'standalone')"
    )
    op.create_index('idx_tenant_parent_workspace_id', 'tenant', ['parent_workspace_id'])
    op.create_index(
        'idx_tenant_sub_accounts',
        'tenant',
        ['parent_workspace_id'],
        postgresql_where=sa.text("workspace_type = 'sub_account'")
    )

    # 2. Create branding_configs table
    op.create_table(
        'branding_configs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), sa.ForeignKey('tenant.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('logo_url', sa.String(), nullable=True),
        sa.Column('primary_colour', sa.String(length=7), nullable=True),
        sa.Column('display_name', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. Create pricing_configs table
    op.create_table(
        'pricing_configs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), sa.ForeignKey('tenant.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('per_minute_rate', sa.Numeric(precision=10, scale=4), server_default='0.12', nullable=False),
        sa.Column('markup_percent', sa.Numeric(precision=5, scale=2), server_default='0.00', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # 4. Create rbac_roles table
    op.create_table(
        'rbac_roles',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), sa.ForeignKey('tenant.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('user.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_check_constraint(
        'chk_rbac_roles_role',
        'rbac_roles',
        "role IN ('admin', 'manager', 'config_only', 'read_only', 'billing_only')"
    )
    op.create_index(
        'uq_rbac_roles_workspace_user_active', 
        'rbac_roles', 
        ['workspace_id', 'user_id'], 
        unique=True, 
        postgresql_where=sa.text("deleted_at IS NULL")
    )

    # 5. Create usage_records table (Distinct from the old 'usagerecord')
    op.create_table(
        'usage_records',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('workspace_id', sa.UUID(), sa.ForeignKey('tenant.id', ondelete='CASCADE'), nullable=False),
        sa.Column('call_id', sa.UUID(), sa.ForeignKey('callsession.id', ondelete='SET NULL'), nullable=True),
        sa.Column('billable_minutes', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_usage_records_workspace_recorded_at', 'usage_records', ['workspace_id', 'recorded_at'])


def downgrade() -> None:
    # Use PostgreSQL conditional drop execution to bypass missing tables safely
    op.execute("DROP TABLE IF EXISTS usage_records CASCADE;")
    op.execute("DROP TABLE IF EXISTS rbac_roles CASCADE;")
    op.execute("DROP TABLE IF EXISTS pricing_configs CASCADE;")
    op.execute("DROP TABLE IF EXISTS branding_configs CASCADE;")
    
    # Clean up added indexes, constraints, and columns from the 'tenant' table
    # Wrapped in try/except blocks so it won't crash if they are partially missing
    try:
        op.drop_index('idx_tenant_parent_workspace_id', table_name='tenant')
    except Exception:
        pass

    try:
        op.drop_index('idx_tenant_sub_accounts', table_name='tenant')
    except Exception:
        pass

    try:
        op.drop_constraint('chk_tenant_workspace_type', table_name='tenant', type_='check')
    except Exception:
        pass

    try:
        op.drop_column('tenant', 'workspace_type')
    except Exception:
        pass

    try:
        op.drop_column('tenant', 'parent_workspace_id')
    except Exception:
        pass