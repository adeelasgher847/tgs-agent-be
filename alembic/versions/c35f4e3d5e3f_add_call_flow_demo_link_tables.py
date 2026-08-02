"""add_call_flow_demo_link_tables

Revision ID: c35f4e3d5e3f
Revises: 20260729_nova_meta
Create Date: 2026-07-31 19:59:54.232646

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c35f4e3d5e3f'
down_revision: Union[str, Sequence[str], None] = '20260729_nova_meta'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'callflowdemolink',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('call_flow_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('token', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('per_user_limit_minutes', sa.Numeric(), nullable=True),
        sa.Column('total_budget_minutes', sa.Numeric(), nullable=True),
        sa.Column('total_minutes_used', sa.Numeric(), server_default='0', nullable=False),
        sa.Column('created_by_user_id', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['call_flow_id'], ['callflow.id'], ),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['user.id'], ),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_callflowdemolink_call_flow_id', 'callflowdemolink', ['call_flow_id'], unique=False)
    op.create_index('ix_callflowdemolink_tenant_id', 'callflowdemolink', ['tenant_id'], unique=False)
    op.create_index(op.f('ix_callflowdemolink_token'), 'callflowdemolink', ['token'], unique=True)

    op.create_table(
        'callflowdemolinkvisitorusage',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('demo_link_id', sa.UUID(), nullable=False),
        sa.Column('visitor_id', sa.String(), nullable=False),
        sa.Column('minutes_used', sa.Numeric(), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['demo_link_id'], ['callflowdemolink.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('demo_link_id', 'visitor_id', name='uq_calldemolink_visitor'),
    )
    op.create_index(
        'ix_callflowdemolinkvisitorusage_demo_link_id',
        'callflowdemolinkvisitorusage',
        ['demo_link_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_callflowdemolinkvisitorusage_visitor_id'),
        'callflowdemolinkvisitorusage',
        ['visitor_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_callflowdemolinkvisitorusage_visitor_id'), table_name='callflowdemolinkvisitorusage')
    op.drop_index('ix_callflowdemolinkvisitorusage_demo_link_id', table_name='callflowdemolinkvisitorusage')
    op.drop_table('callflowdemolinkvisitorusage')

    op.drop_index(op.f('ix_callflowdemolink_token'), table_name='callflowdemolink')
    op.drop_index('ix_callflowdemolink_tenant_id', table_name='callflowdemolink')
    op.drop_index('ix_callflowdemolink_call_flow_id', table_name='callflowdemolink')
    op.drop_table('callflowdemolink')
