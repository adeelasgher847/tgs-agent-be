"""add_auditlog_table_for_hipaa

Revision ID: 5b152b97d6b5
Revises: 20260617_hipaa
Create Date: 2026-06-17 01:31:07.344000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5b152b97d6b5'
down_revision: Union[str, Sequence[str], None] = '20260617_hipaa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'auditlog',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=True),
        sa.Column('action', sa.String(length=128), nullable=False),
        sa.Column('resource_type', sa.String(length=64), nullable=True),
        sa.Column('resource_id', sa.UUID(), nullable=True),
        sa.Column('old_value', sa.Text(), nullable=True),
        sa.Column('new_value', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_auditlog_action', 'auditlog', ['action'], unique=False)
    op.create_index('ix_auditlog_tenant_id', 'auditlog', ['tenant_id'], unique=False)
    op.create_index('ix_auditlog_timestamp', 'auditlog', ['timestamp'], unique=False)
    op.create_index('ix_auditlog_tenant_action', 'auditlog', ['tenant_id', 'action'], unique=False)
    op.create_index('ix_auditlog_tenant_timestamp', 'auditlog', ['tenant_id', 'timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_auditlog_tenant_timestamp', table_name='auditlog')
    op.drop_index('ix_auditlog_tenant_action', table_name='auditlog')
    op.drop_index('ix_auditlog_timestamp', table_name='auditlog')
    op.drop_index('ix_auditlog_tenant_id', table_name='auditlog')
    op.drop_index('ix_auditlog_action', table_name='auditlog')
    op.drop_table('auditlog')