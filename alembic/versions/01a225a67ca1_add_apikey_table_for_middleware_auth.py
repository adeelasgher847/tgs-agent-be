"""add_apikey_table_for_middleware_auth

Revision ID: 01a225a67ca1
Revises: 20260505_followup_agent
Create Date: 2026-05-15 23:45:43.887604

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '01a225a67ca1'
down_revision: Union[str, Sequence[str], None] = '20260505_followup_agent'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'apikey',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash'),
    )
    op.create_index('ix_apikey_key_hash', 'apikey', ['key_hash'], unique=False)
    op.create_index('ix_apikey_tenant_id', 'apikey', ['tenant_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_apikey_tenant_id', table_name='apikey')
    op.drop_index('ix_apikey_key_hash', table_name='apikey')
    op.drop_table('apikey')
