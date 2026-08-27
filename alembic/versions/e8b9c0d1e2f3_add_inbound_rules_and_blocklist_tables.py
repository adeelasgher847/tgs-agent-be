"""add_inbound_rules_and_blocklist_tables

Revision ID: e8b9c0d1e2f3
Revises: d7a8b9c0e1f2
Create Date: 2026-08-26 11:36:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = 'e8b9c0d1e2f3'
down_revision: Union[str, None] = 'd7a8b9c0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Create inboundruleset table
    op.create_table(
        'inboundruleset',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'tenant_id',
            UUID(as_uuid=True),
            sa.ForeignKey('tenant.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column(
            'is_deleted',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
        sa.Column(
            'created_by',
            UUID(as_uuid=True),
            sa.ForeignKey('user.id'),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index('ix_inboundruleset_id', 'inboundruleset', ['id'])
    op.create_index(
        'ix_inboundruleset_tenant_id', 'inboundruleset', ['tenant_id']
    )

    # 2. Create inboundrule table
    op.create_table(
        'inboundrule',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'tenant_id',
            UUID(as_uuid=True),
            sa.ForeignKey('tenant.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'rule_set_id',
            UUID(as_uuid=True),
            sa.ForeignKey('inboundruleset.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'phone_number_pattern', sa.String(length=50), nullable=False
        ),
        sa.Column('normalized_digits', sa.String(length=50), nullable=False),
        sa.Column('label', sa.String(length=100), nullable=True),
        sa.Column(
            'action',
            sa.String(length=20),
            server_default='deny',
            nullable=False,
        ),
        sa.Column(
            'is_deleted',
            sa.Boolean(),
            server_default='false',
            nullable=False,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index('ix_inboundrule_id', 'inboundrule', ['id'])
    op.create_index('ix_inboundrule_tenant_id', 'inboundrule', ['tenant_id'])
    op.create_index(
        'ix_inboundrule_rule_set_id', 'inboundrule', ['rule_set_id']
    )
    op.create_index(
        'ix_inboundrule_normalized_digits', 'inboundrule', ['normalized_digits']
    )
    op.create_index(
        'ix_inboundrule_tenant_normalized',
        'inboundrule',
        ['tenant_id', 'normalized_digits'],
    )
    op.create_index(
        'ix_inboundrule_set_normalized',
        'inboundrule',
        ['rule_set_id', 'normalized_digits'],
    )

    # 3. Add inbound_rule_set_id to callflow
    op.add_column(
        'callflow',
        sa.Column(
            'inbound_rule_set_id',
            UUID(as_uuid=True),
            sa.ForeignKey('inboundruleset.id', ondelete='SET NULL'),
            nullable=True,
        ),
    )
    op.create_index(
        'ix_callflow_inbound_rule_set_id',
        'callflow',
        ['inbound_rule_set_id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_callflow_inbound_rule_set_id', table_name='callflow')
    op.drop_column('callflow', 'inbound_rule_set_id')
    op.drop_table('inboundrule')
    op.drop_table('inboundruleset')
