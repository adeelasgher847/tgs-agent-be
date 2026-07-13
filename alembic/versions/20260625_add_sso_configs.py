"""Add sso_configs table

Revision ID: b1c2d3e4f5a6
Revises: a9f2b3c4d5e6
Create Date: 2026-06-25 20:31:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = 'a9f2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'sso_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'workspace_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('tenant.id', ondelete='CASCADE'),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            'protocol',
            sa.String(10),
            sa.CheckConstraint("protocol IN ('saml', 'oidc')", name='chk_sso_protocol'),
            nullable=False,
        ),
        sa.Column('idp_entity_id', sa.Text(), nullable=True),
        sa.Column('idp_sso_url', sa.Text(), nullable=True),
        sa.Column('idp_x509_certificate', sa.Text(), nullable=True),
        sa.Column('oidc_client_id', sa.Text(), nullable=True),
        sa.Column('oidc_client_secret', sa.Text(), nullable=True),   # Fernet-encrypted
        sa.Column('oidc_discovery_url', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()'), nullable=True),
    )
    op.create_index('idx_sso_configs_workspace_id', 'sso_configs', ['workspace_id'])


def downgrade() -> None:
    op.drop_index('idx_sso_configs_workspace_id', table_name='sso_configs')
    op.drop_table('sso_configs')
