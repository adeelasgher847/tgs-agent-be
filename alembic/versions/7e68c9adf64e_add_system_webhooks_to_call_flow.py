"""add_system_webhooks_to_call_flow

Revision ID: 7e68c9adf64e
Revises: 57d29c10d96d
Create Date: 2026-08-25 08:31:56.695084

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7e68c9adf64e'
down_revision: Union[str, Sequence[str], None] = '57d29c10d96d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'systemwebhookdeliverylog',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('call_flow_id', sa.UUID(), nullable=False),
        sa.Column('call_session_id', sa.UUID(), nullable=True),
        sa.Column('webhook_kind', sa.String(length=20), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=True),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('attempt_count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed', 'timeout')", name='ck_systemwebhookdeliverylog_status'),
        sa.CheckConstraint("webhook_kind IN ('pre_inbound', 'post_call', 'status')", name='ck_systemwebhookdeliverylog_webhook_kind'),
        sa.ForeignKeyConstraint(['call_flow_id'], ['callflow.id'], ),
        sa.ForeignKeyConstraint(['call_session_id'], ['callsession.id'], ),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_systemwebhookdeliverylog_call_flow_id_created_at', 'systemwebhookdeliverylog', ['call_flow_id', 'created_at'], unique=False)
    op.create_index('ix_systemwebhookdeliverylog_tenant_id', 'systemwebhookdeliverylog', ['tenant_id'], unique=False)

    op.add_column('callflow', sa.Column('pre_inbound_webhook_url', sa.String(length=2048), nullable=True))
    op.add_column('callflow', sa.Column('pre_inbound_webhook_headers_encrypted', sa.Text(), nullable=True))
    op.add_column('callflow', sa.Column('pre_inbound_webhook_query_params', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'"), nullable=False))
    op.add_column('callflow', sa.Column('pre_inbound_webhook_static_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'"), nullable=False))
    op.add_column('callflow', sa.Column('dynamic_inbound_routing_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('post_call_webhook_url', sa.String(length=2048), nullable=True))
    op.add_column('callflow', sa.Column('post_call_webhook_headers_encrypted', sa.Text(), nullable=True))
    op.add_column('callflow', sa.Column('post_call_webhook_query_params', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'"), nullable=False))
    op.add_column('callflow', sa.Column('post_call_webhook_custom_payload_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('post_call_webhook_custom_payload_template', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('callflow', sa.Column('status_webhook_enabled', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('callflow', sa.Column('status_webhook_url', sa.String(length=2048), nullable=True))
    op.add_column('callflow', sa.Column('status_webhook_headers_encrypted', sa.Text(), nullable=True))
    op.add_column('callflow', sa.Column('status_webhook_query_params', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'"), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('callflow', 'status_webhook_query_params')
    op.drop_column('callflow', 'status_webhook_headers_encrypted')
    op.drop_column('callflow', 'status_webhook_url')
    op.drop_column('callflow', 'status_webhook_enabled')
    op.drop_column('callflow', 'post_call_webhook_custom_payload_template')
    op.drop_column('callflow', 'post_call_webhook_custom_payload_enabled')
    op.drop_column('callflow', 'post_call_webhook_query_params')
    op.drop_column('callflow', 'post_call_webhook_headers_encrypted')
    op.drop_column('callflow', 'post_call_webhook_url')
    op.drop_column('callflow', 'dynamic_inbound_routing_enabled')
    op.drop_column('callflow', 'pre_inbound_webhook_static_metadata')
    op.drop_column('callflow', 'pre_inbound_webhook_query_params')
    op.drop_column('callflow', 'pre_inbound_webhook_headers_encrypted')
    op.drop_column('callflow', 'pre_inbound_webhook_url')

    op.drop_index('ix_systemwebhookdeliverylog_tenant_id', table_name='systemwebhookdeliverylog')
    op.drop_index('ix_systemwebhookdeliverylog_call_flow_id_created_at', table_name='systemwebhookdeliverylog')
    op.drop_table('systemwebhookdeliverylog')
