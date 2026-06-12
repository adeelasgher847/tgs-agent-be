"""add_missing_agent_columns

Revision ID: 7d1c547d8a1f
Revises: da61d0d331c1
Create Date: 2026-06-12 21:25:47.080730

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7d1c547d8a1f'
down_revision: Union[str, Sequence[str], None] = 'da61d0d331c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('agent', sa.Column('smart_callback', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('agent', sa.Column('stt_provider_slug', sa.String(length=40), nullable=True))
    op.add_column('agent', sa.Column('stt_model_external_id', sa.String(length=255), nullable=True))
    op.add_column('agent', sa.Column('stt_language_code', sa.String(length=20), nullable=True))
    op.add_column('agent', sa.Column('stt_provider_id', sa.UUID(), nullable=True))
    op.add_column('agent', sa.Column('stt_model_id', sa.UUID(), nullable=True))
    op.add_column('agent', sa.Column('stt_settings_json', sa.JSON(), nullable=True))
    op.drop_index(op.f('ix_agent_status'), table_name='agent')
    op.create_index(op.f('ix_agent_stt_model_id'), 'agent', ['stt_model_id'], unique=False)
    op.create_index(op.f('ix_agent_stt_provider_id'), 'agent', ['stt_provider_id'], unique=False)
    op.create_index('ix_agent_tenant_id', 'agent', ['tenant_id'], unique=False)
    op.create_foreign_key(None, 'agent', 'sttmodel', ['stt_model_id'], ['id'], ondelete='SET NULL')
    op.create_foreign_key(None, 'agent', 'ttsvoice', ['tts_voice_id'], ['id'])
    op.create_foreign_key(None, 'agent', 'sttprovider', ['stt_provider_id'], ['id'], ondelete='SET NULL')
    op.create_foreign_key(None, 'agent', 'ttsprovider', ['tts_provider_id'], ['id'])
    op.create_index('ix_callflow_agent_id', 'callflow', ['agent_id'], unique=False)
    op.add_column('callsession', sa.Column('recording_gcs_path', sa.String(length=500), nullable=True))
    op.add_column('callsession', sa.Column('recording_error', sa.Boolean(), server_default='false', nullable=False))
    op.create_index(op.f('ix_jobdescription_id'), 'jobdescription', ['id'], unique=False)
    op.alter_column('phonenumber', 'sip_password',
               existing_type=sa.TEXT(),
               type_=sa.String(length=500),
               existing_nullable=True)
    op.create_index(op.f('ix_ttsprovider_id'), 'ttsprovider', ['id'], unique=False)
    op.create_index(op.f('ix_ttsprovider_slug'), 'ttsprovider', ['slug'], unique=True)
    op.create_index(op.f('ix_ttsvoice_id'), 'ttsvoice', ['id'], unique=False)
    op.create_index(op.f('ix_ttsvoice_is_active'), 'ttsvoice', ['is_active'], unique=False)
    op.create_index(op.f('ix_ttsvoice_language_code'), 'ttsvoice', ['language_code'], unique=False)
    op.create_index(op.f('ix_ttsvoice_provider_id'), 'ttsvoice', ['provider_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    pass
