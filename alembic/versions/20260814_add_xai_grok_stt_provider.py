"""add xai grok stt provider and catalog model

Revision ID: 20260814_xai_grok_stt
Revises: 20260814_elevenlabs_scribe_stt
Create Date: 2026-08-14

Seeds a new 'xai' sttprovider row + one sttmodel row (MULAW 8kHz, Twilio-
native, matching Deepgram/Speechmatics/ElevenLabs convention).

xAI's STT API (wss://api.x.ai/v1/stt) has no model-selection parameter at
all -- confirmed absent from both the REST and streaming query-param tables
in xAI's official docs. external_model_id is schema-required (NOT NULL) but
is a catalog-only placeholder here ('default'), never sent to xAI as an API
parameter -- see app/services/xai_grok_stt_service.py's create_streaming_session().

metadata_json carries xAI's native turn-detection tuning (endpointing_ms,
smart_turn_threshold, smart_turn_timeout_ms) so resolve_stt_runtime() needs
no code changes, same pattern as every other provider's catalog row.
"""
from typing import Sequence, Union

import json

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_xai_grok_stt"
down_revision: Union[str, Sequence[str], None] = "20260814_elevenlabs_scribe_stt"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(
        sa.text(
            """
            INSERT INTO sttprovider (id, slug, display_name, is_active, supports_streaming)
            VALUES (gen_random_uuid(), 'xai', 'xAI Grok', true, true)
            ON CONFLICT (slug) DO NOTHING
            """
        )
    )

    xai_id = conn.execute(
        sa.text("SELECT id FROM sttprovider WHERE slug = 'xai'")
    ).scalar()
    if xai_id is None:
        return

    model_rows = [
        (
            "default",
            "xAI Grok STT",
            "en",
            8000,
            "MULAW",
            {
                "endpointing_ms": 10,
                "smart_turn_threshold": 0.5,
                "smart_turn_timeout_ms": 3000,
            },
        ),
    ]
    for external_model_id, display_name, language_code, sample_rate_hz, encoding, metadata in model_rows:
        conn.execute(
            sa.text(
                """
                INSERT INTO sttmodel (
                  id, provider_id, external_model_id, display_name,
                  language_code, sample_rate_hz, encoding, metadata_json, is_active
                )
                VALUES (
                  gen_random_uuid(), CAST(:provider_id AS uuid), :external_model_id,
                  :display_name, :language_code, :sample_rate_hz, :encoding,
                  CAST(:metadata_json AS jsonb), true
                )
                ON CONFLICT (provider_id, external_model_id) DO NOTHING
                """
            ),
            {
                "provider_id": str(xai_id),
                "external_model_id": external_model_id,
                "display_name": display_name,
                "language_code": language_code,
                "sample_rate_hz": sample_rate_hz,
                "encoding": encoding,
                "metadata_json": json.dumps(metadata),
            },
        )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM sttmodel
            WHERE provider_id IN (SELECT id FROM sttprovider WHERE slug = 'xai')
            """
        )
    )
    op.execute(sa.text("DELETE FROM sttprovider WHERE slug = 'xai'"))
