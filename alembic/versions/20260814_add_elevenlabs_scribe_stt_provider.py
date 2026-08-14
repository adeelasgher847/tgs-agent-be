"""add elevenlabs scribe v2 realtime stt provider and catalog model

Revision ID: 20260814_elevenlabs_scribe_stt
Revises: 20260813_speechmatics_stt
Create Date: 2026-08-14

The base catalog migration (20260608_add_stt_catalog_and_agent_stt_columns)
already seeds a placeholder 'elevenlabs' sttprovider row (supports_streaming=
false) + a 'scrib-v1' sttmodel row for the old batch Scribe v1 model -- that
integration was never actually built. This migration:
  1. Creates the 'elevenlabs' sttprovider row if it doesn't already exist
     (verified: it does not exist in at least one live environment, despite
     being in the base migration's source -- handle both cases).
  2. Corrects supports_streaming to true on whichever row ends up in place,
     since we're now adding real realtime streaming support.
  3. Adds a new 'scribe_v2_realtime' sttmodel row (MULAW 8kHz, Twilio-native,
     matching Deepgram/Speechmatics convention) alongside -- not replacing --
     any pre-existing 'scrib-v1' row, since other environments' agents may
     already reference it by FK.
"""
from typing import Sequence, Union

import json

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_elevenlabs_scribe_stt"
down_revision: Union[str, Sequence[str], None] = "20260813_speechmatics_stt"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(
        sa.text(
            """
            INSERT INTO sttprovider (id, slug, display_name, is_active, supports_streaming)
            VALUES (gen_random_uuid(), 'elevenlabs', 'ElevenLabs Scribe', true, true)
            ON CONFLICT (slug) DO NOTHING
            """
        )
    )

    # Whether the row above was just inserted, or already existed from the
    # base catalog migration (with supports_streaming=false), correct it now.
    conn.execute(
        sa.text(
            """
            UPDATE sttprovider
            SET supports_streaming = true, is_active = true
            WHERE slug = 'elevenlabs'
            """
        )
    )

    elevenlabs_id = conn.execute(
        sa.text("SELECT id FROM sttprovider WHERE slug = 'elevenlabs'")
    ).scalar()
    if elevenlabs_id is None:
        return

    model_rows = [
        (
            "scribe_v2_realtime",
            "ElevenLabs Scribe v2 Realtime",
            "en",
            8000,
            "MULAW",
            {
                "model": "scribe_v2_realtime",
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": 1.5,
                "vad_threshold": 0.4,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 100,
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
                "provider_id": str(elevenlabs_id),
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
            WHERE external_model_id = 'scribe_v2_realtime'
              AND provider_id IN (SELECT id FROM sttprovider WHERE slug = 'elevenlabs')
            """
        )
    )
    # Provider row + supports_streaming correction intentionally left in place
    # on downgrade -- the row may predate this migration (base catalog
    # migration) in some environments, so this migration doesn't own deleting
    # it or reverting its flags.
