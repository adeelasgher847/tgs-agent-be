"""add speechmatics stt provider and catalog model

Revision ID: 20260813_speechmatics_stt
Revises: 20260813_drop_bk_appt
Create Date: 2026-08-13

New sttprovider row for Speechmatics + one sttmodel row ("enhanced") using
MULAW 8kHz (Twilio-native path, same as Deepgram). metadata_json carries
Speechmatics-specific engine params (end_of_utterance_silence_trigger,
max_delay, max_delay_mode) so resolve_stt_runtime() needs no code changes.
"""
from typing import Sequence, Union

import json

from alembic import op
import sqlalchemy as sa

revision: str = "20260813_speechmatics_stt"
down_revision: Union[str, Sequence[str], None] = "20260813_drop_bk_appt"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(
        sa.text(
            """
            INSERT INTO sttprovider (id, slug, display_name, is_active, supports_streaming)
            VALUES (gen_random_uuid(), 'speechmatics', 'Speechmatics', true, true)
            ON CONFLICT (slug) DO NOTHING
            """
        )
    )

    speechmatics_id = conn.execute(
        sa.text("SELECT id FROM sttprovider WHERE slug = 'speechmatics'")
    ).scalar()
    if speechmatics_id is None:
        return

    model_rows = [
        (
            "enhanced",
            "Speechmatics (Enhanced)",
            "en",
            8000,
            "MULAW",
            {
                "model": "enhanced",
                "end_of_utterance_silence_trigger": 0.5,
                "max_delay": 2.0,
                "max_delay_mode": "flexible",
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
                "provider_id": str(speechmatics_id),
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
            WHERE provider_id IN (SELECT id FROM sttprovider WHERE slug = 'speechmatics')
            """
        )
    )
    op.execute(sa.text("DELETE FROM sttprovider WHERE slug = 'speechmatics'"))
