"""add assemblyai universal-streaming stt provider and catalog models

Revision ID: 20260814_assemblyai_stt
Revises: 20260814_xai_grok_stt
Create Date: 2026-08-14

Seeds a new 'assemblyai' sttprovider row + one sttmodel row per real,
selectable `speech_model` value (universal-3-5-pro, universal-streaming-
english, universal-streaming-multilingual), MULAW 8kHz Twilio-native,
matching the Deepgram/Speechmatics/ElevenLabs/xAI convention.

Unlike xAI's placeholder 'default' external_model_id (xAI has no model
parameter at all), AssemblyAI's `speech_model` is a real, sent API
parameter -- so each catalog row's external_model_id IS the value actually
sent to AssemblyAI. See app/services/assemblyai_stt_service.py's
create_streaming_session().

metadata_json carries AssemblyAI's native turn-detection tuning
(min_turn_silence_ms, max_turn_silence_ms, end_of_turn_confidence_threshold,
vad_threshold, format_turns) so resolve_stt_runtime() needs no code changes,
same pattern as every other provider's catalog row. vad_threshold defaults
to 0.2 for universal-3-5-pro and 0.4 for the two Universal-family models,
per AssemblyAI's documented per-model defaults.
"""
from typing import Sequence, Union

import json

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_assemblyai_stt"
down_revision: Union[str, Sequence[str], None] = "20260814_xai_grok_stt"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(
        sa.text(
            """
            INSERT INTO sttprovider (id, slug, display_name, is_active, supports_streaming)
            VALUES (gen_random_uuid(), 'assemblyai', 'AssemblyAI Universal-Streaming', true, true)
            ON CONFLICT (slug) DO NOTHING
            """
        )
    )

    provider_id = conn.execute(
        sa.text("SELECT id FROM sttprovider WHERE slug = 'assemblyai'")
    ).scalar()
    if provider_id is None:
        return

    model_rows = [
        (
            "universal-3-5-pro",
            "AssemblyAI Universal 3.5 Pro",
            "en",
            8000,
            "MULAW",
            {
                "speech_model": "universal-3-5-pro",
                "min_turn_silence_ms": 400,
                "max_turn_silence_ms": 1536,
                "end_of_turn_confidence_threshold": 0.4,
                "vad_threshold": 0.2,
                "format_turns": True,
            },
        ),
        (
            "universal-streaming-english",
            "AssemblyAI Universal Streaming (English)",
            "en",
            8000,
            "MULAW",
            {
                "speech_model": "universal-streaming-english",
                "min_turn_silence_ms": 400,
                "max_turn_silence_ms": 1536,
                "end_of_turn_confidence_threshold": 0.4,
                "vad_threshold": 0.4,
                "format_turns": True,
            },
        ),
        (
            "universal-streaming-multilingual",
            "AssemblyAI Universal Streaming (Multilingual)",
            "en",
            8000,
            "MULAW",
            {
                "speech_model": "universal-streaming-multilingual",
                "min_turn_silence_ms": 400,
                "max_turn_silence_ms": 1536,
                "end_of_turn_confidence_threshold": 0.4,
                "vad_threshold": 0.4,
                "format_turns": True,
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
                "provider_id": str(provider_id),
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
            WHERE provider_id IN (SELECT id FROM sttprovider WHERE slug = 'assemblyai')
            """
        )
    )
    op.execute(sa.text("DELETE FROM sttprovider WHERE slug = 'assemblyai'"))
