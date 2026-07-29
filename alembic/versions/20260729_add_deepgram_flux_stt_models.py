"""add deepgram flux stt models to catalog

Revision ID: 20260729_flux_stt
Revises: 85f24740c31a
Create Date: 2026-07-29

Seeds flux-general-en and flux-general-multi under the existing deepgram
sttprovider row. No schema changes -- sttmodel already supports arbitrary
provider-specific config via metadata_json.
"""
from typing import Sequence, Union

import json

from alembic import op
import sqlalchemy as sa

revision: str = "20260729_flux_stt"
down_revision: Union[str, Sequence[str], None] = "85f24740c31a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    deepgram_id = conn.execute(
        sa.text("SELECT id FROM sttprovider WHERE slug = 'deepgram'")
    ).scalar()
    if deepgram_id is None:
        # Base catalog migration hasn't run in this environment -- nothing to seed onto.
        return

    model_rows = [
        (
            "flux-general-en",
            "Deepgram Flux (English)",
            "en",
            8000,
            "MULAW",
            {"api_model": "flux-general-en", "api_version": "v2", "eot_timeout_ms": 5000},
        ),
        (
            "flux-general-multi",
            "Deepgram Flux (Multilingual)",
            "en",
            8000,
            "MULAW",
            {"api_model": "flux-general-multi", "api_version": "v2", "eot_timeout_ms": 5000},
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
                "provider_id": str(deepgram_id),
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
            WHERE external_model_id IN ('flux-general-en', 'flux-general-multi')
              AND provider_id IN (SELECT id FROM sttprovider WHERE slug = 'deepgram')
            """
        )
    )
