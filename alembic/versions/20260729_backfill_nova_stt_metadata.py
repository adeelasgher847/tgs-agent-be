"""backfill metadata_json for deepgram nova-2/nova-3 stt models

Revision ID: 20260729_nova_meta
Revises: 20260729_flux_stt
Create Date: 2026-07-29

No functional effect on the Nova (v1/listen) code path today -- it resolves
its model from external_model_id directly, never metadata_json. This purely
brings nova-2/nova-3 in line with the other sttmodel rows (Google/ElevenLabs/
Flux), which all carry a real metadata_json, for catalog consistency.
Only touches rows still at the empty '{}' default -- won't clobber anything
set intentionally after the original 20260608 seed.
"""
from typing import Sequence, Union

import json

from alembic import op
import sqlalchemy as sa

revision: str = "20260729_nova_meta"
down_revision: Union[str, Sequence[str], None] = "20260729_flux_stt"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    nova_metadata = {
        "nova-2": {"api_model": "nova-2", "encoding": "mulaw", "sample_rate_hz": 8000},
        "nova-3": {"api_model": "nova-3", "encoding": "mulaw", "sample_rate_hz": 8000},
    }
    for external_model_id, metadata in nova_metadata.items():
        conn.execute(
            sa.text(
                """
                UPDATE sttmodel
                SET metadata_json = CAST(:metadata_json AS jsonb)
                WHERE external_model_id = :external_model_id
                  AND provider_id IN (SELECT id FROM sttprovider WHERE slug = 'deepgram')
                  AND (metadata_json IS NULL OR metadata_json::text = '{}')
                """
            ),
            {"external_model_id": external_model_id, "metadata_json": json.dumps(metadata)},
        )


def downgrade() -> None:
    conn = op.get_bind()
    for external_model_id in ("nova-2", "nova-3"):
        conn.execute(
            sa.text(
                """
                UPDATE sttmodel
                SET metadata_json = '{}'::jsonb
                WHERE external_model_id = :external_model_id
                  AND provider_id IN (SELECT id FROM sttprovider WHERE slug = 'deepgram')
                """
            ),
            {"external_model_id": external_model_id},
        )
