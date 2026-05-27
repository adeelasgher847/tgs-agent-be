"""Seed gemini-2.5-flash catalog row for Vertex voice agents.

Revision ID: 20260527_gemini_25_flash
Revises: 20260526_numconfig_rename
Create Date: 2026-05-27

Inserts an active ``model`` row for ``gemini-2.5-flash`` under the Google provider
so ``AgentService._resolve_llm_model`` accepts llmModel on create/update.

Auth for this model is via GOOGLE_APPLICATION_CREDENTIALS (Vertex ADC), not model.api_key.
Idempotent: skips insert if an active row already exists.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260527_gemini_25_flash"
down_revision: Union[str, Sequence[str], None] = "20260526_numconfig_rename"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO provider (id, name, api_key, is_active, created_at)
        SELECT gen_random_uuid(), 'Google', NULL, true, NOW()
        WHERE NOT EXISTS (
            SELECT 1 FROM provider
            WHERE lower(name) LIKE '%google%'
               OR lower(name) LIKE '%gemini%'
        );
        """
    )

    op.execute(
        """
        INSERT INTO model (
            id,
            provider_id,
            model_name,
            api_key,
            description,
            archive,
            created_at
        )
        SELECT
            gen_random_uuid(),
            p.id,
            'gemini-2.5-flash',
            NULL,
            'Gemini 2.5 Flash via Vertex AI (GOOGLE_APPLICATION_CREDENTIALS / ADC)',
            false,
            NOW()
        FROM provider p
        WHERE (lower(p.name) LIKE '%google%' OR lower(p.name) LIKE '%gemini%')
          AND NOT EXISTS (
              SELECT 1 FROM model m
              WHERE m.model_name = 'gemini-2.5-flash'
                AND m.archive = false
          )
        ORDER BY p.created_at ASC
        LIMIT 1;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM model
        WHERE model_name = 'gemini-2.5-flash'
          AND description LIKE '%Vertex AI%';
        """
    )
