"""Add TTS provider and voice catalog tables.

Revision ID: 20260415_add_tts_provider_and_voice
Revises: 1f2e3d4c5b6a, d9e4f6a1b2c3
Create Date: 2026-04-15
"""

from alembic import op

revision = "20260415_add_tts_provider_and_voice"
down_revision = ("1f2e3d4c5b6a", "d9e4f6a1b2c3")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ttsprovider (
            id uuid PRIMARY KEY,
            slug VARCHAR(50) NOT NULL UNIQUE,
            display_name VARCHAR(100) NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            supports_streaming BOOLEAN NOT NULL DEFAULT FALSE,
            supports_ssml BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_ttsprovider_id ON ttsprovider(id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ttsprovider_slug ON ttsprovider(slug);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ttsvoice (
            id uuid PRIMARY KEY,
            provider_id uuid NOT NULL REFERENCES ttsprovider(id),
            external_voice_id VARCHAR(255) NOT NULL,
            display_name VARCHAR(255) NOT NULL,
            language_code VARCHAR(20),
            gender VARCHAR(32),
            accent VARCHAR(64),
            description TEXT,
            preview_audio_url VARCHAR(1000),
            sample_rate_hz INTEGER,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            metadata_json JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ,
            CONSTRAINT uq_ttsvoice_provider_external UNIQUE (provider_id, external_voice_id)
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_ttsvoice_id ON ttsvoice(id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ttsvoice_provider_id ON ttsvoice(provider_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ttsvoice_is_active ON ttsvoice(is_active);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ttsvoice_language_code ON ttsvoice(language_code);")

    op.execute(
        """
        ALTER TABLE agent
        ADD COLUMN IF NOT EXISTS tts_provider_id uuid NULL REFERENCES ttsprovider(id);
        """
    )
    op.execute(
        """
        ALTER TABLE agent
        ADD COLUMN IF NOT EXISTS tts_voice_id uuid NULL REFERENCES ttsvoice(id);
        """
    )
    op.execute(
        """
        ALTER TABLE agent
        ADD COLUMN IF NOT EXISTS tts_settings_json JSONB NULL;
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_tts_provider_id ON agent(tts_provider_id);")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_tts_voice_id ON agent(tts_voice_id);")

    op.execute(
        """
        INSERT INTO ttsprovider (
            id, slug, display_name, is_active, supports_streaming, supports_ssml
        )
        VALUES (
            '11111111-1111-1111-1111-111111111111', 'elevenlabs', 'ElevenLabs', TRUE, FALSE, TRUE
        )
        ON CONFLICT (slug) DO NOTHING;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_tts_voice_id;")
    op.execute("DROP INDEX IF EXISTS ix_agent_tts_provider_id;")
    op.execute("ALTER TABLE agent DROP COLUMN IF EXISTS tts_settings_json;")
    op.execute("ALTER TABLE agent DROP COLUMN IF EXISTS tts_voice_id;")
    op.execute("ALTER TABLE agent DROP COLUMN IF EXISTS tts_provider_id;")

    op.execute("DROP INDEX IF EXISTS ix_ttsvoice_language_code;")
    op.execute("DROP INDEX IF EXISTS ix_ttsvoice_is_active;")
    op.execute("DROP INDEX IF EXISTS ix_ttsvoice_provider_id;")
    op.execute("DROP INDEX IF EXISTS ix_ttsvoice_id;")
    op.execute("DROP TABLE IF EXISTS ttsvoice;")

    op.execute("DROP INDEX IF EXISTS ix_ttsprovider_slug;")
    op.execute("DROP INDEX IF EXISTS ix_ttsprovider_id;")
    op.execute("DROP TABLE IF EXISTS ttsprovider;")
