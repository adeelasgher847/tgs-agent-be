"""Add WATI staff prompt and customer SMS confirmation columns on appointment.

Revision ID: 20260418_appointment_wati_staff_prompt
Revises: 20260415_add_tts_provider_and_voice
Create Date: 2026-04-18
"""

from alembic import op


revision = "20260418_appointment_wati_staff_prompt"
down_revision = "20260415_add_tts_provider_and_voice"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE appointment
        ADD COLUMN IF NOT EXISTS staff_whatsapp_ack_token VARCHAR(32) NULL;
        """
    )
    op.execute(
        """
        ALTER TABLE appointment
        ADD COLUMN IF NOT EXISTS staff_whatsapp_prompt_sent_at TIMESTAMPTZ NULL;
        """
    )
    op.execute(
        """
        ALTER TABLE appointment
        ADD COLUMN IF NOT EXISTS customer_sms_confirmed_notified_at TIMESTAMPTZ NULL;
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_appointment_staff_whatsapp_ack_token
        ON appointment (staff_whatsapp_ack_token)
        WHERE staff_whatsapp_ack_token IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_appointment_staff_whatsapp_ack_token;")
    op.execute(
        "ALTER TABLE appointment DROP COLUMN IF EXISTS customer_sms_confirmed_notified_at;"
    )
    op.execute(
        "ALTER TABLE appointment DROP COLUMN IF EXISTS staff_whatsapp_prompt_sent_at;"
    )
    op.execute(
        "ALTER TABLE appointment DROP COLUMN IF EXISTS staff_whatsapp_ack_token;"
    )
