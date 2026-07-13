"""add HIPAA compliance fields to callflow and tenant

Revision ID: 20260617_hipaa
Revises: (merges all current heads)
Create Date: 2026-06-17

Adds:
  callflow.hipaa_compliance  BOOL DEFAULT FALSE NOT NULL
  tenant.kms_key_name        TEXT NULLABLE  (Cloud KMS key resource name)
  tenant.baa_on_file         BOOL DEFAULT FALSE NOT NULL  (manually set by admin)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260617_hipaa"
down_revision: Union[str, Sequence[str], None] = (
    "20260415_add_tts_provider_and_voice",
    "20260420_resume_candidate_status",
    "20260518_stripe_fulfillment",
    "20260521_role_config_readonly",
    "20260526_numconfig_rename",
    "20260529_call_flows",
    "20260608_stt_catalog",
    "20260616_callback_arq",
    "20260608_outbound_status_idx",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "callflow",
        sa.Column(
            "hipaa_compliance",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "tenant",
        sa.Column("kms_key_name", sa.Text(), nullable=True),
    )
    op.add_column(
        "tenant",
        sa.Column(
            "baa_on_file",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )


def downgrade() -> None:
    op.drop_column("tenant", "baa_on_file")
    op.drop_column("tenant", "kms_key_name")
    op.drop_column("callflow", "hipaa_compliance")
