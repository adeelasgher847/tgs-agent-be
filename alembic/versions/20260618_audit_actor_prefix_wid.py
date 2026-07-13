"""audit_actor_prefix_widen

actor_api_key_prefix was VARCHAR(8) — exactly enough for the 8-char API key
prefix it normally stores, but one short of the 9-char '[DELETED]'
placeholder the GDPR erasure trigger (20260618_audit_update_bypass) writes
on anonymization. Widening to 16 to give headroom for both.

Revision ID: 20260618_audit_actor_prefix_widen
Revises: 20260618_user_email_partial_unique
Create Date: 2026-06-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260618_audit_actor_prefix_wid"
down_revision: Union[str, Sequence[str], None] = "20260618_user_email_partial_uq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "auditlog",
        "actor_api_key_prefix",
        existing_type=sa.String(8),
        type_=sa.String(16),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "auditlog",
        "actor_api_key_prefix",
        existing_type=sa.String(16),
        type_=sa.String(8),
        existing_nullable=True,
    )
