"""public_access + alloweddomain

Backs the Web SDK public-call-token endpoints:
  callflow.public_access  BOOL DEFAULT FALSE NOT NULL
  alloweddomain table     — per-workspace Origin whitelist (table name follows
                            the base-class convention: lowercased class name,
                            not the ticket's literal "allowed_domains")

Revision ID: 20260618_public_call_token
Revises: 20260618_audit_actor_prefix_widen
Create Date: 2026-06-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "20260618_public_call_token"
down_revision: Union[str, Sequence[str], None] = "20260618_audit_actor_prefix_widen"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "callflow",
        sa.Column("public_access", sa.Boolean(), nullable=False, server_default="false"),
    )

    op.create_table(
        "alloweddomain",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False),
        sa.Column("domain", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_alloweddomain_workspace_id", "alloweddomain", ["workspace_id"])


def downgrade() -> None:
    op.drop_index("ix_alloweddomain_workspace_id", table_name="alloweddomain")
    op.drop_table("alloweddomain")
    op.drop_column("callflow", "public_access")
