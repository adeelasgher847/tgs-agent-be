"""add business knowledge table

Revision ID: 20260429_add_businessknowledge
Revises: 20260429_user_products_link
Create Date: 2026-04-29 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260429_add_businessknowledge"
down_revision: Union[str, Sequence[str], None] = "20260429_user_products_link"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "businessknowledge",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenant.id"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent.id"),
            nullable=True,
            index=True,
        ),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("business_name", sa.String(255), nullable=True),
        sa.Column("business_type", sa.String(255), nullable=True),
        sa.Column("business_description", sa.Text, nullable=True),
        sa.Column("address", sa.Text, nullable=True),
        sa.Column("phone", sa.String(255), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("website_url", sa.String(512), nullable=True),
        sa.Column("primary_service", sa.Text, nullable=True),
        sa.Column("secondary_service", sa.Text, nullable=True),
        sa.Column("service_areas", sa.Text, nullable=True),
        sa.Column("specializations", sa.Text, nullable=True),
        sa.Column("pricing_information", sa.Text, nullable=True),
        sa.Column("additional_information", sa.Text, nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # tenant_id/agent_id columns above already declare index=True, which
    # creates ix_businessknowledge_tenant_id/ix_businessknowledge_agent_id
    # as part of create_table — these explicit calls were redundant and
    # failed with DuplicateTable the first time this migration ever ran
    # against a genuinely fresh database (every prior environment inherited
    # an already-migrated DB, so this never surfaced before).


def downgrade() -> None:
    op.drop_index("ix_businessknowledge_agent_id", table_name="businessknowledge")
    op.drop_index("ix_businessknowledge_tenant_id", table_name="businessknowledge")
    op.drop_table("businessknowledge")
