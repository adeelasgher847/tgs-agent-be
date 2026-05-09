"""Add product_links table for tenant product associations.

Revision ID: 20260429_add_product_links
Revises: 20260425_add_slot_reservation
Create Date: 2026-04-29
"""

from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "20260429_add_product_links"
down_revision: Union[str, Sequence[str], None] = "20260425_add_slot_reservation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector, name: str) -> bool:
    return name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if _has_table(inspector, "product_links"):
        return

    op.create_table(
        "product_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenant.id"), nullable=False),
        sa.Column("source_product", sa.String(length=40), nullable=False),
        sa.Column("target_product", sa.String(length=40), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index("ix_product_links_id", "product_links", ["id"], unique=False)
    op.create_index("ix_product_links_tenant_id", "product_links", ["tenant_id"], unique=False)
    op.create_index(
        "uq_product_links_active_pair",
        "product_links",
        ["tenant_id", "source_product", "target_product"],
        unique=True,
    )
    op.create_check_constraint(
        "ck_product_links_source_product",
        "product_links",
        "source_product IN ('TALENTSYNC', 'ASSISTLY')",
    )
    op.create_check_constraint(
        "ck_product_links_target_product",
        "product_links",
        "target_product IN ('TALENTSYNC', 'ASSISTLY')",
    )

    tenant_rows = bind.execute(sa.text("SELECT id FROM tenant")).fetchall()
    for row in tenant_rows:
        tenant_id = row[0]
        exists = bind.execute(
            sa.text(
                """
                SELECT 1
                FROM product_links
                WHERE tenant_id = :tenant_id
                  AND source_product = 'TALENTSYNC'
                  AND target_product = 'ASSISTLY'
                LIMIT 1
                """
            ),
            {"tenant_id": tenant_id},
        ).fetchone()
        if exists:
            continue
        bind.execute(
            sa.text(
                """
                INSERT INTO product_links (id, tenant_id, source_product, target_product, is_active, linked_at)
                VALUES (:id, :tenant_id, 'TALENTSYNC', 'ASSISTLY', TRUE, now())
                """
            ),
            {"id": str(uuid.uuid4()), "tenant_id": tenant_id},
        )


def downgrade() -> None:
    op.drop_constraint("ck_product_links_target_product", "product_links", type_="check")
    op.drop_constraint("ck_product_links_source_product", "product_links", type_="check")
    op.drop_index("uq_product_links_active_pair", table_name="product_links", if_exists=True)
    op.drop_index("ix_product_links_tenant_id", table_name="product_links", if_exists=True)
    op.drop_index("ix_product_links_id", table_name="product_links", if_exists=True)
    op.drop_table("product_links")
