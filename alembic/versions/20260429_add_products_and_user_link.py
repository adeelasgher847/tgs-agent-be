"""Add products table and link users to products via user_tenant_association.

Revision ID: 20260429_user_products_link
Revises: 20260429_add_product_links
Create Date: 2026-04-29
"""

from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "20260429_user_products_link"
down_revision: Union[str, Sequence[str], None] = "20260429_add_product_links"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector, name: str) -> bool:
    return name in inspector.get_table_names()


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not _has_table(inspector, "product"):
        op.create_table(
            "product",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        )
        op.create_index("ix_product_id", "product", ["id"], unique=False)
        op.create_index("ix_product_name", "product", ["name"], unique=True)

    # Seed enum-backed products.
    for product_name in ("TALENTSYNC", "ASSISTLY"):
        exists = bind.execute(
            sa.text("SELECT id FROM product WHERE name = :name LIMIT 1"),
            {"name": product_name},
        ).fetchone()
        if not exists:
            bind.execute(
                sa.text("INSERT INTO product (id, name, created_at) VALUES (:id, :name, now())"),
                {"id": str(uuid.uuid4()), "name": product_name},
            )

    # Add product_id to user_tenant_association (role-like linking).
    inspector = inspect(bind)
    if _has_table(inspector, "user_tenant_association") and not _has_column(
        inspector, "user_tenant_association", "product_id"
    ):
        op.add_column(
            "user_tenant_association",
            sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
        op.create_foreign_key(
            "fk_user_tenant_association_product_id",
            "user_tenant_association",
            "product",
            ["product_id"],
            ["id"],
        )
        op.create_index(
            "ix_user_tenant_association_product_id",
            "user_tenant_association",
            ["product_id"],
            unique=False,
        )

    default_product = bind.execute(
        sa.text("SELECT id FROM product WHERE name = 'TALENTSYNC' LIMIT 1")
    ).fetchone()
    if default_product:
        bind.execute(
            sa.text(
                """
                UPDATE user_tenant_association
                SET product_id = :pid
                WHERE product_id IS NULL
                """
            ),
            {"pid": default_product[0]},
        )

    # Drop incorrectly introduced product_links table if present.
    inspector = inspect(bind)
    if _has_table(inspector, "product_links"):
        op.execute("DROP TABLE IF EXISTS product_links CASCADE")


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if _has_table(inspector, "user_tenant_association") and _has_column(
        inspector, "user_tenant_association", "product_id"
    ):
        op.drop_index("ix_user_tenant_association_product_id", table_name="user_tenant_association")
        op.drop_constraint("fk_user_tenant_association_product_id", "user_tenant_association", type_="foreignkey")
        op.drop_column("user_tenant_association", "product_id")

    if _has_table(inspector, "product"):
        op.drop_index("ix_product_name", table_name="product")
        op.drop_index("ix_product_id", table_name="product")
        op.drop_table("product")
