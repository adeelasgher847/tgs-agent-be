"""rename number_configurations to numberconfiguration (Base tablename)

Revision ID: 20260526_numconfig_rename
Revises: a1b2c3d4e5f6
Create Date: 2026-05-26

Idempotent: only renames when the legacy table exists.
"""

from alembic import op
from sqlalchemy import inspect

revision = "20260526_numconfig_rename"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if "number_configurations" in tables and "numberconfiguration" not in tables:
        op.rename_table("number_configurations", "numberconfiguration")


def downgrade() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if "numberconfiguration" in tables and "number_configurations" not in tables:
        op.rename_table("numberconfiguration", "number_configurations")
