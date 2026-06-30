"""allow_owner_role

Revision ID: 86cc29de8616
Revises: 20260630_add_allowed_email_domains
Create Date: 2026-07-01 01:29:25.419957

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '86cc29de8616'
down_revision: Union[str, Sequence[str], None] = '20260630_add_allowed_email_domains'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing check constraint
    op.execute("ALTER TABLE role DROP CONSTRAINT IF EXISTS ck_role_name_valid")
    
    # Re-add check constraint with 'owner' included
    op.create_check_constraint(
        "ck_role_name_valid",
        "role",
        "name IN ('owner', 'admin', 'manager', 'config_only', 'read_only', 'billing_only')"
    )
    
    # Insert the owner role
    op.execute(
        """
        INSERT INTO role (id, name, description, created_at)
        VALUES (gen_random_uuid(), 'owner', 'Workspace owner with full access', NOW())
        ON CONFLICT (name) DO NOTHING
        """
    )


def downgrade() -> None:
    # Delete 'owner' role
    op.execute("DELETE FROM role WHERE name = 'owner'")
    
    # Drop constraint
    op.execute("ALTER TABLE role DROP CONSTRAINT IF EXISTS ck_role_name_valid")
    
    # Re-add constraint without 'owner'
    op.create_check_constraint(
        "ck_role_name_valid",
        "role",
        "name IN ('admin', 'manager', 'config_only', 'read_only', 'billing_only')"
    )
