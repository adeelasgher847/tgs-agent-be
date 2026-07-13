"""schema v1 gaps: updated_at on core tables, user soft-delete, FK SET NULL, invite composite index, UUID server defaults

Revision ID: 20260521_schema_v1_gaps
Revises: 20260518_tenant_name_uq
Create Date: 2026-05-21 00:00:00.000000

Run:
    alembic upgrade head
Revert:
    alembic downgrade -1
"""
from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260521_schema_v1_gaps"
down_revision: Union[str, Sequence[str], None] = "20260518_tenant_name_uq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_USER_FK_NAME = "user_current_tenant_id_fkey"


def _has_constraint(conn, table: str, constraint: str) -> bool:
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_schema = 'public' AND table_name = :tbl AND constraint_name = :con)"
        ),
        {"tbl": table, "con": constraint},
    ).scalar()


def _fk_on_column(conn, table: str, column: str) -> Optional[str]:
    return conn.execute(
        sa.text(
            "SELECT tc.constraint_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_schema = kcu.constraint_schema "
            "  AND tc.constraint_name = kcu.constraint_name "
            "WHERE tc.table_schema = 'public' "
            "  AND tc.table_name = :tbl "
            "  AND tc.constraint_type = 'FOREIGN KEY' "
            "  AND kcu.column_name = :col "
            "LIMIT 1"
        ),
        {"tbl": table, "col": column},
    ).scalar()


def _drop_user_current_tenant_fk(conn) -> None:
    fk_name = _fk_on_column(conn, "user", "current_tenant_id")
    if fk_name:
        op.drop_constraint(fk_name, "user", type_="foreignkey")


def upgrade() -> None:
    # ------------------------------------------------------------------ tenant
    op.add_column(
        "tenant",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Ensure DB-level UUID default (belt-and-suspenders alongside Python uuid4)
    op.execute(
        sa.text(
            "ALTER TABLE tenant ALTER COLUMN id SET DEFAULT gen_random_uuid()"
        )
    )

    # -------------------------------------------------------------------- user
    op.add_column(
        "user",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "user",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_user_deleted_at", "user", ["deleted_at"], unique=False)

    # Drop existing FK (name may differ from SQLAlchemy default) and recreate with ON DELETE SET NULL.
    conn = op.get_bind()
    _drop_user_current_tenant_fk(conn)
    if not _has_constraint(conn, "user", _USER_FK_NAME):
        op.create_foreign_key(
            _USER_FK_NAME,
            "user",
            "tenant",
            ["current_tenant_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.execute(
        sa.text(
            "ALTER TABLE \"user\" ALTER COLUMN id SET DEFAULT gen_random_uuid()"
        )
    )

    # ------------------------------------------------------------------ apikey
    op.add_column(
        "apikey",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            "ALTER TABLE apikey ALTER COLUMN id SET DEFAULT gen_random_uuid()"
        )
    )

    # ------------------------------------------------------------------ invite
    op.add_column(
        "invite",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Composite index required by ticket
    op.create_index(
        "ix_invite_email_tenant_id",
        "invite",
        ["email", "tenant_id"],
        unique=False,
    )
    # DB-level default for expires_at: NOW() + 7 days (app code may override)
    op.execute(
        sa.text(
            "ALTER TABLE invite ALTER COLUMN expires_at SET DEFAULT NOW() + INTERVAL '7 days'"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE invite ALTER COLUMN id SET DEFAULT gen_random_uuid()"
        )
    )

    # ------------------------------------------------------------ refreshtoken
    op.add_column(
        "refreshtoken",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            "ALTER TABLE refreshtoken ALTER COLUMN id SET DEFAULT gen_random_uuid()"
        )
    )


def downgrade() -> None:
    # ------------------------------------------------------------ refreshtoken
    op.drop_column("refreshtoken", "updated_at")
    op.execute(
        sa.text("ALTER TABLE refreshtoken ALTER COLUMN id DROP DEFAULT")
    )

    # ------------------------------------------------------------------ invite
    op.drop_index("ix_invite_email_tenant_id", table_name="invite")
    op.drop_column("invite", "updated_at")
    op.execute(
        sa.text("ALTER TABLE invite ALTER COLUMN expires_at DROP DEFAULT")
    )
    op.execute(
        sa.text("ALTER TABLE invite ALTER COLUMN id DROP DEFAULT")
    )

    # ------------------------------------------------------------------ apikey
    op.drop_column("apikey", "updated_at")
    op.execute(
        sa.text("ALTER TABLE apikey ALTER COLUMN id DROP DEFAULT")
    )

    # -------------------------------------------------------------------- user
    conn = op.get_bind()
    _drop_user_current_tenant_fk(conn)
    if not _fk_on_column(conn, "user", "current_tenant_id"):
        op.create_foreign_key(
            _USER_FK_NAME,
            "user",
            "tenant",
            ["current_tenant_id"],
            ["id"],
        )
    op.drop_index("ix_user_deleted_at", table_name="user")
    op.drop_column("user", "deleted_at")
    op.drop_column("user", "updated_at")
    op.execute(
        sa.text("ALTER TABLE \"user\" ALTER COLUMN id DROP DEFAULT")
    )

    # ------------------------------------------------------------------ tenant
    op.drop_column("tenant", "updated_at")
    op.execute(
        sa.text("ALTER TABLE tenant ALTER COLUMN id DROP DEFAULT")
    )
