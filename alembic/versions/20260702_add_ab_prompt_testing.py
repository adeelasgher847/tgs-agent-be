"""Add A/B prompt testing columns to callflow and callsession

Revision ID: 20260702_ab_prompt_testing
Revises: 86cc29de8616
Create Date: 2026-07-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260702_ab_prompt_testing"
down_revision: Union[str, Sequence[str], None] = "86cc29de8616"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "callflow",
        sa.Column(
            "ab_test_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "callflow",
        sa.Column(
            "ab_prompt_a_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "callflow",
        sa.Column(
            "ab_prompt_b_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "callflow",
        sa.Column(
            "ab_split_ratio",
            sa.Numeric(3, 2),
            nullable=False,
            server_default="0.50",
        ),
    )
    op.create_foreign_key(
        "fk_callflow_ab_prompt_a",
        "callflow",
        "promptversion",
        ["ab_prompt_a_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_callflow_ab_prompt_b",
        "callflow",
        "promptversion",
        ["ab_prompt_b_id"],
        ["id"],
    )
    op.create_check_constraint(
        "ck_callflow_ab_split_ratio",
        "callflow",
        "ab_split_ratio > 0 AND ab_split_ratio < 1",
    )

    op.add_column(
        "callsession",
        sa.Column("ab_variant", sa.String(length=1), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("callsession", "ab_variant")

    op.drop_constraint("ck_callflow_ab_split_ratio", "callflow", type_="check")
    op.drop_constraint("fk_callflow_ab_prompt_b", "callflow", type_="foreignkey")
    op.drop_constraint("fk_callflow_ab_prompt_a", "callflow", type_="foreignkey")
    op.drop_column("callflow", "ab_split_ratio")
    op.drop_column("callflow", "ab_prompt_b_id")
    op.drop_column("callflow", "ab_prompt_a_id")
    op.drop_column("callflow", "ab_test_enabled")
