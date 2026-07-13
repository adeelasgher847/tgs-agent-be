"""add smart callback scheduler tables and columns

Revision ID: 20260615_callback
Revises: 7d1c547d8a1f
Create Date: 2026-06-15 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260615_callback"
down_revision: Union[str, None] = "7d1c547d8a1f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. New smart-callback config columns on the agent table ──────────────
    op.add_column(
        "agent",
        sa.Column(
            "smart_callback_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "agent",
        sa.Column(
            "max_callback_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("5"),
        ),
    )
    op.add_column(
        "agent",
        sa.Column(
            "callback_gap_schedule",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "agent",
        sa.Column("callback_timezone", sa.Text(), nullable=True),
    )

    # ── 2. Self-referential parent_call_id on callsession ────────────────────
    op.add_column(
        "callsession",
        sa.Column(
            "parent_call_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_callsession_parent_call_id",
        "callsession",
        "callsession",
        ["parent_call_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_callsession_parent_call_id",
        "callsession",
        ["parent_call_id"],
    )

    # ── 3. callbackschedule table ─────────────────────────────────────────────
    op.create_table(
        "callbackschedule",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "original_call_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("phone_number", sa.Text(), nullable=False),
        sa.Column(
            "attempt_number",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "scheduled_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("timezone", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "executed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["original_call_id"],
            ["callsession.id"],
            ondelete="CASCADE",
            name="fk_callbackschedule_original_call_id",
        ),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
            ondelete="CASCADE",
            name="fk_callbackschedule_agent_id",
        ),
        sa.CheckConstraint(
            "status IN ('pending','executed','cancelled','exhausted')",
            name="ck_callbackschedule_status",
        ),
    )
    op.create_index(
        "ix_callbackschedule_original_call_id",
        "callbackschedule",
        ["original_call_id"],
    )
    op.create_index(
        "ix_callbackschedule_agent_id",
        "callbackschedule",
        ["agent_id"],
    )
    # Composite index used by the polling query: pending rows due now
    op.create_index(
        "ix_callbackschedule_status_scheduled_at",
        "callbackschedule",
        ["status", "scheduled_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_callbackschedule_status_scheduled_at", table_name="callbackschedule")
    op.drop_index("ix_callbackschedule_agent_id", table_name="callbackschedule")
    op.drop_index("ix_callbackschedule_original_call_id", table_name="callbackschedule")
    op.drop_table("callbackschedule")

    op.drop_index("ix_callsession_parent_call_id", table_name="callsession")
    op.drop_constraint("fk_callsession_parent_call_id", "callsession", type_="foreignkey")
    op.drop_column("callsession", "parent_call_id")

    op.drop_column("agent", "callback_timezone")
    op.drop_column("agent", "callback_gap_schedule")
    op.drop_column("agent", "max_callback_attempts")
    op.drop_column("agent", "smart_callback_enabled")
