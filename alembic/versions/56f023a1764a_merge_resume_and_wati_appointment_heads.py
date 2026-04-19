"""merge_resume_and_wati_appointment_heads

Revision ID: 56f023a1764a
Revises: 20260414_resume_jd_setnull, 20260418_appointment_wati_staff_prompt
Create Date: 2026-04-20 02:29:19.659207

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56f023a1764a'
down_revision: Union[str, Sequence[str], None] = ('20260414_resume_jd_setnull', '20260418_appointment_wati_staff_prompt')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
