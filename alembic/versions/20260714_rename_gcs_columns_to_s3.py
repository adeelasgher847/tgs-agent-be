"""rename_gcs_columns_to_s3

Renames GCS-named storage path columns to S3-neutral equivalents
across all four tables that held GCS object paths.

    callsession.recording_gcs_path  -> recording_s3_path
    batchjob.gcs_path               -> s3_path
    dataexportjob.gcs_path          -> s3_path
    kbfile.gcs_path                 -> s3_path

The column types and constraints are unchanged — only the names change.
Data already in these columns (S3 object keys written after the GCS→S3
storage migration) is preserved automatically by ALTER TABLE … RENAME.

Revision ID: 20260714_rename_gcs_columns_to_s3
Revises: d414601cdea1
Create Date: 2026-07-14
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260714_rename_gcs_columns_to_s3"
down_revision: Union[str, Sequence[str], None] = "d414601cdea1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # callsession: recording_gcs_path -> recording_s3_path
    op.alter_column(
        "callsession",
        "recording_gcs_path",
        new_column_name="recording_s3_path",
    )

    # batchjob: gcs_path -> s3_path
    op.alter_column(
        "batchjob",
        "gcs_path",
        new_column_name="s3_path",
    )

    # dataexportjob: gcs_path -> s3_path
    op.alter_column(
        "dataexportjob",
        "gcs_path",
        new_column_name="s3_path",
    )

    # kbfile: gcs_path -> s3_path
    op.alter_column(
        "kbfile",
        "gcs_path",
        new_column_name="s3_path",
    )


def downgrade() -> None:
    # kbfile: s3_path -> gcs_path
    op.alter_column(
        "kbfile",
        "s3_path",
        new_column_name="gcs_path",
    )

    # dataexportjob: s3_path -> gcs_path
    op.alter_column(
        "dataexportjob",
        "s3_path",
        new_column_name="gcs_path",
    )

    # batchjob: s3_path -> gcs_path
    op.alter_column(
        "batchjob",
        "s3_path",
        new_column_name="gcs_path",
    )

    # callsession: recording_s3_path -> recording_gcs_path
    op.alter_column(
        "callsession",
        "recording_s3_path",
        new_column_name="recording_gcs_path",
    )
