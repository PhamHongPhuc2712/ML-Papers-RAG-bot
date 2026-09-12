"""Add an indexed normalized title so identity resolution stops scanning.

Every record whose identifiers are all new falls through to title matching,
which loaded the whole papers table and normalized each title in Python. That
is O(n) per insert and O(n^2) per ingestion run: measured at 32 ms per resolve
with 271 papers, projecting to about 1 second at 8,905 and roughly 10 seconds
at the specification's eventual corpus size.

Revision ID: 0004_paper_normalized_title
Revises: 0003_jobs
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_paper_normalized_title"
down_revision = "0003_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("normalized_title", sa.Text(), nullable=True))
    # Backfill with the same NFKC + whitespace-collapse + casefold the application
    # applies, so existing rows match without a re-ingest. Postgres normalize()
    # provides NFKC; regexp_replace collapses runs of whitespace.
    op.execute(
        """
        UPDATE papers
           SET normalized_title =
               lower(btrim(regexp_replace(normalize(title, NFKC), '\s+', ' ', 'g')))
        """
    )
    op.alter_column("papers", "normalized_title", nullable=False)
    op.create_index("ix_papers_normalized_title", "papers", ["normalized_title"])


def downgrade() -> None:
    op.drop_index("ix_papers_normalized_title", table_name="papers")
    op.drop_column("papers", "normalized_title")
