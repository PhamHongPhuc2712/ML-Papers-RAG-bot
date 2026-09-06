"""Make document-version identity unique when version is NULL."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_version_hardening"
down_revision: str | None = "0002_identity_hardening"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VERSION_CONSTRAINT = "uq_paper_versions_source_revision_content_version"


def upgrade() -> None:
    # PostgreSQL's regular UNIQUE constraint treats NULL values as distinct.
    # Retain the earliest legacy row before replacing it with NULLS NOT DISTINCT.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY paper_id, source, source_revision,
                                    content_sha256, version
                       ORDER BY created_at ASC, id ASC
                   ) AS row_number
            FROM paper_versions
        )
        DELETE FROM paper_versions
        WHERE id IN (
            SELECT id FROM ranked WHERE row_number > 1
        )
        """
    )
    op.drop_constraint(_VERSION_CONSTRAINT, "paper_versions", type_="unique")
    op.execute(
        f"""
        ALTER TABLE paper_versions
        ADD CONSTRAINT {_VERSION_CONSTRAINT}
        UNIQUE NULLS NOT DISTINCT
            (paper_id, source, source_revision, content_sha256, version)
        """
    )


def downgrade() -> None:
    op.drop_constraint(_VERSION_CONSTRAINT, "paper_versions", type_="unique")
    op.create_unique_constraint(
        _VERSION_CONSTRAINT,
        "paper_versions",
        ["paper_id", "source", "source_revision", "content_sha256", "version"],
    )
