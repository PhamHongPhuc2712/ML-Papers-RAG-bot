"""Make review conflicts idempotent for a source record and reason."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_identity_hardening"
down_revision: str | None = "0001_corpus"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The first implementation could emit the same review conflict more than
    # once during replay. Retain the earliest row for each source/reason key
    # before adding the constraint required by the resolver.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY source_record_id, reason
                       ORDER BY created_at ASC, id ASC
                   ) AS row_number
            FROM identity_conflicts
            WHERE source_record_id IS NOT NULL
        )
        DELETE FROM identity_conflicts
        WHERE id IN (
            SELECT id FROM ranked WHERE row_number > 1
        )
        """
    )
    op.create_unique_constraint(
        "uq_identity_conflicts_source_reason",
        "identity_conflicts",
        ["source_record_id", "reason"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_identity_conflicts_source_reason",
        "identity_conflicts",
        type_="unique",
    )
