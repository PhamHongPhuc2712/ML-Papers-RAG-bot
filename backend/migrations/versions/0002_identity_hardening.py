"""Make review conflicts idempotent for a source record and reason."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_identity_hardening"
down_revision: str | None = "0001_corpus"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
