"""Widen ``search_orderings.mode`` for ``hybrid_rerank_llm``, 17 characters (spec §7)."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_search_ordering_mode"
down_revision: str | None = "0005_llm_calls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "search_orderings",
        "mode",
        existing_type=sa.String(length=16),
        type_=sa.String(length=32),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Orderings expire within minutes; any deep-search row would not fit the old width.
    op.execute("delete from search_orderings where length(mode) > 16")
    op.alter_column(
        "search_orderings",
        "mode",
        existing_type=sa.String(length=32),
        type_=sa.String(length=16),
        existing_nullable=False,
    )
