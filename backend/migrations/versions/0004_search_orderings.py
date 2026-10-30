"""Add cached search orderings for signed-cursor pagination (spec §7)."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_search_orderings"
down_revision: str | None = "0003_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_orderings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("query_key", sa.String(length=64), nullable=False),
        sa.Column("corpus_release_id", sa.String(length=255), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("items", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("expires_at > created_at", name="ck_search_orderings_expiry"),
        # A dropped release takes its cached orderings with it.
        sa.ForeignKeyConstraint(
            ["corpus_release_id"],
            ["corpus_releases.id"],
            name="fk_search_orderings_release",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_search_orderings"),
    )
    op.create_index(
        "ix_search_orderings_key_expiry", "search_orderings", ["cache_key", "expires_at"]
    )
    op.create_index("ix_search_orderings_expiry", "search_orderings", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_search_orderings_expiry", table_name="search_orderings")
    op.drop_index("ix_search_orderings_key_expiry", table_name="search_orderings")
    op.drop_table("search_orderings")
