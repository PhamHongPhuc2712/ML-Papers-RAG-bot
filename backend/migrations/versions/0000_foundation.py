"""Create corpus release and active release pointer tables."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0000_foundation"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "corpus_releases",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("paper_collection", sa.String(length=255), nullable=False),
        sa.Column("chunk_collection", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "counts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_corpus_releases"),
    )
    op.create_table(
        "active_release",
        sa.Column("singleton_key", sa.String(length=64), nullable=False),
        sa.Column("corpus_release_id", sa.String(length=255), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["corpus_release_id"],
            ["corpus_releases.id"],
            name="fk_active_release_corpus_release_id",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "singleton_key = 'active'",
            name="ck_active_release_singleton_key",
        ),
        sa.PrimaryKeyConstraint("singleton_key", name="pk_active_release"),
    )


def downgrade() -> None:
    op.drop_table("active_release")
    op.drop_table("corpus_releases")
