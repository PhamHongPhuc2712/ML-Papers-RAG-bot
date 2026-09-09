"""Add chunks for parsed document versions."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_chunks"
down_revision: str | None = "0001_corpus"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=False),
        sa.Column("section_ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="body"),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("chunker_version", sa.String(length=64), nullable=False),
        sa.Column("evidence_default", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("fragment", sa.String(length=16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["paper_version_id"], ["paper_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint("paper_version_id", "ordinal", name="uq_chunks_version_ordinal"),
    )
    op.create_index("ix_chunks_paper_version_id", "chunks", ["paper_version_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_paper_version_id", table_name="chunks")
    op.drop_table("chunks")
