"""Add canonical paper identity and source provenance tables."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_corpus"
down_revision: str | None = "0000_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _created_at() -> sa.Column[object]:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


def _json_column(name: str, *, nullable: bool = False) -> sa.Column[object]:
    return sa.Column(
        name,
        postgresql.JSONB(astext_type=sa.Text()),
        nullable=nullable,
        server_default=sa.text("'{}'::jsonb") if not nullable else None,
    )


def upgrade() -> None:
    op.create_table(
        "venues",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("track", sa.String(length=255), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_venues"),
        sa.UniqueConstraint("name", "track", name="uq_venues_name_track"),
    )
    op.create_table(
        "papers",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("venue_id", _uuid(), nullable=True),
        sa.Column("first_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column(
            "acceptance_type", sa.String(length=64), nullable=False, server_default="unknown"
        ),
        sa.Column("metadata_status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("merged_into", _uuid(), nullable=True),
        sa.ForeignKeyConstraint(["merged_into"], ["papers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["venue_id"], ["venues.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_papers"),
    )
    op.create_index("ix_papers_merged_into", "papers", ["merged_into"])

    op.create_table(
        "authors",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_authors"),
        sa.UniqueConstraint("normalized_name", name="uq_authors_normalized_name"),
    )
    op.create_table(
        "paper_identifiers",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("paper_id", _uuid(), nullable=False),
        sa.Column("namespace", sa.String(length=64), nullable=False),
        sa.Column("value", sa.String(length=512), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_identifiers"),
        sa.UniqueConstraint(
            "namespace", "value", name="uq_paper_identifiers_namespace_value"
        ),
    )
    op.create_index(
        "ix_paper_identifiers_lookup", "paper_identifiers", ["namespace", "value"]
    )

    op.create_table(
        "paper_versions",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("paper_id", _uuid(), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_revision", sa.String(length=255), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column("license_label", sa.String(length=255), nullable=True),
        sa.Column("redistribution", sa.String(length=32), nullable=False, server_default="unknown"),
        sa.Column(
            "parser_version", sa.String(length=255), nullable=False, server_default="unparsed"
        ),
        sa.Column("parse_status", sa.String(length=32), nullable=False, server_default="pending"),
        _created_at(),
        sa.CheckConstraint(
            "redistribution IN ('eligible', 'restricted', 'unknown')",
            name="ck_paper_versions_redistribution",
        ),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_versions"),
        sa.UniqueConstraint(
            "paper_id",
            "source",
            "source_revision",
            "content_sha256",
            "version",
            name="uq_paper_versions_source_revision_content_version",
        ),
    )

    op.create_table(
        "paper_authors",
        sa.Column("paper_id", _uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("author_id", _uuid(), nullable=False),
        sa.ForeignKeyConstraint(["author_id"], ["authors.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("paper_id", "position", name="pk_paper_authors"),
        sa.UniqueConstraint("paper_id", "author_id", name="uq_paper_authors_paper_author"),
    )

    op.create_table(
        "source_records",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("paper_id", _uuid(), nullable=True),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("source_item_id", sa.String(length=512), nullable=True),
        sa.Column("source_revision", sa.String(length=255), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("original_title", sa.Text(), nullable=True),
        sa.Column("acceptance_decision", sa.String(length=64), nullable=True),
        _json_column("raw_json"),
        _created_at(),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_source_records"),
        sa.UniqueConstraint(
            "source",
            "source_revision",
            "content_sha256",
            name="uq_source_records_revision_checksum",
        ),
    )

    op.create_table(
        "field_provenance",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("source_record_id", _uuid(), nullable=False),
        sa.Column("paper_id", _uuid(), nullable=True),
        sa.Column("field_name", sa.String(length=128), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("original_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("source_revision", sa.String(length=255), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_record_id"], ["source_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_field_provenance"),
        sa.UniqueConstraint(
            "source_record_id", "field_name", name="uq_field_provenance_record_field"
        ),
    )

    op.create_table(
        "identity_conflicts",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="open"),
        sa.Column("paper_id", _uuid(), nullable=True),
        sa.Column("other_paper_id", _uuid(), nullable=True),
        sa.Column("source_record_id", _uuid(), nullable=True),
        _json_column("details"),
        _created_at(),
        sa.ForeignKeyConstraint(["other_paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_record_id"], ["source_records.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_identity_conflicts"),
    )

    op.create_table(
        "quarantine_records",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("source_revision", sa.String(length=255), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        _json_column("raw_json"),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_quarantine_records"),
        sa.UniqueConstraint(
            "source",
            "source_revision",
            "content_sha256",
            "reason",
            name="uq_quarantine_records_input_reason",
        ),
    )

    op.create_table(
        "paper_redirects",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("from_paper_id", _uuid(), nullable=False),
        sa.Column("to_paper_id", _uuid(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "merged_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint("from_paper_id <> to_paper_id", name="ck_paper_redirects_distinct"),
        sa.ForeignKeyConstraint(["from_paper_id"], ["papers.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["to_paper_id"], ["papers.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_redirects"),
        sa.UniqueConstraint("from_paper_id", name="uq_paper_redirects_from_paper"),
    )


def downgrade() -> None:
    op.drop_table("paper_redirects")
    op.drop_table("quarantine_records")
    op.drop_table("identity_conflicts")
    op.drop_table("field_provenance")
    op.drop_table("source_records")
    op.drop_table("paper_authors")
    op.drop_table("paper_versions")
    op.drop_index("ix_paper_identifiers_lookup", table_name="paper_identifiers")
    op.drop_table("paper_identifiers")
    op.drop_table("authors")
    op.drop_index("ix_papers_merged_into", table_name="papers")
    op.drop_table("papers")
    op.drop_table("venues")
