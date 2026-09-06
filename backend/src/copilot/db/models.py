"""SQLAlchemy models owned by the foundation migration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import UUID as SAUUID


def _uuid() -> UUID:
    """Return a generated primary key for identity tables."""

    return uuid4()


class Base(DeclarativeBase):
    """Declarative metadata for all application models."""


class CorpusRelease(Base):
    """Immutable metadata describing one validated public vector release."""

    __tablename__ = "corpus_releases"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    paper_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    counts: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    active_pointer: Mapped[ActiveRelease | None] = relationship(
        back_populates="release", uselist=False
    )


class ActiveRelease(Base):
    """Singleton pointer to the release served by request handlers."""

    __tablename__ = "active_release"
    __table_args__ = (
        CheckConstraint("singleton_key = 'active'", name="ck_active_release_singleton_key"),
    )

    singleton_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    corpus_release_id: Mapped[str] = mapped_column(
        ForeignKey("corpus_releases.id", ondelete="RESTRICT"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    release: Mapped[CorpusRelease] = relationship(back_populates="active_pointer")

    @property
    def key(self) -> str:
        """Compatibility spelling for callers that call the singleton field ``key``."""

        return self.singleton_key


class Venue(Base):
    """A publication venue and track used by canonical paper metadata."""

    __tablename__ = "venues"
    __table_args__ = (UniqueConstraint("name", "track", name="uq_venues_name_track"),)

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    track: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    papers: Mapped[list[Paper]] = relationship(back_populates="venue")


class Paper(Base):
    """Stable work identity independent of source or document versions."""

    __tablename__ = "papers"

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue_id: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("venues.id", ondelete="SET NULL"), nullable=True
    )
    first_published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    acceptance_type: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    metadata_status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    merged_into: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    venue: Mapped[Venue | None] = relationship(back_populates="papers", foreign_keys=[venue_id])
    identifiers: Mapped[list[PaperIdentifier]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    versions: Mapped[list[PaperVersion]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    authorships: Mapped[list[PaperAuthor]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="PaperAuthor.position"
    )
    source_records: Mapped[list[SourceRecord]] = relationship(back_populates="paper")


class PaperIdentifier(Base):
    """Canonical external alias for a work, with arXiv version kept separate."""

    __tablename__ = "paper_identifiers"
    __table_args__ = (
        UniqueConstraint("namespace", "value", name="uq_paper_identifiers_namespace_value"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    paper_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    namespace: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(String(512), nullable=False)
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    paper: Mapped[Paper] = relationship(back_populates="identifiers")


class PaperVersion(Base):
    """A source document/version belonging to a stable paper work."""

    __tablename__ = "paper_versions"
    __table_args__ = (
        UniqueConstraint(
            "paper_id",
            "source",
            "source_revision",
            "content_sha256",
            "version",
            name="uq_paper_versions_source_revision_content_version",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "redistribution IN ('eligible', 'restricted', 'unknown')",
            name="ck_paper_versions_redistribution",
        ),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    paper_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    license_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    redistribution: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    parser_version: Mapped[str] = mapped_column(String(255), nullable=False, default="unparsed")
    parse_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    paper: Mapped[Paper] = relationship(back_populates="versions")


class Author(Base):
    """An author identity normalized conservatively for matching."""

    __tablename__ = "authors"
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_authors_normalized_name"),)

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    paper_links: Mapped[list[PaperAuthor]] = relationship(back_populates="author")


class PaperAuthor(Base):
    """Ordered association preserving the author order supplied by a source."""

    __tablename__ = "paper_authors"
    __table_args__ = (
        UniqueConstraint("paper_id", "author_id", name="uq_paper_authors_paper_author"),
    )

    paper_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    author_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("authors.id", ondelete="RESTRICT"), nullable=False
    )

    paper: Mapped[Paper] = relationship(back_populates="authorships")
    author: Mapped[Author] = relationship(back_populates="paper_links")


class SourceRecord(Base):
    """Immutable source metadata observation and its local raw JSON artifact."""

    __tablename__ = "source_records"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "source_revision",
            "content_sha256",
            name="uq_source_records_revision_checksum",
        ),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    paper_id: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_item_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    acceptance_decision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    paper: Mapped[Paper | None] = relationship(back_populates="source_records")
    fields: Mapped[list[FieldProvenance]] = relationship(
        back_populates="source_record", cascade="all, delete-orphan"
    )


class FieldProvenance(Base):
    """Value-level provenance for each canonical metadata field."""

    __tablename__ = "field_provenance"
    __table_args__ = (
        UniqueConstraint("source_record_id", "field_name", name="uq_field_provenance_record_field"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    source_record_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True
    )
    field_name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)
    original_value: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    source_record: Mapped[SourceRecord] = relationship(back_populates="fields")


class IdentityConflict(Base):
    """Durable review record for contradictory IDs or incompatible metadata."""

    __tablename__ = "identity_conflicts"
    __table_args__ = (
        UniqueConstraint(
            "source_record_id",
            "reason",
            name="uq_identity_conflicts_source_reason",
        ),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    paper_id: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True
    )
    other_paper_id: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("source_records.id", ondelete="SET NULL"), nullable=True
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    @property
    def existing_paper_id(self) -> UUID | None:
        """Compatibility spelling used by review tooling."""

        return self.paper_id


class QuarantineRecord(Base):
    """Source input retained for review when a supplied identifier is invalid."""

    __tablename__ = "quarantine_records"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "source_revision",
            "content_sha256",
            "reason",
            name="uq_quarantine_records_input_reason",
        ),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class PaperRedirect(Base):
    """History-preserving redirect from a manually merged work to its survivor."""

    __tablename__ = "paper_redirects"
    __table_args__ = (
        UniqueConstraint("from_paper_id", name="uq_paper_redirects_from_paper"),
        CheckConstraint("from_paper_id <> to_paper_id", name="ck_paper_redirects_distinct"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    from_paper_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="RESTRICT"), nullable=False
    )
    to_paper_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True), ForeignKey("papers.id", ondelete="RESTRICT"), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    merged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
