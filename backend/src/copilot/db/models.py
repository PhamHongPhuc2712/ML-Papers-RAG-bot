"""SQLAlchemy models owned by the foundation migration."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
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
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="version", cascade="all, delete-orphan", order_by="Chunk.ordinal"
    )


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


class Chunk(Base):
    """One token window of a parsed document version, with source spans."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_version_id", "ordinal", name="uq_chunks_version_ordinal"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True)
    paper_version_id: Mapped[UUID] = mapped_column(
        SAUUID(as_uuid=True),
        ForeignKey("paper_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    section_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="body")
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    fragment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    version: Mapped[PaperVersion] = relationship(back_populates="chunks")


JOB_STATUSES = ("queued", "running", "succeeded", "retry_wait", "failed", "cancelled")


class Job(Base):
    """Durable unit of work leased by workers (spec §4 jobs and publication)."""

    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),
        CheckConstraint(
            "status IN ('queued','running','succeeded','retry_wait','failed','cancelled')",
            name="ck_jobs_status",
        ),
        Index("ix_jobs_status_next_attempt", "status", "next_attempt_at"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_token: Mapped[UUID | None] = mapped_column(SAUUID(as_uuid=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    progress_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class SourceCheckpoint(Base):
    """Pagination cursor and run provenance per source partition."""

    __tablename__ = "source_checkpoints"
    __table_args__ = (
        UniqueConstraint("source", "partition", name="uq_source_checkpoints_source_partition"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    partition: Mapped[str] = mapped_column(String(255), nullable=False)
    cursor: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class SearchOrdering(Base):
    """A search's ordering, kept briefly so signed cursors can page through it (spec §7).

    Only hashes of the query are stored. Rows expire after the configured
    lifetime and go with their release when it is dropped.
    """

    __tablename__ = "search_orderings"
    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="ck_search_orderings_expiry"),
        Index("ix_search_orderings_key_expiry", "cache_key", "expires_at"),
        Index("ix_search_orderings_expiry", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    query_key: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_release_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("corpus_releases.id", ondelete="CASCADE", name="fk_search_orderings_release"),
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    warnings: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LlmCall(Base):
    """One priced hosted-LLM call: its reservation, then what it actually cost (spec §12).

    A row counts at ``estimated_usd`` against the daily cap until it is settled,
    then at ``cost_usd``. There is no column for a prompt, a query or a response.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (
        CheckConstraint("purpose in ('search', 'evaluation')", name="ck_llm_calls_purpose"),
        CheckConstraint(
            "status in ('reserved', 'succeeded', 'failed')", name="ck_llm_calls_status"
        ),
        CheckConstraint(
            "estimated_usd >= 0 and (cost_usd is null or cost_usd >= 0)",
            name="ck_llm_calls_nonnegative",
        ),
        Index("ix_llm_calls_created_at", "created_at"),
        Index("ix_llm_calls_run_id", "run_id"),
    )

    id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True)
    request_id: Mapped[UUID | None] = mapped_column(SAUUID(as_uuid=True), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    purpose: Mapped[str] = mapped_column(String(16), nullable=False)
    model_key: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    served_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    estimated_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
