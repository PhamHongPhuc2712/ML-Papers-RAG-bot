"""SQLAlchemy models owned by the foundation migration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
