"""Corpus release staging and the guarded active-release switch.

A release moves ``staged -> validated -> active``. Staging records which Qdrant
collections and model revision a release *would* serve; it grants nothing. The
pointer only moves once an index validator confirms the matching paper and chunk
collections exist with the expected counts and dimensions (spec §4).

Those indexes are built by P2.2. Until then the default validator refuses, so a
snapshot exported here cannot accidentally start serving against collections
that do not exist. The prior release row is left untouched for rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import ActiveRelease, CorpusRelease

STAGED = "staged"
VALIDATED = "validated"
ACTIVE = "active"
SUPERSEDED = "superseded"


class ReleaseError(Exception):
    """Base class for release lifecycle failures."""


class ReleaseNotFoundError(ReleaseError):
    """Raised when the named release does not exist."""


class ReleaseNotValidatedError(ReleaseError):
    """Raised when activation is attempted before the indexes validate."""

    def __init__(self, release_id: str, reasons: list[str]) -> None:
        self.release_id = release_id
        self.reasons = reasons
        super().__init__(f"{release_id}:{','.join(reasons) or 'not_validated'}")


@dataclass(frozen=True)
class IndexValidation:
    """Outcome of checking a release's serving indexes."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


class IndexValidator(Protocol):
    def __call__(self, release: CorpusRelease) -> IndexValidation:
        """Confirm the release's paper and chunk collections are servable."""


def refuse_until_indexes_exist(release: CorpusRelease) -> IndexValidation:
    """Default validator: nothing can be activated before P2.2 builds indexes.

    This is deliberately a refusal rather than a no-op. An empty check that
    returned ``ok`` would let a release become active with no vectors behind it,
    which readiness would then report as healthy.
    """

    _ = release
    return IndexValidation(False, ["index_validation_unavailable_until_p2_2"])


def stage_release(
    session: Session,
    *,
    release_id: str,
    manifest_sha256: str,
    paper_collection: str,
    chunk_collection: str,
    model_revision: str,
    counts: dict[str, Any] | None = None,
) -> CorpusRelease:
    """Record a release candidate. Staging never changes what is served."""

    existing = session.get(CorpusRelease, release_id)
    if existing is not None:
        raise ReleaseError(f"release_exists:{release_id}")
    if paper_collection == chunk_collection:
        raise ReleaseError("paper_and_chunk_collections_must_differ")
    release = CorpusRelease(
        id=release_id,
        manifest_sha256=manifest_sha256,
        paper_collection=paper_collection,
        chunk_collection=chunk_collection,
        model_revision=model_revision,
        status=STAGED,
        counts=dict(counts or {}),
        created_at=datetime.now(UTC),
    )
    session.add(release)
    session.flush()
    return release


def activate_release(
    session: Session,
    release_id: str,
    *,
    validator: IndexValidator = refuse_until_indexes_exist,
) -> CorpusRelease:
    """Validate the release's indexes, then move the singleton pointer atomically."""

    release = session.get(CorpusRelease, release_id)
    if release is None:
        raise ReleaseNotFoundError(release_id)

    outcome = validator(release)
    if not outcome.ok:
        raise ReleaseNotValidatedError(release_id, list(outcome.reasons))

    release.status = VALIDATED
    session.flush()

    pointer = session.execute(
        select(ActiveRelease).where(ActiveRelease.singleton_key == ACTIVE).with_for_update()
    ).scalar_one_or_none()
    previous_id = pointer.corpus_release_id if pointer is not None else None
    if pointer is None:
        session.add(
            ActiveRelease(
                singleton_key=ACTIVE,
                corpus_release_id=release.id,
                updated_at=datetime.now(UTC),
            )
        )
    else:
        pointer.corpus_release_id = release.id
        pointer.updated_at = datetime.now(UTC)

    # The prior release row is kept for rollback; only its status changes.
    if previous_id is not None and previous_id != release.id:
        previous = session.get(CorpusRelease, previous_id)
        if previous is not None:
            previous.status = SUPERSEDED

    release.status = ACTIVE
    session.flush()
    return release
