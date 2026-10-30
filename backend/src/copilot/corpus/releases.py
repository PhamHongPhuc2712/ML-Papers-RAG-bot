"""Staged corpus releases and the transactional active-release switch (spec §4).

A release names the two Qdrant collections a request will read and the model
revision that built them. It is created ``staged``, each collection build is
recorded against it, and it can only become active once something has actually
checked those collections — counts, dimensions, checksums and search canaries,
which ``search.index`` owns. The validator is injected here so this module never
guesses whether an index is serviceable, and so the invariant "no active
release without a validated index pair" holds whoever builds the index.

A request reads the pointer once and keeps the release it captured: collection
names never change after staging, so a request that started on the previous
release finishes on it even if the pointer moves underneath it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, text

# Returns True to allow activation. It may instead raise ReleaseError itself,
# which is how a validator reports *why* an index pair is not serviceable.
Validator = Callable[[Mapping[str, Any]], bool]


class ReleaseError(RuntimeError):
    """A release cannot be staged or activated."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class ReleaseRecord:
    """One release row. The collection pair and model revision are fixed at staging."""

    id: str
    manifest_sha256: str
    paper_collection: str
    chunk_collection: str
    model_revision: str
    status: str
    counts: Mapping[str, Any]

    def collection(self, kind: str) -> str:
        if kind == "papers":
            return self.paper_collection
        if kind == "chunks":
            return self.chunk_collection
        raise ReleaseError("collection_kind_unknown", kind)

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "manifest_sha256": self.manifest_sha256,
            "paper_collection": self.paper_collection,
            "chunk_collection": self.chunk_collection,
            "model_revision": self.model_revision,
            "status": self.status,
            "counts": dict(self.counts),
        }


_RELEASE_COLUMNS = (
    "id, manifest_sha256, paper_collection, chunk_collection, model_revision, status, counts"
)


def _record(row: Mapping[Any, Any]) -> ReleaseRecord:
    return ReleaseRecord(
        id=str(row["id"]),
        manifest_sha256=str(row["manifest_sha256"]),
        paper_collection=str(row["paper_collection"]),
        chunk_collection=str(row["chunk_collection"]),
        model_revision=str(row["model_revision"]),
        status=str(row["status"]),
        counts=dict(row["counts"] or {}),
    )


def load_release(engine: Engine, release_id: str) -> ReleaseRecord:
    """A release by id, whether staged, active or superseded."""

    with engine.connect() as connection:
        row = connection.execute(
            text(f"select {_RELEASE_COLUMNS} from corpus_releases where id = :id"),
            {"id": release_id},
        ).mappings().one_or_none()
    if row is None:
        raise ReleaseError("release_unknown", release_id)
    return _record(row)


def capture_release(engine: Engine) -> ReleaseRecord | None:
    """Read the active pointer once; the caller uses this record for the whole request."""

    with engine.connect() as connection:
        row = connection.execute(
            text(
                f"select {', '.join('r.' + c.strip() for c in _RELEASE_COLUMNS.split(','))}"
                " from active_release a join corpus_releases r on r.id = a.corpus_release_id"
                " where a.singleton_key = 'active'"
            )
        ).mappings().one_or_none()
    return None if row is None else _record(row)


def record_collection_build(
    engine: Engine, release_id: str, kind: str, details: Mapping[str, Any]
) -> None:
    """Attach one collection's build record to a staged release.

    Only a staged release accepts builds: once a release has served requests,
    rewriting what it claims to contain would make its validation meaningless.
    """

    with engine.begin() as connection:
        updated = connection.execute(
            text(
                "update corpus_releases set counts = counts || jsonb_build_object("
                " cast(:kind as text), cast(:details as jsonb))"
                " where id = :id and status = 'staged'"
            ),
            {"id": release_id, "kind": kind, "details": _json(details)},
        ).rowcount
    if updated != 1:
        raise ReleaseError("release_not_staged", release_id)


def drop_release(engine: Engine, release_id: str) -> ReleaseRecord:
    """Forget a release that is not serving. Its collections are the caller's to delete."""

    with engine.begin() as connection:
        active = connection.execute(
            text("select corpus_release_id from active_release where singleton_key = 'active'")
        ).scalar_one_or_none()
        if active == release_id:
            raise ReleaseError("release_active", release_id)
        row = connection.execute(
            text(f"delete from corpus_releases where id = :id returning {_RELEASE_COLUMNS}"),
            {"id": release_id},
        ).mappings().one_or_none()
    if row is None:
        raise ReleaseError("release_unknown", release_id)
    return _record(row)


def stage_release(
    engine: Engine,
    release_id: str,
    *,
    paper_collection: str,
    chunk_collection: str,
    model_revision: str,
    manifest_sha256: str,
    counts: Mapping[str, Any] | None = None,
) -> str:
    """Record a release as staged. Staging never changes what requests read."""

    if paper_collection == chunk_collection:
        raise ReleaseError("collections_must_differ", paper_collection)
    with engine.begin() as connection:
        existing = connection.execute(
            text(
                "select paper_collection, chunk_collection, model_revision, manifest_sha256"
                " from corpus_releases where id = :id"
            ),
            {"id": release_id},
        ).one_or_none()
        if existing is not None:
            # Re-staging is how a build resumes, but only into the same release:
            # another snapshot or model under this id would be a different index.
            if tuple(existing) != (
                paper_collection,
                chunk_collection,
                model_revision,
                manifest_sha256,
            ):
                raise ReleaseError("release_conflict", release_id)
            return release_id
        connection.execute(
            text(
                "insert into corpus_releases (id, manifest_sha256, paper_collection,"
                " chunk_collection, model_revision, status, counts, created_at) values"
                " (:id, :sha, :papers, :chunks, :revision, 'staged', cast(:counts as jsonb), now())"
                " on conflict (id) do nothing"
            ),
            {
                "id": release_id,
                "sha": manifest_sha256,
                "papers": paper_collection,
                "chunks": chunk_collection,
                "revision": model_revision,
                "counts": _json(counts or {}),
            },
        )
    return release_id


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, default=str)


def activate_release(engine: Engine, release_id: str, *, validator: Validator) -> str:
    """Point requests at a release, but only once its index pair validates.

    The pointer moves in one transaction, so a reader sees either the previous
    release or this one, never a half-switched pair.
    """

    with engine.begin() as connection:
        row = connection.execute(
            text(
                f"select {_RELEASE_COLUMNS} from corpus_releases where id = :id for update"
            ),
            {"id": release_id},
        ).mappings().one_or_none()
        if row is None:
            raise ReleaseError("release_unknown", release_id)
        if not validator(dict(row)):
            # The previous release keeps serving; nothing is rolled forward.
            raise ReleaseError("index_validation_failed", release_id)
        # The release it replaces stays intact for rollback; only its label moves.
        connection.execute(
            text(
                "update corpus_releases set status = 'superseded' where status = 'active'"
                " and id <> :id"
            ),
            {"id": release_id},
        )
        connection.execute(
            text("update corpus_releases set status = 'active' where id = :id"), {"id": release_id}
        )
        connection.execute(
            text(
                "insert into active_release (singleton_key, corpus_release_id, updated_at)"
                " values ('active', :id, now()) on conflict (singleton_key) do update set"
                " corpus_release_id = excluded.corpus_release_id, updated_at = excluded.updated_at"
            ),
            {"id": release_id},
        )
    return release_id
