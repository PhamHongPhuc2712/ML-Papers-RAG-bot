"""Staged corpus releases and the transactional active-release switch (spec §4).

A release names the two Qdrant collections a request will read and the model
revision that built them. It is created ``staged`` and can only become active
once something has actually checked those collections — counts, dimensions and
a search canary, which P2.2 owns. The validator is injected here so this module
never guesses whether an index is serviceable, and so the invariant "no active
release without a validated index pair" holds even before that code exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sqlalchemy import Engine, text

Validator = Callable[[Mapping[str, Any]], bool]


class ReleaseError(RuntimeError):
    """A release cannot be staged or activated."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


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
    import json

    return json.dumps(dict(value), sort_keys=True, default=str)


def activate_release(engine: Engine, release_id: str, *, validator: Validator) -> str:
    """Point requests at a release, but only once its index pair validates.

    The pointer moves in one transaction, so a reader sees either the previous
    release or this one, never a half-switched pair.
    """

    with engine.begin() as connection:
        row = connection.execute(
            text(
                "select id, paper_collection, chunk_collection, model_revision, status, counts"
                " from corpus_releases where id = :id for update"
            ),
            {"id": release_id},
        ).mappings().one_or_none()
        if row is None:
            raise ReleaseError("release_unknown", release_id)
        if not validator(dict(row)):
            # The previous release keeps serving; nothing is rolled forward.
            raise ReleaseError("index_validation_failed", release_id)
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
