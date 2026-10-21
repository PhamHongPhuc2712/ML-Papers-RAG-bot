"""Relabel bibliography chunks the parser stored as body text, in place.

Found 2026-10-20: ``pypdf-text-v2`` read Interspeech's numbered "6. References" heading
as a body section, and each numbered LNCS entry after ECCV's "References" ("2. Barron,
J.T., ...") as a new body section. Those chunks were stored with ``evidence_default``
true, so the first stage and the evidence step both searched them, and a query naming a
cited paper ranked the paper that cites it.

The parser is fixed for future parses. The stored corpus is repaired here with the same
rules, without re-parsing (the PDFs are deleted) or re-embedding (the text and vectors
are right; only the label is wrong): each matching chunk becomes ``references`` with
``evidence_default`` false, in the active release's chunk collection first and then in
PostgreSQL, so an interrupted run selects the same chunks again. The previous labels
are written to a JSON-lines file before anything changes.

Found the same day: 13 ACL, EMNLP and NAACL records are whole proceedings volumes, with no
abstract, whose text is the volume's front matter and table of contents. A table of
contents names hundreds of papers, so it matched many queries. Their chunks keep their
kind and leave search the same way (``evidence_default`` false).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text

from .parse import _APPENDIX, _CITATION, _NAMED

REFERENCES = "references"
_VOLUME = re.compile(r"^Proceedings of the\b")
# A cheap superset of the rule below, so PostgreSQL scans once and Python decides.
_CANDIDATES = text(
    """
    select c.id::text as chunk_id, c.kind, c.evidence_default, c.section_path,
           v.name as venue, p.publication_year as year
    from chunks c
    join paper_versions pv on pv.id = c.paper_version_id
    join papers p on p.id = pv.paper_id
    left join venues v on v.id = p.venue_id
    where c.kind in ('body', 'appendix')
      and (c.section_path ~* '^(references|bibliography)[.:]?$'
           or c.section_path ~ '^[[:alpha:]][^,]{0,60}, ?[A-Z]\\.')
    order by c.id
    """
)


_VOLUME_CHUNKS = text(
    """
    select c.id::text as chunk_id, c.kind, c.evidence_default, p.title, p.abstract,
           v.name as venue, p.publication_year as year
    from chunks c
    join paper_versions pv on pv.id = c.paper_version_id
    join papers p on p.id = pv.paper_id
    left join venues v on v.id = p.venue_id
    where c.evidence_default and p.title ~ '^\\s*Proceedings of the'
    order by c.id
    """
)


def misread_reference(section_name: str) -> bool:
    """Whether a stored section name is one the fixed parser would not have kept as body:
    a references heading, or a bibliography entry taken for a heading."""

    name = section_name.strip()
    if _APPENDIX.match(name):
        return False
    named = _NAMED.get(name.rstrip(".:").lower())
    if named is not None:
        return named[1] == REFERENCES
    return _CITATION.match(name) is not None


def proceedings_volume(title: str, abstract: str | None) -> bool:
    """Whether a record is a whole proceedings volume rather than one paper: a title naming
    the proceedings, and no abstract."""

    return _VOLUME.match(title.strip()) is not None and not (abstract or "").strip()


@dataclass(frozen=True)
class Relabel:
    chunk_id: str
    kind: str
    evidence_default: bool
    venue: str | None
    year: int | None


def find_misread_references(engine: Engine) -> list[Relabel]:
    with engine.connect() as connection:
        rows = connection.execute(_CANDIDATES).mappings().all()
    return [
        Relabel(
            chunk_id=row["chunk_id"],
            kind=row["kind"],
            evidence_default=bool(row["evidence_default"]),
            venue=row["venue"],
            year=row["year"],
        )
        for row in rows
        if misread_reference(row["section_path"])
    ]


def find_proceedings_volumes(engine: Engine) -> list[Relabel]:
    with engine.connect() as connection:
        rows = connection.execute(_VOLUME_CHUNKS).mappings().all()
    return [
        Relabel(
            chunk_id=row["chunk_id"],
            kind=row["kind"],
            evidence_default=bool(row["evidence_default"]),
            venue=row["venue"],
            year=row["year"],
        )
        for row in rows
        if proceedings_volume(row["title"], row["abstract"])
    ]


def summarize(chunks: Sequence[Relabel]) -> dict[str, int]:
    counts = Counter(f"{chunk.venue} {chunk.year}" for chunk in chunks)
    return dict(sorted(counts.items()))


def apply_relabel(
    engine: Engine,
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Relabel],
    *,
    undo_log: Path,
    kind: str | None = REFERENCES,
    batch: int = 1_000,
    progress: Callable[[int], None] | None = None,
) -> None:
    """Write the undo log, relabel the points, then the rows. Every chunk leaves evidence;
    its kind becomes ``kind`` unless that is None."""

    undo_log.parent.mkdir(parents=True, exist_ok=True)
    with undo_log.open("x", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "kind": chunk.kind,
                        "evidence_default": chunk.evidence_default,
                        "collection": collection,
                    }
                )
                + "\n"
            )
    ids = [chunk.chunk_id for chunk in chunks]
    points: list[models.ExtendedPointId] = list(ids)
    payload: dict[str, Any] = {"evidence_default": False}
    if kind is not None:
        payload["kind"] = kind
    update = "update chunks set evidence_default = false" + (
        ", kind = :kind" if kind is not None else ""
    )
    for start in range(0, len(ids), batch):
        selected = points[start : start + batch]
        client.set_payload(collection, payload=payload, points=selected, wait=True)
        if progress is not None:
            progress(min(start + batch, len(ids)))
    with engine.begin() as connection:
        for start in range(0, len(ids), batch):
            connection.execute(
                text(f"{update} where id::text = any(:ids)"),
                {"kind": kind, "ids": ids[start : start + batch]},
            )
