"""Immutable corpus snapshots and the coverage they can honestly claim.

Two rules shape this module. Public availability of a PDF does not establish a
right to redistribute it, so full text leaves only when a record says
``allowed``; everything else — restricted, unknown, and every private upload —
stays on this machine. And an unknown denominator is reported as unknown: a
venue-year whose published total we cannot verify gets a null coverage
percentage rather than a flattering one (spec §4).

A snapshot is a directory of Parquet shards plus a manifest naming each shard's
row count, byte size and sha256. Validation recomputes those checksums before
anything reads a shard, so a tampered or truncated snapshot fails before it can
be restored over live data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, text

SCHEMA_VERSION = 1
# Only this value releases text. "unknown" is not "probably fine".
REDISTRIBUTABLE = "allowed"
PRIVATE_KINDS = frozenset({"upload"})
_READ_BLOCK = 1 << 20


class ExportError(RuntimeError):
    """A snapshot cannot be written, validated or restored."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def public_export_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only corpus rows whose redistribution is explicitly allowed."""

    return [
        dict(row)
        for row in rows
        if row.get("kind") not in PRIVATE_KINDS and row.get("redistribution") == REDISTRIBUTABLE
    ]


def coverage_rows(
    papers: Sequence[Mapping[str, Any]],
    expected: Mapping[tuple[str, int], int],
) -> list[dict[str, Any]]:
    """Per venue-year counts, with a null percentage when the total is unknown."""

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for paper in papers:
        key = (str(paper.get("venue") or ""), int(paper.get("year") or 0))
        row = grouped.setdefault(
            key,
            {
                "venue": key[0],
                "year": key[1],
                "papers": 0,
                "with_abstract": 0,
                "with_fulltext": 0,
                "failed_parse": 0,
            },
        )
        row["papers"] += 1
        if paper.get("abstract"):
            row["with_abstract"] += 1
        status = str(paper.get("parse_status") or "")
        if status == "parsed":
            row["with_fulltext"] += 1
        elif status not in ("", "pending"):
            row["failed_parse"] += 1
    result: list[dict[str, Any]] = []
    for key, row in sorted(grouped.items()):
        total = expected.get(key)
        row["expected"] = total
        row["coverage_pct"] = round(100 * row["papers"] / total, 1) if total else None
        result.append(row)
    return result


def _checksum(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Check schema and every shard checksum before a caller reads the snapshot."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ExportError("manifest_missing", str(manifest_path))
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    # Checked before any shard is opened, so an unsupported snapshot cannot
    # half-apply itself.
    if version != SCHEMA_VERSION:
        raise ExportError("schema_version_unsupported", str(version))
    for shard in manifest.get("shards", []):
        shard_path = manifest_path.parent / str(shard["path"])
        if not shard_path.is_file():
            raise ExportError("shard_missing", str(shard_path))
        digest, size = _checksum(shard_path)
        if digest != shard.get("sha256"):
            raise ExportError("checksum_mismatch", str(shard_path))
        if int(shard.get("bytes", size)) != size:
            raise ExportError("size_mismatch", str(shard_path))
    return manifest


def _write_shard(destination: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else ["id"]
    table = pa.table({column: [row.get(column) for row in rows] for column in columns})
    path = destination / name
    pq.write_table(table, path)
    digest, size = _checksum(path)
    return {"path": name, "rows": len(rows), "bytes": size, "sha256": digest}


_PAPERS_SQL = """
select p.id::text as paper_id,
       p.title,
       p.abstract,
       p.publication_year as year,
       v.name as venue,
       v.track as track,
       p.acceptance_type,
       p.pdf_url,
       p.first_seen_at,
       pv.source,
       pv.source_revision,
       pv.content_sha256,
       pv.parse_status,
       pv.redistribution,
       pv.parser_version,
       coalesce(a.names, '') as authors,
       coalesce(i.ids, '') as identifiers
from papers p
left join venues v on v.id = p.venue_id
left join paper_versions pv on pv.paper_id = p.id
left join (
    select pa.paper_id, string_agg(au.name, '; ' order by pa.position) as names
    from paper_authors pa join authors au on au.id = pa.author_id group by pa.paper_id
) a on a.paper_id = p.id
left join (
    select namespace_values.paper_id,
           string_agg(namespace_values.namespace || ':' || namespace_values.value, ' ') as ids
    from paper_identifiers namespace_values group by namespace_values.paper_id
) i on i.paper_id = p.id
where p.merged_into is null
order by p.id
"""

_FULLTEXT_SQL = """
select c.id::text as chunk_id,
       pv.paper_id::text as paper_id,
       c.section_path,
       c.kind,
       c.page_start,
       c.page_end,
       c.ordinal,
       c.text,
       c.token_count,
       c.parser_version,
       c.chunker_version,
       c.evidence_default,
       pv.redistribution
from chunks c join paper_versions pv on pv.id = c.paper_version_id
order by c.paper_version_id, c.ordinal
"""


def export_snapshot(
    run_id: str,
    destination: str | Path,
    *,
    engine: Engine,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write an immutable metadata + eligible-full-text snapshot and its manifest."""

    target = Path(destination)
    if (target / "manifest.json").exists():
        # Snapshots are immutable; a new export gets a new directory.
        raise ExportError("snapshot_exists", str(target))
    target.mkdir(parents=True, exist_ok=True)

    with engine.connect() as connection:
        papers = [dict(row) for row in connection.execute(text(_PAPERS_SQL)).mappings()]
        chunks = [dict(row) for row in connection.execute(text(_FULLTEXT_SQL)).mappings()]

    # Bibliographic metadata is factual; the redistribution filter governs text.
    fulltext = public_export_rows({**row, "kind": "corpus"} for row in chunks)
    for row in fulltext:
        row.pop("kind", None)

    shards = [
        _write_shard(target, "papers.parquet", papers),
        _write_shard(target, "fulltext.parquet", fulltext),
    ]
    versions = sorted({str(row.get("parser_version") or "") for row in papers} - {""})
    chunkers = sorted({str(row.get("chunker_version") or "") for row in chunks} - {""})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "shards": shards,
        "counts": {
            "papers": len(papers),
            "chunks_total": len(chunks),
            "chunks_exported": len(fulltext),
            "withheld_chunks": len(chunks) - len(fulltext),
        },
        "versions": {"parser": versions, "chunker": chunkers},
        "rights": {
            "exported_when": REDISTRIBUTABLE,
            "withheld": sorted(
                {str(row.get("redistribution") or "unknown") for row in chunks}
                - {REDISTRIBUTABLE}
            ),
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return manifest


def restore_snapshot(manifest_path: str | Path, *, engine: Engine) -> dict[str, int]:
    """Load a validated snapshot into an empty database and report what landed."""

    # Validate first: a tampered or unsupported snapshot must never touch the
    # target database, even a nominally empty one.
    validate_manifest(manifest_path)
    directory = Path(manifest_path).parent
    with engine.connect() as connection:
        existing = connection.execute(text("select count(*) from papers")).scalar_one()
    if existing:
        raise ExportError("target_not_empty", f"{existing} papers")

    papers = pq.read_table(directory / "papers.parquet").to_pylist()
    venues: dict[tuple[str, str], str] = {}
    with engine.begin() as connection:
        for row in papers:
            key = (str(row.get("venue") or ""), str(row.get("track") or ""))
            if key not in venues and key[0]:
                venues[key] = str(
                    connection.execute(
                        text(
                            "insert into venues (id, name, track) values (gen_random_uuid(),"
                            " :name, :track) on conflict (name, track) do update set name ="
                            " excluded.name returning id"
                        ),
                        {"name": key[0], "track": key[1]},
                    ).scalar_one()
                )
            connection.execute(
                text(
                    "insert into papers (id, title, abstract, publication_year, venue_id,"
                    " first_seen_at, pdf_url, acceptance_type, metadata_status) values"
                    " (:id, :title, :abstract, :year, :venue_id, coalesce(:first_seen_at, now()),"
                    " :pdf_url, :acceptance_type, 'active')"
                ),
                {
                    "id": row["paper_id"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "year": row["year"],
                    "venue_id": venues.get(key),
                    "first_seen_at": row.get("first_seen_at"),
                    "pdf_url": row.get("pdf_url"),
                    "acceptance_type": row.get("acceptance_type") or "unknown",
                },
            )
            for token in str(row.get("identifiers") or "").split():
                namespace, _, value = token.partition(":")
                if namespace and value:
                    connection.execute(
                        text(
                            "insert into paper_identifiers (id, paper_id, namespace, value)"
                            " values (gen_random_uuid(), :paper_id, :namespace, :value)"
                            " on conflict (namespace, value) do nothing"
                        ),
                        {"paper_id": row["paper_id"], "namespace": namespace, "value": value},
                    )
    with engine.connect() as connection:
        return {
            "papers": connection.execute(text("select count(*) from papers")).scalar_one(),
            "identifiers": connection.execute(
                text("select count(*) from paper_identifiers")
            ).scalar_one(),
        }
