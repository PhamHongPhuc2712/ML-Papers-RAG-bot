"""Immutable corpus snapshots and the coverage they can honestly claim.

Two rules shape this module. Public availability of a PDF does not establish a
right to redistribute it, so full text leaves only when a record says
``allowed``; everything else — restricted, unknown, and every private upload —
stays on this machine. And an unknown denominator is reported as unknown: a
venue-year whose published total we cannot verify gets a null coverage
percentage rather than a flattering one (spec §4).

A snapshot is a directory of Parquet shards plus a manifest naming each shard's
table, row count, byte size and sha256. Validation recomputes those checksums
before anything reads a shard, so a tampered or truncated snapshot fails before
it can be restored over live data.

The corpus is 3.4 M chunks, several gigabytes of text, so nothing here holds a
table in memory: rows stream from a server-side cursor into shards that roll
over at a bounded size. Withheld text never leaves the database at all — what
the snapshot keeps of a withheld chunk is its identity and the sha256 of its
text, which is what lets an index built from the database later prove it
indexed exactly the chunks this snapshot names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, text

from ..search.lexical import paper_text

SCHEMA_VERSION = 2
# Version 1 wrote one unsharded file per table; its file stem names the table.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, SCHEMA_VERSION})
# Only this value releases text. "unknown" is not "probably fine".
REDISTRIBUTABLE = "allowed"
PRIVATE_KINDS = frozenset({"upload"})
_READ_BLOCK = 1 << 20
# Spec §4 targets 128-512 MiB shards. The rollover is measured on the Arrow
# batches going in, so compressed files land below this.
SHARD_BYTES = 512 * 1024**2
BATCH_ROWS = 20_000
DIGEST_RULE = "sha256 over '<id>\\t<text_sha256>\\n' per row, in id order"

PAPERS_SCHEMA = pa.schema(
    [
        ("paper_id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("year", pa.int32()),
        ("venue", pa.string()),
        ("track", pa.string()),
        ("acceptance_type", pa.string()),
        ("pdf_url", pa.string()),
        ("first_seen_at", pa.timestamp("us", tz="UTC")),
        ("source", pa.string()),
        ("source_revision", pa.string()),
        ("content_sha256", pa.string()),
        ("parse_status", pa.string()),
        ("redistribution", pa.string()),
        ("parser_version", pa.string()),
        ("authors", pa.string()),
        ("identifiers", pa.string()),
        # sha256 of the canonical lexical document, title + "\n" + abstract.
        ("text_sha256", pa.string()),
    ]
)
CHUNKS_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("paper_id", pa.string()),
        ("paper_version_id", pa.string()),
        ("section_ordinal", pa.int32()),
        ("ordinal", pa.int32()),
        ("kind", pa.string()),
        ("token_count", pa.int32()),
        ("parser_version", pa.string()),
        ("chunker_version", pa.string()),
        ("evidence_default", pa.bool_()),
        ("redistribution", pa.string()),
        ("text_sha256", pa.string()),
    ]
)
FULLTEXT_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("paper_id", pa.string()),
        ("section_path", pa.string()),
        ("kind", pa.string()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("ordinal", pa.int32()),
        ("text", pa.string()),
        ("token_count", pa.int32()),
        ("parser_version", pa.string()),
        ("chunker_version", pa.string()),
        ("evidence_default", pa.bool_()),
        ("redistribution", pa.string()),
    ]
)


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
    if version not in SUPPORTED_SCHEMA_VERSIONS:
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


def shard_table(shard: Mapping[str, Any]) -> str:
    """The table a shard belongs to; version 1 manifests name it by file stem."""

    table = shard.get("table")
    return str(table) if table else Path(str(shard["path"])).stem


def shard_paths(manifest: Mapping[str, Any], table: str) -> list[str]:
    """A table's shard paths, relative to the manifest, in the order written."""

    return [
        str(shard["path"]) for shard in manifest.get("shards", []) if shard_table(shard) == table
    ]


def record_digest(rows: Iterable[tuple[str, str]]) -> str:
    """Digest of (id, text_sha256) pairs in order; see ``DIGEST_RULE``."""

    digest = hashlib.sha256()
    for identifier, text_sha256 in rows:
        digest.update(f"{identifier}\t{text_sha256}\n".encode())
    return digest.hexdigest()


class _ShardWriter:
    """Stream batches into one table's shards, rolling over at a size bound."""

    def __init__(self, destination: Path, table: str, schema: pa.Schema, shard_bytes: int):
        self._destination = destination
        self._table = table
        self._schema = schema
        self._shard_bytes = shard_bytes
        self._writer: pq.ParquetWriter | None = None
        self._path: Path | None = None
        self._rows = 0
        self._pending = 0
        self.shards: list[dict[str, Any]] = []

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        batch = pa.RecordBatch.from_pylist(
            [{name: row.get(name) for name in self._schema.names} for row in rows],
            schema=self._schema,
        )
        if self._writer is None:
            self._open()
        assert self._writer is not None
        self._writer.write_batch(batch)
        self._rows += len(rows)
        self._pending += batch.nbytes
        if self._pending >= self._shard_bytes:
            self._close()

    def close(self) -> list[dict[str, Any]]:
        # An empty table still gets a shard, so its schema is part of the record.
        if self._writer is None and not self.shards:
            self._open()
        self._close()
        return self.shards

    def _open(self) -> None:
        name = f"{self._table}-{len(self.shards):05d}.parquet"
        self._path = self._destination / name
        self._writer = pq.ParquetWriter(self._path, self._schema)
        self._rows = 0
        self._pending = 0

    def _close(self) -> None:
        if self._writer is None or self._path is None:
            return
        self._writer.close()
        digest, size = _checksum(self._path)
        self.shards.append(
            {
                "table": self._table,
                "path": self._path.name,
                "rows": self._rows,
                "bytes": size,
                "sha256": digest,
            }
        )
        self._writer = None
        self._path = None


# One version represents a paper: a parsed one if it has any, then the newest.
# Joining every version would repeat a paper once per version, and the paper
# index keys its points by paper.
_CHOSEN_VERSIONS = """
select distinct on (paper_id) id, paper_id, source, source_revision, content_sha256,
       parse_status, redistribution, parser_version
from paper_versions
order by paper_id, (parse_status = 'parsed') desc, created_at desc, id
"""

_PAPERS_SQL = f"""
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
left join ({_CHOSEN_VERSIONS}) pv on pv.paper_id = p.id
left join (
    select pa.paper_id, string_agg(au.name, '; ' order by pa.position) as names
    from paper_authors pa join authors au on au.id = pa.author_id group by pa.paper_id
) a on a.paper_id = p.id
left join (
    select namespace_values.paper_id,
           string_agg(namespace_values.namespace || ':' || namespace_values.value, ' '
                      order by namespace_values.namespace, namespace_values.value) as ids
    from paper_identifiers namespace_values group by namespace_values.paper_id
) i on i.paper_id = p.id
where p.merged_into is null
order by p.id
"""

# The chunk set a snapshot names. The chunk index reads exactly this set back
# from the database, which is what makes its digest comparable to this one.
EXPORTED_CHUNKS = f"""
from chunks c
join ({_CHOSEN_VERSIONS}) pv on pv.id = c.paper_version_id
join papers p on p.id = pv.paper_id and p.merged_into is null
"""

# Identity only: the text is hashed inside PostgreSQL and never fetched.
_CHUNKS_SQL = f"""
select c.id::text as chunk_id,
       pv.paper_id::text as paper_id,
       c.paper_version_id::text as paper_version_id,
       c.section_ordinal,
       c.ordinal,
       c.kind,
       c.token_count,
       c.parser_version,
       c.chunker_version,
       c.evidence_default,
       pv.redistribution,
       encode(sha256(convert_to(c.text, 'UTF8')), 'hex') as text_sha256
{EXPORTED_CHUNKS}
order by c.id
"""

# The rights filter runs in SQL, so withheld text is never read into this process.
_FULLTEXT_SQL = f"""
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
{EXPORTED_CHUNKS}
where pv.redistribution = :allowed
order by c.paper_version_id, c.ordinal
"""


def _stream(
    engine: Engine, sql: str, params: Mapping[str, Any], batch_rows: int
) -> Iterator[list[dict[str, Any]]]:
    """Yield result batches from a server-side cursor, never the whole result."""

    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True, yield_per=batch_rows).execute(
            text(sql), dict(params)
        )
        for partition in result.mappings().partitions(batch_rows):
            yield [dict(row) for row in partition]


def export_snapshot(
    run_id: str,
    destination: str | Path,
    *,
    engine: Engine,
    now: datetime | None = None,
    batch_rows: int = BATCH_ROWS,
    shard_bytes: int = SHARD_BYTES,
) -> dict[str, Any]:
    """Write an immutable metadata + eligible-full-text snapshot and its manifest."""

    target = Path(destination)
    if (target / "manifest.json").exists():
        # Snapshots are immutable; a new export gets a new directory.
        raise ExportError("snapshot_exists", str(target))
    target.mkdir(parents=True, exist_ok=True)

    # Bibliographic metadata is factual; the redistribution filter governs text.
    papers = _ShardWriter(target, "papers", PAPERS_SCHEMA, shard_bytes)
    papers_digest = hashlib.sha256()
    parser_versions: set[str] = set()
    paper_count = 0
    for batch in _stream(engine, _PAPERS_SQL, {}, batch_rows):
        for row in batch:
            document = paper_text(str(row["title"]), row.get("abstract"))
            row["text_sha256"] = hashlib.sha256(document.encode("utf-8")).hexdigest()
            papers_digest.update(f"{row['paper_id']}\t{row['text_sha256']}\n".encode())
            parser_versions.add(str(row.get("parser_version") or ""))
        papers.write(batch)
        paper_count += len(batch)

    chunks = _ShardWriter(target, "chunks", CHUNKS_SCHEMA, shard_bytes)
    chunks_digest = hashlib.sha256()
    chunker_versions: set[str] = set()
    withheld: dict[str, int] = {}
    chunk_count = 0
    for batch in _stream(engine, _CHUNKS_SQL, {}, batch_rows):
        for row in batch:
            chunks_digest.update(f"{row['chunk_id']}\t{row['text_sha256']}\n".encode())
            chunker_versions.add(str(row.get("chunker_version") or ""))
            rights = str(row.get("redistribution") or "unknown")
            if rights != REDISTRIBUTABLE:
                withheld[rights] = withheld.get(rights, 0) + 1
        chunks.write(batch)
        chunk_count += len(batch)

    fulltext = _ShardWriter(target, "fulltext", FULLTEXT_SCHEMA, shard_bytes)
    exported = 0
    for batch in _stream(engine, _FULLTEXT_SQL, {"allowed": REDISTRIBUTABLE}, batch_rows):
        # The SQL filter is the bound; the policy function is the rule, applied again.
        released = public_export_rows({**row, "kind": "corpus"} for row in batch)
        fulltext.write(released)
        exported += len(released)

    shards = [*papers.close(), *chunks.close(), *fulltext.close()]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "shards": shards,
        "counts": {
            "papers": paper_count,
            "chunks_total": chunk_count,
            "chunks_exported": exported,
            "withheld_chunks": chunk_count - exported,
        },
        "digests": {"papers": papers_digest.hexdigest(), "chunks": chunks_digest.hexdigest()},
        "digest_rule": DIGEST_RULE,
        "versions": {
            "parser": sorted(parser_versions - {""}),
            "chunker": sorted(chunker_versions - {""}),
        },
        "rights": {"exported_when": REDISTRIBUTABLE, "withheld": sorted(withheld)},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return manifest


def _table_rows(
    directory: Path, manifest: Mapping[str, Any], table: str
) -> Iterator[dict[str, Any]]:
    for shard in shard_paths(manifest, table):
        for batch in pq.ParquetFile(directory / shard).iter_batches(batch_size=BATCH_ROWS):
            yield from batch.to_pylist()


def restore_snapshot(manifest_path: str | Path, *, engine: Engine) -> dict[str, int]:
    """Load a validated snapshot into an empty database and report what landed."""

    # Validate first: a tampered or unsupported snapshot must never touch the
    # target database, even a nominally empty one.
    manifest = validate_manifest(manifest_path)
    directory = Path(manifest_path).parent
    with engine.connect() as connection:
        existing = connection.execute(text("select count(*) from papers")).scalar_one()
    if existing:
        raise ExportError("target_not_empty", f"{existing} papers")

    venues: dict[tuple[str, str], str] = {}
    with engine.begin() as connection:
        for row in _table_rows(directory, manifest, "papers"):
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
