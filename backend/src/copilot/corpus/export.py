"""Immutable corpus snapshots: eligibility policy, Parquet shards, manifests.

An export is a directory of Parquet shards plus ``manifest.json``. The manifest
carries a checksum per shard and one over its own canonical body, so tampering
with either a shard or the manifest is detectable without the database that
produced it.

Two separate decisions live here and must not be conflated. A *local* snapshot
exists for restore and portability and contains everything the corpus holds. A
*public* export additionally passes :func:`public_export_rows`, which keeps only
corpus rows whose redistribution rights are positively established. Public
availability of a PDF establishes nothing by itself (spec §4), so rows default
to withheld and never the other way round.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Author,
    Chunk,
    Paper,
    PaperAuthor,
    PaperIdentifier,
    PaperVersion,
    SourceCheckpoint,
    Venue,
)
from ..db.session import session_scope

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Spec §5 spells the paper_versions enum 'eligible'; the P1.5 plan example spells
# the same decision 'allowed'. Accept both rather than silently withholding every
# row the real pipeline writes, and treat anything else -- including absent,
# None, or a different case -- as not publishable.
PUBLISHABLE_REDISTRIBUTION = frozenset({"allowed", "eligible"})

# Columns that must never reach an export artifact, public or local. Private
# user content is out of scope for corpus exports entirely (spec §4).
PRIVATE_FIELDS = frozenset(
    {
        "user_id",
        "owner_id",
        "upload_id",
        "blob_key",
        "email",
        "password_hash",
        "access_token",
        "request_id",
        "impression_id",
    }
)

ARTIFACT_KINDS = ("metadata", "fulltext", "edges")


class ExportError(Exception):
    """Base class for export and validation failures."""


class ExportSchemaError(ExportError):
    """Raised for an unknown schema version or a forbidden field."""


class ManifestValidationError(ExportError):
    """Raised when a manifest or one of its shards fails verification."""


def public_export_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only corpus rows whose redistribution rights are positively established."""

    return [
        dict(row)
        for row in rows
        if row.get("kind") == "corpus"
        and isinstance(row.get("redistribution"), str)
        and row["redistribution"] in PUBLISHABLE_REDISTRIBUTION
    ]


def reject_private_fields(rows: Iterable[Mapping[str, Any]]) -> None:
    """Fail before writing anything if a row carries a private field."""

    for row in rows:
        forbidden = sorted(PRIVATE_FIELDS.intersection(row))
        if forbidden:
            raise ExportSchemaError(f"private_field:{','.join(forbidden)}")


def require_schema_version(document: Mapping[str, Any]) -> int:
    """Refuse a manifest whose schema this build does not know."""

    version = document.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ExportSchemaError("schema_version_missing")
    if version != SCHEMA_VERSION:
        raise ExportSchemaError(f"schema_version_unsupported:{version}")
    return version


def coverage_percentage(count: int, denominator: int | None) -> float | None:
    """Percentage against an authoritative denominator, or None when unknown.

    An unknown denominator stays null and never becomes 100%: the count alone
    says nothing about what fraction of the venue-year it represents. A
    denominator smaller than the count is inconsistent rather than complete, so
    it also yields null instead of an over-100% figure.
    """

    if denominator is None or denominator <= 0 or count > denominator:
        return None
    return round(count / denominator * 100, 4)


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Checksum over the manifest body, excluding the checksum field itself."""

    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def metadata_rows(session: Session) -> list[dict[str, Any]]:
    """One row per canonical paper, with its aliases and eligibility."""

    statement = (
        select(Paper, Venue)
        .join(Venue, Venue.id == Paper.venue_id, isouter=True)
        .where(Paper.merged_into.is_(None))
        .order_by(Paper.id)
    )
    rows: list[dict[str, Any]] = []
    for paper, venue in session.execute(statement).all():
        identifiers = list(
            session.execute(
                select(PaperIdentifier)
                .where(PaperIdentifier.paper_id == paper.id)
                .order_by(PaperIdentifier.namespace, PaperIdentifier.value)
            ).scalars()
        )
        authors = [
            name
            for name in session.execute(
                select(Author.name)
                .join(PaperAuthor, PaperAuthor.author_id == Author.id)
                .where(PaperAuthor.paper_id == paper.id)
                .order_by(PaperAuthor.position)
            ).scalars()
        ]
        versions = list(
            session.execute(
                select(PaperVersion)
                .where(PaperVersion.paper_id == paper.id)
                .order_by(PaperVersion.created_at, PaperVersion.id)
            ).scalars()
        )
        # A work is publishable only if every one of its versions is; a single
        # restricted or unknown document withholds the whole record.
        eligibility = _combined_eligibility(version.redistribution for version in versions)
        rows.append(
            {
                "paper_id": str(paper.id),
                "kind": "corpus",
                "redistribution": eligibility,
                "title": paper.title,
                "abstract": paper.abstract,
                "publication_year": paper.publication_year,
                "venue": venue.name if venue is not None else None,
                "track": venue.track if venue is not None else None,
                "acceptance_type": paper.acceptance_type,
                "first_published_at": _iso(paper.first_published_at),
                "first_seen_at": _iso(paper.first_seen_at),
                "pdf_url": paper.pdf_url,
                "authors": json.dumps(authors, ensure_ascii=False),
                "identifiers": json.dumps(
                    [
                        {
                            "namespace": row.namespace,
                            "value": row.value,
                            "version": row.version,
                        }
                        for row in identifiers
                    ],
                    sort_keys=True,
                ),
                # Document versions travel inside the metadata shard so a restore
                # can rebuild a version that produced no chunks (an unparsed or
                # failed document) instead of silently dropping it.
                "versions": json.dumps(
                    [
                        {
                            "id": str(version.id),
                            "source": version.source,
                            "source_url": version.source_url,
                            "source_revision": version.source_revision,
                            "content_sha256": version.content_sha256,
                            "version": version.version,
                            "license_label": version.license_label,
                            "redistribution": version.redistribution,
                            "parser_version": version.parser_version,
                            "parse_status": version.parse_status,
                        }
                        for version in versions
                    ],
                    sort_keys=True,
                ),
                "source_revisions": json.dumps(
                    sorted({version.source_revision for version in versions})
                ),
                "has_abstract": bool(paper.abstract),
                "has_pdf_url": bool(paper.pdf_url),
                "has_fulltext": any(
                    version.parse_status == "parsed" for version in versions
                ),
            }
        )
    return rows


def _combined_eligibility(values: Iterable[str]) -> str:
    seen = set(values)
    if not seen:
        return "unknown"
    if seen <= PUBLISHABLE_REDISTRIBUTION:
        return "eligible"
    if "restricted" in seen:
        return "restricted"
    return "unknown"


def fulltext_rows(session: Session) -> list[dict[str, Any]]:
    """One row per chunk, carrying the provenance needed to trace it back."""

    statement = (
        select(Chunk, PaperVersion)
        .join(PaperVersion, PaperVersion.id == Chunk.paper_version_id)
        .order_by(PaperVersion.paper_id, Chunk.paper_version_id, Chunk.ordinal)
    )
    return [
        {
            "chunk_id": str(chunk.id),
            "paper_id": str(version.paper_id),
            "paper_version_id": str(version.id),
            "kind": "corpus",
            "redistribution": version.redistribution,
            "section_path": chunk.section_path,
            "section_ordinal": chunk.section_ordinal,
            "chunk_kind": chunk.kind,
            "ordinal": chunk.ordinal,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "evidence_default": chunk.evidence_default,
            "fragment": chunk.fragment,
            "parser_version": chunk.parser_version,
            "chunker_version": chunk.chunker_version,
            "content_sha256": version.content_sha256,
            "source_revision": version.source_revision,
        }
        for chunk, version in session.execute(statement).all()
    ]


def edges_rows(session: Session) -> list[dict[str, Any]]:
    """Citation edges. Empty until P6.3 ingests them; the shard still exists."""

    _ = session
    return []


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _write_shards(
    rows: Sequence[Mapping[str, Any]],
    kind: str,
    destination: Path,
    *,
    target_shard_bytes: int,
) -> list[dict[str, Any]]:
    """Write ``rows`` as one or more Parquet shards and describe each one."""

    reject_private_fields(rows)
    if not rows:
        table = pa.table({"placeholder": pa.array([], type=pa.string())})
        path = destination / f"{kind}-0000.parquet"
        pq.write_table(table, path, compression="zstd")
        return [_describe(path, kind, 0, destination)]

    shards: list[dict[str, Any]] = []
    batch: list[Mapping[str, Any]] = []
    batch_bytes = 0
    for row in rows:
        batch.append(row)
        batch_bytes += len(_canonical_bytes(row))
        if batch_bytes >= target_shard_bytes:
            shards.append(_flush(batch, kind, len(shards), destination))
            batch, batch_bytes = [], 0
    if batch:
        shards.append(_flush(batch, kind, len(shards), destination))
    return shards


def _flush(
    batch: Sequence[Mapping[str, Any]], kind: str, index: int, destination: Path
) -> dict[str, Any]:
    table = pa.Table.from_pylist([dict(row) for row in batch])
    path = destination / f"{kind}-{index:04d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return _describe(path, kind, len(batch), destination)


def _describe(path: Path, kind: str, rows: int, destination: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.relative_to(destination).as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def snapshot_coverage(session: Session) -> list[dict[str, Any]]:
    """Per venue-year counts with authoritative denominators only when known."""

    statement = (
        select(
            Venue.name,
            Venue.track,
            Paper.publication_year,
            func.count(Paper.id),
            func.count(Paper.abstract),
            func.count(Paper.pdf_url),
        )
        .join(Venue, Venue.id == Paper.venue_id, isouter=True)
        .where(Paper.merged_into.is_(None))
        .group_by(Venue.name, Venue.track, Paper.publication_year)
        .order_by(Venue.name, Paper.publication_year)
    )
    checkpoints = {
        checkpoint.partition: checkpoint
        for checkpoint in session.execute(select(SourceCheckpoint)).scalars()
    }
    rows: list[dict[str, Any]] = []
    for venue, track, year, papers, abstracts, pdf_urls in session.execute(statement).all():
        partition = f"{venue}:{year}:{track}"
        checkpoint = checkpoints.get(partition)
        fulltext = session.execute(
            select(func.count(func.distinct(PaperVersion.paper_id)))
            .join(Paper, Paper.id == PaperVersion.paper_id)
            .where(PaperVersion.parse_status == "parsed", Paper.publication_year == year)
        ).scalar_one()
        rows.append(
            {
                "venue": venue,
                "track": track,
                "year": year,
                # No authoritative proceedings count has been confirmed for this
                # venue-year, so the denominator stays null and every percentage
                # below stays null with it.
                "expected_count": None,
                "ingested_count": papers,
                "abstract_count": abstracts,
                "abstract_coverage_percent": coverage_percentage(abstracts, None),
                "fulltext_count": fulltext,
                "fulltext_coverage_percent": coverage_percentage(fulltext, None),
                "missing_abstract_count": papers - abstracts,
                "missing_pdf_url_count": papers - pdf_urls,
                "missing_fulltext_count": papers - fulltext,
                "source": checkpoint.source if checkpoint is not None else None,
                "source_cursor": checkpoint.cursor if checkpoint is not None else None,
                "as_of": _iso(checkpoint.updated_at) if checkpoint is not None else None,
            }
        )
    return rows


def export_snapshot(
    run_id: str,
    destination: Path,
    *,
    engine: Engine,
    public_only: bool = False,
    target_shard_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    """Write an immutable snapshot and return its manifest.

    Refuses to overwrite an existing destination, so prior snapshots survive.
    """

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ExportError(f"destination_not_empty:{destination}")
    destination.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(UTC)
    try:
        with session_scope(engine) as session:
            builders = {
                "metadata": metadata_rows,
                "fulltext": fulltext_rows,
                "edges": edges_rows,
            }
            artifacts: list[dict[str, Any]] = []
            withheld: dict[str, int] = {}
            for kind in ARTIFACT_KINDS:
                rows: Sequence[Mapping[str, Any]] = builders[kind](session)
                total = len(rows)
                if public_only:
                    rows = public_export_rows(rows)
                    withheld[kind] = total - len(rows)
                artifacts.extend(
                    _write_shards(
                        rows, kind, destination, target_shard_bytes=target_shard_bytes
                    )
                )
            coverage = snapshot_coverage(session)
            parser_versions = sorted(
                set(session.execute(select(PaperVersion.parser_version)).scalars())
            )
            chunker_versions = sorted(set(session.execute(select(Chunk.chunker_version)).scalars()))
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "public_only": public_only,
        "parser_versions": parser_versions,
        "chunker_versions": chunker_versions,
        "counts": {artifact["kind"]: 0 for artifact in artifacts},
        "withheld_rows": withheld,
        "coverage": coverage,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    for artifact in artifacts:
        manifest["counts"][artifact["kind"]] += artifact["rows"]
    manifest["manifest_sha256"] = manifest_checksum(manifest)

    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def validate_manifest(path: Path) -> dict[str, Any]:
    """Verify schema, manifest checksum and every shard checksum and size."""

    path = Path(path)
    manifest_path = path / MANIFEST_NAME if path.is_dir() else path
    if not manifest_path.is_file():
        raise ManifestValidationError(f"manifest_missing:{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestValidationError("manifest_not_json") from error
    if not isinstance(manifest, dict):
        raise ManifestValidationError("manifest_not_an_object")

    # Schema is checked before anything is read or written, so an unknown
    # snapshot layout cannot be partially consumed.
    require_schema_version(manifest)

    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str) or recorded != manifest_checksum(manifest):
        raise ManifestValidationError("manifest_checksum_mismatch")

    root = manifest_path.parent
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestValidationError("artifacts_missing")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ManifestValidationError("artifact_not_an_object")
        shard = root / str(artifact.get("path", ""))
        if not shard.is_file():
            raise ManifestValidationError(f"shard_missing:{artifact.get('path')}")
        if shard.stat().st_size != artifact.get("bytes"):
            raise ManifestValidationError(f"shard_size_mismatch:{artifact.get('path')}")
        if _sha256_file(shard) != artifact.get("sha256"):
            raise ManifestValidationError(f"shard_checksum_mismatch:{artifact.get('path')}")
    return manifest


def read_shards(path: Path, kind: str) -> list[dict[str, Any]]:
    """Read every shard of one kind from a validated snapshot directory."""

    manifest = validate_manifest(path)
    root = Path(path)
    rows: list[dict[str, Any]] = []
    for artifact in manifest["artifacts"]:
        if artifact["kind"] != kind or artifact["rows"] == 0:
            continue
        rows.extend(pq.read_table(root / artifact["path"]).to_pylist())
    return rows


def restore_snapshot(path: Path, engine: Engine) -> dict[str, int]:
    """Rebuild a corpus from a validated snapshot into an empty database.

    Refuses a database that already holds papers: restore proves portability
    into a clean namespace, and merging two corpora is not a restore. Returns
    the row counts it created so a caller can compare them with the manifest.
    """

    manifest = validate_manifest(path)
    metadata = read_shards(path, "metadata")
    fulltext = read_shards(path, "fulltext")

    counts = {"papers": 0, "identifiers": 0, "versions": 0, "chunks": 0, "authors": 0}
    with session_scope(engine) as session:
        if session.execute(select(func.count(Paper.id))).scalar_one():
            raise ExportError("restore_target_not_empty")

        venues: dict[tuple[str, str], UUID] = {}
        authors: dict[str, UUID] = {}
        for row in metadata:
            venue_id = None
            venue_name = row.get("venue")
            if venue_name:
                key = (str(venue_name), str(row.get("track") or ""))
                if key not in venues:
                    venue = Venue(id=uuid4(), name=key[0], track=key[1])
                    session.add(venue)
                    session.flush()
                    venues[key] = venue.id
                venue_id = venues[key]

            paper_id = UUID(str(row["paper_id"]))
            session.add(
                Paper(
                    id=paper_id,
                    title=str(row["title"]),
                    abstract=row.get("abstract"),
                    publication_year=row.get("publication_year"),
                    venue_id=venue_id,
                    first_published_at=_parse_iso(row.get("first_published_at")),
                    first_seen_at=_parse_iso(row.get("first_seen_at")) or datetime.now(UTC),
                    pdf_url=row.get("pdf_url"),
                    acceptance_type=str(row.get("acceptance_type") or "unknown"),
                    metadata_status="active",
                )
            )
            session.flush()
            counts["papers"] += 1

            for position, name in enumerate(json.loads(row.get("authors") or "[]")):
                normalized = str(name).casefold()
                if normalized not in authors:
                    author = Author(id=uuid4(), name=str(name), normalized_name=normalized)
                    session.add(author)
                    session.flush()
                    authors[normalized] = author.id
                    counts["authors"] += 1
                session.add(
                    PaperAuthor(
                        paper_id=paper_id, position=position, author_id=authors[normalized]
                    )
                )

            for identifier in json.loads(row.get("identifiers") or "[]"):
                session.add(
                    PaperIdentifier(
                        id=uuid4(),
                        paper_id=paper_id,
                        namespace=str(identifier["namespace"]),
                        value=str(identifier["value"]),
                        version=identifier.get("version"),
                    )
                )
                counts["identifiers"] += 1

            for version in json.loads(row.get("versions") or "[]"):
                session.add(
                    PaperVersion(
                        id=UUID(str(version["id"])),
                        paper_id=paper_id,
                        source=str(version["source"]),
                        source_url=str(version["source_url"]),
                        source_revision=str(version["source_revision"]),
                        content_sha256=str(version["content_sha256"]),
                        version=version.get("version"),
                        license_label=version.get("license_label"),
                        redistribution=str(version.get("redistribution") or "unknown"),
                        parser_version=str(version.get("parser_version") or "unparsed"),
                        parse_status=str(version.get("parse_status") or "pending"),
                    )
                )
                counts["versions"] += 1
            session.flush()

        for row in fulltext:
            session.add(
                Chunk(
                    id=UUID(str(row["chunk_id"])),
                    paper_version_id=UUID(str(row["paper_version_id"])),
                    section_path=str(row["section_path"]),
                    section_ordinal=int(row["section_ordinal"]),
                    kind=str(row["chunk_kind"]),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    ordinal=int(row["ordinal"]),
                    text=str(row["text"]),
                    token_count=int(row["token_count"]),
                    parser_version=str(row["parser_version"]),
                    chunker_version=str(row["chunker_version"]),
                    evidence_default=bool(row["evidence_default"]),
                    fragment=row.get("fragment"),
                )
            )
            counts["chunks"] += 1
        session.flush()

    counts["manifest_papers"] = int(manifest["counts"].get("metadata", 0))
    counts["manifest_chunks"] = int(manifest["counts"].get("fulltext", 0))
    return counts


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
