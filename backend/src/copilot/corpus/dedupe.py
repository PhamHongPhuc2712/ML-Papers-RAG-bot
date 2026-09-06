"""Transactional paper identity resolution and source provenance."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from ..db.models import (
    Author,
    FieldProvenance,
    IdentityConflict,
    Paper,
    PaperAuthor,
    PaperIdentifier,
    PaperRedirect,
    PaperVersion,
    QuarantineRecord,
    SourceRecord,
    Venue,
)
from .normalize import normalize_arxiv, normalize_author_name, normalize_doi, normalize_title

_ARTIFACT_LOCK = threading.Lock()


class IdentityResolutionError(ValueError):
    """Base class for reviewable identity-resolution outcomes."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RecordValidationError(IdentityResolutionError):
    """Raised when a source record has a malformed required field."""


class QuarantineError(IdentityResolutionError):
    """Raised after an invalid supplied identifier is retained for review."""

    def __init__(self, code: str, quarantine_id: UUID | None = None) -> None:
        self.quarantine_id = quarantine_id
        super().__init__(code)


QuarantinedRecordError = QuarantineError


class IdentityConflictError(IdentityResolutionError):
    """Raised after a durable identity conflict has been flushed."""

    def __init__(
        self,
        code: str,
        conflict_id: UUID | None = None,
    ) -> None:
        self.conflict_id = conflict_id
        super().__init__(code)


class MergeConflictError(IdentityConflictError):
    """Raised when a manual merge cannot preserve unique identity history."""


class MergeValidationError(IdentityResolutionError):
    """Raised when a manual merge request is invalid or would create a cycle."""


@dataclass(frozen=True)
class _ValidatedRecord:
    raw: dict[str, Any]
    source: str
    source_revision: str
    retrieved_at: datetime
    title: str
    title_key: str
    authors: tuple[str, ...]
    author_keys: frozenset[str]
    venue_name: str | None
    venue_track: str
    publication_year: int | None
    abstract: str | None
    source_url: str | None
    acceptance_decision: str | None
    first_published_at: datetime | None
    pdf_url: str | None


@dataclass(frozen=True)
class _ExternalIdentifier:
    namespace: str
    value: str
    version: str | None = None


def _string(value: object, code: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise RecordValidationError(code)
        return None
    if not isinstance(value, str):
        raise RecordValidationError(code)
    stripped = value.strip()
    if required and not stripped:
        raise RecordValidationError(code)
    return stripped or None


def _parse_datetime(value: object, code: str, *, default_now: bool = False) -> datetime:
    if value is None:
        if default_now:
            return datetime.now(UTC)
        raise RecordValidationError(code)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RecordValidationError(code) from exc
    else:
        raise RecordValidationError(code)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_optional_datetime(value: object, code: str) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value, code)


def _author_name(value: object) -> str:
    if isinstance(value, str):
        name = value
    elif isinstance(value, Mapping):
        name_value = value.get("name")
        if not isinstance(name_value, str):
            raise RecordValidationError("author_name_required")
        name = name_value
    else:
        raise RecordValidationError("author_name_required")
    if not name.strip():
        raise RecordValidationError("author_name_required")
    return name.strip()


def _parse_venue(value: object) -> tuple[str | None, str]:
    if value is None:
        return None, ""
    if isinstance(value, str):
        name = value.strip()
        return (name or None), ""
    if isinstance(value, Mapping):
        venue_name = _string(
            value.get("name") or value.get("venue"), "venue_name_invalid"
        )
        track = _string(value.get("track"), "venue_track_invalid") or ""
        return venue_name, track
    raise RecordValidationError("venue_invalid")


def _validate_record(record: object) -> _ValidatedRecord:
    if not isinstance(record, Mapping):
        raise RecordValidationError("record_required")
    raw = {str(key): value for key, value in record.items()}
    source = _string(raw.get("source"), "source_required", required=True)
    title_value = raw.get("title")
    if not isinstance(title_value, str) or not title_value.strip():
        raise RecordValidationError("title_required")
    title = title_value
    assert source is not None
    assert title is not None

    authors_value = raw.get("authors", [])
    if authors_value is None:
        authors_value = []
    if not isinstance(authors_value, Sequence) or isinstance(authors_value, str | bytes):
        raise RecordValidationError("authors_invalid")
    authors: list[str] = []
    author_keys: set[str] = set()
    for item in authors_value:
        name = _author_name(item)
        key = normalize_author_name(name)
        if key not in author_keys:
            authors.append(name)
            author_keys.add(key)

    year_value = raw.get("publication_year", raw.get("year"))
    if year_value is not None and (isinstance(year_value, bool) or not isinstance(year_value, int)):
        raise RecordValidationError("publication_year_invalid")
    publication_year = cast(int | None, year_value)
    venue_name, venue_track = _parse_venue(raw.get("venue"))
    source_revision = _string(raw.get("source_revision"), "source_revision_required") or "unknown"
    retrieved_at = _parse_datetime(
        raw.get("retrieved_at"), "retrieved_at_invalid", default_now=True
    )
    abstract = _string(raw.get("abstract"), "abstract_invalid")
    source_url = _string(
        raw.get("source_url", raw.get("url", raw.get("pdf_url"))), "source_url_invalid"
    )
    pdf_url = _string(raw.get("pdf_url", source_url), "pdf_url_invalid")
    acceptance = _string(
        raw.get("acceptance_type", raw.get("acceptance_decision")), "acceptance_decision_invalid"
    )
    first_published_at = _parse_optional_datetime(
        raw.get("first_published_at", raw.get("published_at")), "first_published_at_invalid"
    )
    return _ValidatedRecord(
        raw=raw,
        source=source,
        source_revision=source_revision,
        retrieved_at=retrieved_at,
        title=title,
        title_key=normalize_title(title),
        authors=tuple(authors),
        author_keys=frozenset(author_keys),
        venue_name=venue_name,
        venue_track=venue_track,
        publication_year=publication_year,
        abstract=abstract,
        source_url=source_url,
        acceptance_decision=acceptance,
        first_published_at=first_published_at,
        pdf_url=pdf_url,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable(item) for item in value]
    return value


def _path_contains_symlink(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _write_artifact(staging_dir: Path, checksum: str, encoded: bytes) -> Path:
    with _ARTIFACT_LOCK:
        return _write_artifact_unlocked(staging_dir, checksum, encoded)


def _write_artifact_unlocked(staging_dir: Path, checksum: str, encoded: bytes) -> Path:
    artifact_path = staging_dir / f"{checksum}.json"
    if artifact_path.is_symlink():
        raise RecordValidationError("artifact_symlink")
    if artifact_path.exists():
        if not artifact_path.is_file():
            raise RecordValidationError("artifact_invalid")
        existing = artifact_path.read_bytes()
        if existing != encoded or hashlib.sha256(existing).hexdigest() != checksum:
            raise RecordValidationError("artifact_checksum_mismatch")
        return artifact_path

    temporary_path = staging_dir / f".{checksum}.{uuid4().hex}.tmp"
    try:
        file_descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
            raise
        try:
            os.replace(temporary_path, artifact_path)
        except OSError:
            # Windows refuses a competing replace once another worker has
            # published the same checksum.  The winner is valid if its bytes
            # match; retain the atomic publication semantics and reuse it.
            if artifact_path.is_symlink() or not artifact_path.exists():
                raise
            existing = artifact_path.read_bytes()
            if existing != encoded or hashlib.sha256(existing).hexdigest() != checksum:
                raise RecordValidationError("artifact_checksum_mismatch") from None
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass

    if artifact_path.is_symlink():
        raise RecordValidationError("artifact_symlink")
    existing = artifact_path.read_bytes()
    if existing != encoded or hashlib.sha256(existing).hexdigest() != checksum:
        raise RecordValidationError("artifact_checksum_mismatch")
    return artifact_path


def _canonical_payload(
    record: Mapping[str, Any], *, staging_dir: str | Path | None
) -> tuple[dict[str, Any], str, Path]:
    payload = {
        str(key): value
        for key, value in record.items()
        if not str(key).startswith("_") and str(key) != "staging_dir"
    }
    try:
        encoded = json.dumps(
            _jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecordValidationError("record_not_json_serializable") from exc
    checksum = hashlib.sha256(encoded).hexdigest()
    if staging_dir is None:
        raise RecordValidationError("staging_dir_required")
    if not isinstance(staging_dir, str | os.PathLike):
        raise RecordValidationError("staging_dir_invalid")
    configured_dir = Path(staging_dir).expanduser()
    if _path_contains_symlink(configured_dir):
        raise RecordValidationError("staging_dir_symlink")
    try:
        configured_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RecordValidationError("staging_dir_invalid") from exc
    if not configured_dir.is_dir() or _path_contains_symlink(configured_dir):
        raise RecordValidationError("staging_dir_invalid")
    artifact_path = _write_artifact(configured_dir.resolve(), checksum, encoded)
    return cast(dict[str, Any], json.loads(encoded.decode("utf-8"))), checksum, artifact_path


def _iter_external_values(value: object) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        if "namespace" in value or "name" in value:
            namespace = value.get("namespace", value.get("name"))
            return [(str(namespace), value.get("value", value.get("id")))]
        return [(str(key), item) for key, item in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        result: list[tuple[str, object]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError("invalid_external_id")
            namespace = item.get("namespace", item.get("name"))
            if namespace is None:
                raise ValueError("invalid_external_id")
            result.append((str(namespace), item.get("value", item.get("id"))))
        return result
    raise ValueError("invalid_external_id")


def _normalize_external_ids(record: Mapping[str, Any]) -> tuple[_ExternalIdentifier, ...]:
    raw_values: list[tuple[str, object]] = []
    external = record.get("external_ids")
    if external is not None:
        raw_values.extend(_iter_external_values(external))
    for namespace in ("doi", "arxiv", "arxiv_id"):
        if namespace in record:
            raw_values.append((namespace, record[namespace]))

    normalized: dict[tuple[str, str], str | None] = {}
    for namespace_value, value in raw_values:
        namespace = namespace_value.strip().lower().replace("-", "_")
        if value is None:
            continue
        if namespace in {"doi", "doi_id", "digital_object_identifier"}:
            if not isinstance(value, str):
                raise ValueError("invalid_doi")
            try:
                canonical = normalize_doi(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_doi") from exc
            key = ("doi", canonical)
            version = None
        elif namespace in {"arxiv", "arxiv_id", "ar_xiv"}:
            if not isinstance(value, str):
                raise ValueError("invalid_arxiv_id")
            try:
                canonical, version = normalize_arxiv(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_arxiv_id") from exc
            key = ("arxiv", canonical)
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid_external_id")
            canonical = value.strip().casefold()
            key = (namespace, canonical)
            version = None
        prior = normalized.get(key, "__missing__")
        if prior != "__missing__" and prior != version:
            raise ValueError("conflicting_identifier_versions")
        normalized[key] = version

    return tuple(
        _ExternalIdentifier(namespace=namespace, value=value, version=version)
        for (namespace, value), version in normalized.items()
    )


def _root_paper(session: Session, paper_id: UUID) -> UUID:
    """Follow manual redirects while guarding against corrupt cycles."""

    seen: set[UUID] = set()
    current = paper_id
    while current not in seen:
        seen.add(current)
        paper = session.get(Paper, current)
        if paper is None or paper.merged_into is None:
            return current
        current = paper.merged_into
    raise IdentityResolutionError("redirect_cycle")


def _paper_author_keys(session: Session, paper_id: UUID) -> set[str]:
    statement = (
        select(Author.normalized_name)
        .join(PaperAuthor, PaperAuthor.author_id == Author.id)
        .where(PaperAuthor.paper_id == paper_id)
    )
    return set(session.execute(statement).scalars())


def _authors_compatible(existing: set[str], incoming: frozenset[str]) -> bool:
    return not existing or not incoming or bool(existing & incoming)


def _years_compatible(existing: int | None, incoming: int | None) -> bool:
    return existing is None or incoming is None or existing == incoming


def _metadata_compatible(
    paper: Paper,
    record: _ValidatedRecord,
    session: Session,
    *,
    check_year: bool = True,
) -> bool:
    if normalize_title(paper.title) != record.title_key:
        return False
    years_ok = not check_year or _years_compatible(
        paper.publication_year, record.publication_year
    )
    return years_ok and _authors_compatible(
        _paper_author_keys(session, paper.id), record.author_keys
    )


def _find_title_candidate(
    session: Session,
    record: _ValidatedRecord,
    incoming_namespaces: set[str] | None = None,
) -> Paper | None:
    papers = session.execute(select(Paper).where(Paper.merged_into.is_(None))).scalars().all()
    compatible: list[tuple[int, Paper]] = []
    for paper in papers:
        if normalize_title(paper.title) != record.title_key:
            continue
        existing_namespaces = set(
            session.execute(
                select(PaperIdentifier.namespace).where(PaperIdentifier.paper_id == paper.id)
            ).scalars()
        )
        existing_strong = existing_namespaces & {"doi", "arxiv"}
        incoming_strong = (incoming_namespaces or set()) & {"doi", "arxiv"}
        aliases_crossing_versions = (
            len(existing_strong) == 1
            and len(incoming_strong) == 1
            and existing_strong != incoming_strong
        )
        if not aliases_crossing_versions and not _years_compatible(
            paper.publication_year, record.publication_year
        ):
            continue
        existing_authors = _paper_author_keys(session, paper.id)
        if not _authors_compatible(existing_authors, record.author_keys):
            continue
        compatible.append((len(existing_authors & record.author_keys), paper))
    if not compatible:
        return None
    compatible.sort(key=lambda item: item[0], reverse=True)
    if len(compatible) > 1 and compatible[0][0] == compatible[1][0]:
        return None
    return compatible[0][1]


def _identifier_rows(
    session: Session, identifiers: Sequence[_ExternalIdentifier]
) -> list[PaperIdentifier]:
    rows: list[PaperIdentifier] = []
    for identifier in identifiers:
        statement = select(PaperIdentifier).where(
            PaperIdentifier.namespace == identifier.namespace,
            PaperIdentifier.value == identifier.value,
        )
        row = session.execute(statement).scalar_one_or_none()
        if row is not None:
            rows.append(row)
    return rows


def _upsert_source_record(
    session: Session,
    record: _ValidatedRecord,
    payload: dict[str, Any],
    checksum: str,
    artifact_path: Path,
) -> SourceRecord:
    statement = select(SourceRecord).where(
        SourceRecord.source == record.source,
        SourceRecord.source_revision == record.source_revision,
        SourceRecord.content_sha256 == checksum,
    )
    existing = session.execute(statement).scalar_one_or_none()
    if existing is not None:
        return existing
    source_item_id = record.raw.get(
        "source_item_id", record.raw.get("source_id", record.raw.get("id"))
    )
    if source_item_id is not None and not isinstance(source_item_id, str | int):
        raise RecordValidationError("source_item_id_invalid")
    insert_statement = (
        pg_insert(SourceRecord)
        .values(
            id=uuid4(),
            paper_id=None,
            source=record.source,
            source_item_id=str(source_item_id) if source_item_id is not None else None,
            source_revision=record.source_revision,
            retrieved_at=record.retrieved_at,
            content_sha256=checksum,
            artifact_path=str(artifact_path),
            source_url=record.source_url,
            original_title=record.title,
            acceptance_decision=record.acceptance_decision,
            raw_json=payload,
        )
        .on_conflict_do_nothing(
            index_elements=["source", "source_revision", "content_sha256"]
        )
    )
    session.execute(insert_statement)
    session.flush()
    source_record = session.execute(
        select(SourceRecord).where(
            SourceRecord.source == record.source,
            SourceRecord.source_revision == record.source_revision,
            SourceRecord.content_sha256 == checksum,
        )
    ).scalar_one_or_none()
    if source_record is None:
        raise IdentityResolutionError("source_record_upsert_failed")
    return source_record


def _upsert_provenance(
    session: Session, record: _ValidatedRecord, source_record: SourceRecord
) -> None:
    field_values: dict[str, object] = {
        "title": record.title,
        "authors": list(record.authors),
        "venue": {"name": record.venue_name, "track": record.venue_track},
        "publication_year": record.publication_year,
        "external_ids": record.raw.get("external_ids", {}),
        "acceptance_decision": record.acceptance_decision,
        "source_url": record.source_url,
    }
    for field_name, value in field_values.items():
        statement = select(FieldProvenance).where(
            FieldProvenance.source_record_id == source_record.id,
            FieldProvenance.field_name == field_name,
        )
        existing = session.execute(statement).scalar_one_or_none()
        if existing is not None:
            if existing.paper_id is None and source_record.paper_id is not None:
                existing.paper_id = source_record.paper_id
            continue
        session.add(
            FieldProvenance(
                id=uuid4(),
                source_record_id=source_record.id,
                paper_id=source_record.paper_id,
                field_name=field_name,
                value=cast(Any, _jsonable(value)),
                original_value=cast(Any, _jsonable(value)),
                source=record.source,
                source_revision=record.source_revision,
                retrieved_at=record.retrieved_at,
            )
        )
    session.flush()


def _upsert_venue(session: Session, name: str | None, track: str) -> UUID | None:
    if not name:
        return None
    statement = select(Venue).where(Venue.name == name, Venue.track == track)
    venue = session.execute(statement).scalar_one_or_none()
    if venue is None:
        venue = Venue(id=uuid4(), name=name, track=track)
        session.add(venue)
        session.flush()
    return venue.id


def _upsert_author(session: Session, name: str) -> Author:
    normalized = normalize_author_name(name)
    statement = select(Author).where(Author.normalized_name == normalized)
    author = session.execute(statement).scalar_one_or_none()
    if author is None:
        author = Author(id=uuid4(), name=name, normalized_name=normalized)
        session.add(author)
        session.flush()
    return author


def _upsert_authors(session: Session, paper_id: UUID, names: Sequence[str]) -> None:
    existing = (
        session.execute(select(PaperAuthor).where(PaperAuthor.paper_id == paper_id))
        .scalars()
        .all()
    )
    existing_ids = {link.author_id for link in existing}
    next_position = max((link.position for link in existing), default=-1) + 1
    for name in names:
        author = _upsert_author(session, name)
        if author.id in existing_ids:
            continue
        session.add(PaperAuthor(paper_id=paper_id, position=next_position, author_id=author.id))
        existing_ids.add(author.id)
        next_position += 1
    session.flush()


def _upsert_version(
    session: Session,
    paper_id: UUID,
    record: _ValidatedRecord,
    source_record: SourceRecord,
    identifiers: Sequence[_ExternalIdentifier],
) -> None:
    version = _document_version(record, identifiers)
    content = record.raw.get("content_sha256", source_record.content_sha256)
    if not isinstance(content, str) or len(content) != 64:
        content = source_record.content_sha256
    source_url = record.source_url or ""
    statement = select(PaperVersion).where(
        PaperVersion.paper_id == paper_id,
        PaperVersion.source == record.source,
        PaperVersion.source_revision == record.source_revision,
        PaperVersion.content_sha256 == content,
        PaperVersion.version == version if version is not None else PaperVersion.version.is_(None),
    )
    if session.execute(statement).scalar_one_or_none() is not None:
        return
    redistribution = record.raw.get("redistribution", "unknown")
    if redistribution not in {"eligible", "restricted", "unknown"}:
        raise RecordValidationError("redistribution_invalid")
    parser_version = record.raw.get("parser_version", "unparsed")
    parse_status = record.raw.get("parse_status", "pending")
    if not isinstance(parser_version, str) or not isinstance(parse_status, str):
        raise RecordValidationError("version_metadata_invalid")
    session.add(
        PaperVersion(
            id=uuid4(),
            paper_id=paper_id,
            source=record.source,
            source_url=source_url,
            source_revision=record.source_revision,
            content_sha256=content,
            version=version,
            license_label=cast(str | None, record.raw.get("license_label")),
            redistribution=cast(str, redistribution),
            parser_version=parser_version,
            parse_status=parse_status,
        )
    )
    session.flush()


def _document_version(
    record: _ValidatedRecord, identifiers: Sequence[_ExternalIdentifier]
) -> str | None:
    arxiv_version = next(
        (identifier.version for identifier in identifiers if identifier.namespace == "arxiv"), None
    )
    explicit_version = record.raw.get("version")
    if explicit_version is not None and not isinstance(explicit_version, str):
        raise RecordValidationError("version_invalid")
    if arxiv_version is not None or any(
        identifier.namespace == "arxiv" for identifier in identifiers
    ):
        if explicit_version is not None and explicit_version != arxiv_version:
            raise RecordValidationError("version_mismatch")
        return arxiv_version
    return cast(str | None, explicit_version)


def _create_quarantine(
    session: Session,
    record: _ValidatedRecord,
    payload: dict[str, Any],
    checksum: str,
    artifact_path: Path,
    reason: str,
) -> QuarantineRecord:
    insert_statement = (
        pg_insert(QuarantineRecord)
        .values(
            id=uuid4(),
            reason=reason,
            source=record.source,
            source_revision=record.source_revision,
            content_sha256=checksum,
            artifact_path=str(artifact_path),
            raw_json=payload,
        )
        .on_conflict_do_nothing(
            index_elements=["source", "source_revision", "content_sha256", "reason"]
        )
        .returning(QuarantineRecord.id)
    )
    quarantine_id = session.execute(insert_statement).scalar_one_or_none()
    session.flush()
    if quarantine_id is None:
        statement = select(QuarantineRecord).where(
            QuarantineRecord.source == record.source,
            QuarantineRecord.source_revision == record.source_revision,
            QuarantineRecord.content_sha256 == checksum,
            QuarantineRecord.reason == reason,
        )
        quarantine = session.execute(statement).scalar_one_or_none()
        if quarantine is None:
            raise IdentityResolutionError("quarantine_upsert_failed")
        return quarantine
    quarantine = session.get(QuarantineRecord, quarantine_id)
    if quarantine is None:
        raise IdentityResolutionError("quarantine_upsert_failed")
    return quarantine


def _create_conflict(
    session: Session,
    reason: str,
    source_record: SourceRecord,
    *,
    paper_id: UUID | None = None,
    other_paper_id: UUID | None = None,
    details: Mapping[str, object] | None = None,
) -> IdentityConflict:
    insert_statement = (
        pg_insert(IdentityConflict)
        .values(
            id=uuid4(),
            reason=reason,
            status="open",
            paper_id=paper_id,
            other_paper_id=other_paper_id,
            source_record_id=source_record.id,
            details=cast(dict[str, Any], _jsonable(dict(details or {}))),
        )
        .on_conflict_do_nothing(index_elements=["source_record_id", "reason"])
        .returning(IdentityConflict.id)
    )
    conflict_id = session.execute(insert_statement).scalar_one_or_none()
    session.flush()
    if conflict_id is None:
        statement = select(IdentityConflict).where(
            IdentityConflict.source_record_id == source_record.id,
            IdentityConflict.reason == reason,
        )
        conflict = session.execute(statement).scalar_one_or_none()
        if conflict is None:
            raise IdentityResolutionError("identity_conflict_upsert_failed")
        return conflict
    conflict = session.get(IdentityConflict, conflict_id)
    if conflict is None:
        raise IdentityResolutionError("identity_conflict_upsert_failed")
    return conflict


def _discard_candidate(session: Session, paper_id: UUID) -> None:
    session.execute(delete(Paper).where(Paper.id == paper_id))
    session.flush()


def _adopt_candidate_aliases(
    session: Session,
    candidate_id: UUID,
    winner_id: UUID,
    source_record: SourceRecord,
    record: _ValidatedRecord,
) -> Paper:
    """Move aliases from a raced candidate to the database winner, then remove it."""

    winner = session.get(Paper, winner_id)
    if winner is None:
        raise IdentityResolutionError("winning_paper_missing")
    if not _metadata_compatible(winner, record, session):
        _discard_candidate(session, candidate_id)
        conflict = _create_conflict(
            session,
            "incompatible_metadata",
            source_record,
            paper_id=winner_id,
            details={
                "title": record.title,
                "publication_year": record.publication_year,
            },
        )
        raise IdentityConflictError(conflict.reason, conflict.id)

    candidate_aliases = session.execute(
        select(PaperIdentifier).where(PaperIdentifier.paper_id == candidate_id)
    ).scalars().all()
    for candidate_alias in candidate_aliases:
        existing = session.execute(
            select(PaperIdentifier).where(
                PaperIdentifier.namespace == candidate_alias.namespace,
                PaperIdentifier.value == candidate_alias.value,
                PaperIdentifier.paper_id != candidate_id,
            )
        ).scalar_one_or_none()
        if existing is None:
            candidate_alias.paper_id = winner_id
            continue
        existing_root = _root_paper(session, existing.paper_id)
        if existing_root != winner_id:
            _discard_candidate(session, candidate_id)
            conflict = _create_conflict(
                session,
                "contradictory_strong_ids"
                if candidate_alias.namespace in {"doi", "arxiv"}
                else "contradictory_identifiers",
                source_record,
                paper_id=winner_id,
                other_paper_id=existing_root,
                details={
                    "namespace": candidate_alias.namespace,
                    "value": candidate_alias.value,
                },
            )
            raise IdentityConflictError(conflict.reason, conflict.id)
        session.delete(candidate_alias)
    session.flush()
    _discard_candidate(session, candidate_id)
    winner = session.get(Paper, winner_id)
    if winner is None:
        raise IdentityResolutionError("winning_paper_missing")
    return winner


def _update_paper_metadata(session: Session, paper: Paper, record: _ValidatedRecord) -> None:
    if paper.abstract is None and record.abstract:
        paper.abstract = record.abstract
    if paper.publication_year is None:
        paper.publication_year = record.publication_year
    if paper.venue_id is None:
        paper.venue_id = _upsert_venue(session, record.venue_name, record.venue_track)
    if paper.first_published_at is None:
        paper.first_published_at = record.first_published_at
    if paper.pdf_url is None:
        paper.pdf_url = record.pdf_url
    if paper.acceptance_type == "unknown" and record.acceptance_decision:
        paper.acceptance_type = record.acceptance_decision


def resolve_paper(
    record: dict[str, object], session: Session, *, staging_dir: str | Path | None = None
) -> UUID:
    """Resolve one source record to a stable paper UUID without committing the session."""

    validated = _validate_record(record)
    payload, checksum, artifact_path = _canonical_payload(validated.raw, staging_dir=staging_dir)
    try:
        identifiers = _normalize_external_ids(validated.raw)
    except ValueError as exc:
        reason = str(exc)
        quarantine = _create_quarantine(
            session, validated, payload, checksum, artifact_path, reason
        )
        raise QuarantineError(reason, quarantine.id) from exc
    _document_version(validated, identifiers)

    source_record = _upsert_source_record(session, validated, payload, checksum, artifact_path)
    _upsert_provenance(session, validated, source_record)
    if source_record.paper_id is not None:
        return _root_paper(session, source_record.paper_id)

    identifier_rows = _identifier_rows(session, identifiers)
    identifier_roots = {_root_paper(session, row.paper_id) for row in identifier_rows}
    if len(identifier_roots) > 1:
        conflict = _create_conflict(
            session,
            "contradictory_strong_ids" if any(
                item.namespace in {"doi", "arxiv"} for item in identifiers
            ) else "contradictory_identifiers",
            source_record,
            paper_id=next(iter(identifier_roots)),
            other_paper_id=next(iter(identifier_roots - {next(iter(identifier_roots))})),
            details={"identifiers": [identifier.__dict__ for identifier in identifiers]},
        )
        raise IdentityConflictError(conflict.reason, conflict.id)

    created = False
    if identifier_roots:
        paper_id = next(iter(identifier_roots))
        paper = session.get(Paper, paper_id)
        if paper is None:
            raise IdentityResolutionError("paper_missing_for_identifier")
        incoming_namespaces = {identifier.namespace for identifier in identifiers}
        existing_namespaces = {row.namespace for row in identifier_rows}
        if not _metadata_compatible(
            paper,
            validated,
            session,
            check_year=bool(existing_namespaces & incoming_namespaces),
        ):
            conflict = _create_conflict(
                session,
                "incompatible_metadata",
                source_record,
                paper_id=paper.id,
                details={"title": validated.title, "publication_year": validated.publication_year},
            )
            raise IdentityConflictError(conflict.reason, conflict.id)
    else:
        paper = _find_title_candidate(
            session,
            validated,
            {identifier.namespace for identifier in identifiers},
        )
        if paper is None:
            paper = Paper(
                id=uuid4(),
                title=validated.title,
                abstract=validated.abstract,
                publication_year=validated.publication_year,
                venue_id=None,
                first_published_at=validated.first_published_at,
                first_seen_at=datetime.now(UTC),
                pdf_url=validated.pdf_url,
                acceptance_type=validated.acceptance_decision or "unknown",
                metadata_status="active",
            )
            session.add(paper)
            session.flush()
            created = True

    for identifier in identifiers:
        statement = select(PaperIdentifier).where(
            PaperIdentifier.namespace == identifier.namespace,
            PaperIdentifier.value == identifier.value,
        )
        existing = session.execute(statement).scalar_one_or_none()
        if existing is not None:
            existing_root = _root_paper(session, existing.paper_id)
            if existing_root != paper.id:
                if created:
                    paper = _adopt_candidate_aliases(
                        session, paper.id, existing_root, source_record, validated
                    )
                    created = False
                    continue
                conflict = _create_conflict(
                    session,
                    "contradictory_strong_ids"
                    if identifier.namespace in {"doi", "arxiv"}
                    else "contradictory_identifiers",
                    source_record,
                    paper_id=paper.id,
                    other_paper_id=existing_root,
                    details={"namespace": identifier.namespace, "value": identifier.value},
                )
                raise IdentityConflictError(conflict.reason, conflict.id)
            continue
        insert_statement = pg_insert(PaperIdentifier).values(
            id=uuid4(),
            paper_id=paper.id,
            namespace=identifier.namespace,
            value=identifier.value,
            version=None if identifier.namespace == "arxiv" else identifier.version,
        ).on_conflict_do_nothing(index_elements=["namespace", "value"]).returning(
            PaperIdentifier.paper_id
        )
        result = cast(CursorResult[Any], session.execute(insert_statement))
        inserted_paper_id = result.scalar_one_or_none()
        session.flush()
        if inserted_paper_id is None:
            existing = session.execute(
                select(PaperIdentifier).where(
                    PaperIdentifier.namespace == identifier.namespace,
                    PaperIdentifier.value == identifier.value,
                )
            ).scalar_one()
            existing_root = _root_paper(session, existing.paper_id)
            if existing_root != paper.id:
                if created:
                    paper = _adopt_candidate_aliases(
                        session, paper.id, existing_root, source_record, validated
                    )
                    created = False
                    continue
                conflict = _create_conflict(
                    session,
                    "contradictory_strong_ids"
                    if identifier.namespace in {"doi", "arxiv"}
                    else "contradictory_identifiers",
                    source_record,
                    paper_id=paper.id,
                    other_paper_id=existing_root,
                    details={"namespace": identifier.namespace, "value": identifier.value},
                )
                raise IdentityConflictError(conflict.reason, conflict.id)

    _update_paper_metadata(session, paper, validated)
    source_record.paper_id = paper.id
    session.flush()
    _upsert_provenance(session, validated, source_record)
    _upsert_authors(session, paper.id, validated.authors)
    _upsert_version(session, paper.id, validated, source_record, identifiers)
    session.flush()
    return _root_paper(session, paper.id)


def _merge_papers(
    surviving_paper_id: UUID, losing_paper_id: UUID, session: Session, reason: str | None = None
) -> UUID:
    if surviving_paper_id == losing_paper_id:
        raise MergeValidationError("cannot_merge_paper_with_itself")
    surviving = session.get(Paper, surviving_paper_id)
    losing = session.get(Paper, losing_paper_id)
    if surviving is None or losing is None:
        raise MergeValidationError("paper_not_found")
    if surviving.merged_into is not None:
        raise MergeValidationError("surviving_paper_already_redirected")
    if losing.merged_into is not None:
        raise MergeValidationError("losing_paper_already_redirected")
    if _root_paper(session, surviving.id) == _root_paper(session, losing.id):
        raise MergeValidationError("merge_would_create_cycle")

    survivor_versions = session.execute(
        select(PaperVersion).where(PaperVersion.paper_id == surviving.id)
    ).scalars().all()
    losing_versions = session.execute(
        select(PaperVersion).where(PaperVersion.paper_id == losing.id)
    ).scalars().all()
    survivor_version_keys = {
        (item.source, item.source_revision, item.content_sha256, item.version)
        for item in survivor_versions
    }
    if len(survivor_version_keys) != len(survivor_versions):
        raise MergeConflictError("duplicate_paper_version")
    losing_seen: set[tuple[str, str, str, str | None]] = set()
    for item in losing_versions:
        version_key = (item.source, item.source_revision, item.content_sha256, item.version)
        if version_key in survivor_version_keys or version_key in losing_seen:
            raise MergeConflictError("duplicate_paper_version")
        losing_seen.add(version_key)

    identifiers = session.execute(
        select(PaperIdentifier).where(PaperIdentifier.paper_id == losing.id)
    ).scalars().all()
    for identifier in identifiers:
        existing = session.execute(
            select(PaperIdentifier).where(
                PaperIdentifier.namespace == identifier.namespace,
                PaperIdentifier.value == identifier.value,
                PaperIdentifier.paper_id != losing.id,
            )
        ).scalar_one_or_none()
        if existing is not None and _root_paper(session, existing.paper_id) != surviving.id:
            raise IdentityConflictError("merge_identifier_conflict")
        if existing is not None:
            session.delete(identifier)
        else:
            identifier.paper_id = surviving.id

    versions = (
        session.execute(select(PaperVersion).where(PaperVersion.paper_id == losing.id))
        .scalars()
        .all()
    )
    for version in versions:
        version.paper_id = surviving.id

    survivor_links = session.execute(
        select(PaperAuthor).where(PaperAuthor.paper_id == surviving.id)
    ).scalars().all()
    survivor_author_ids = {link.author_id for link in survivor_links}
    next_position = max((link.position for link in survivor_links), default=-1) + 1
    losing_links = (
        session.execute(select(PaperAuthor).where(PaperAuthor.paper_id == losing.id))
        .scalars()
        .all()
    )
    for link in losing_links:
        if link.author_id in survivor_author_ids:
            session.delete(link)
        else:
            link.paper_id = surviving.id
            link.position = next_position
            survivor_author_ids.add(link.author_id)
            next_position += 1

    session.execute(
        sa.update(SourceRecord)
        .where(SourceRecord.paper_id == losing.id)
        .values(paper_id=surviving.id)
    )
    session.execute(
        sa.update(FieldProvenance)
        .where(FieldProvenance.paper_id == losing.id)
        .values(paper_id=surviving.id)
    )
    losing.merged_into = surviving.id
    losing.metadata_status = "merged"
    session.add(PaperRedirect(from_paper_id=losing.id, to_paper_id=surviving.id, reason=reason))
    session.flush()
    return surviving.id


def merge_papers(
    surviving_paper_id: UUID | Session,
    losing_paper_id: UUID,
    session: Session | UUID | None = None,
    *,
    reason: str | None = None,
) -> UUID:
    """Record a manual merge, accepting both ``(survivor, loser, session)`` and
    ``(session, loser, survivor)`` calling conventions for migration tooling.
    """

    if isinstance(surviving_paper_id, Session):
        actual_session = surviving_paper_id
        if not isinstance(session, UUID):
            raise TypeError("surviving_paper_id_required")
        actual_losing = losing_paper_id
        actual_surviving = session
    else:
        actual_surviving = surviving_paper_id
        actual_losing = losing_paper_id
        if not isinstance(session, Session):
            raise TypeError("session_required")
        actual_session = session
    return _merge_papers(actual_surviving, actual_losing, actual_session, reason)


def merge_paper_identities(
    losing_paper_id: UUID, surviving_paper_id: UUID, session: Session, *, reason: str | None = None
) -> UUID:
    """Explicit loser-first spelling for callers performing a manual merge."""

    return _merge_papers(surviving_paper_id, losing_paper_id, session, reason)
