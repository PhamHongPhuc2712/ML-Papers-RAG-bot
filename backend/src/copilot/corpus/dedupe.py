"""Transactional paper identity resolution and source provenance."""

from __future__ import annotations

import hashlib
import json
import os
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


def _canonical_payload(record: Mapping[str, Any]) -> tuple[dict[str, Any], str, Path]:
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
    staging_value = (
        record.get("_staging_dir")
        or record.get("staging_dir")
        or os.environ.get("COPILOT_STAGING_DIR")
        or ".copilot-staging"
    )
    if not isinstance(staging_value, str | os.PathLike[str]):
        raise RecordValidationError("staging_dir_invalid")
    staging_dir = Path(staging_value).expanduser().resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = staging_dir / f"{checksum}.json"
    if not artifact_path.exists():
        artifact_path.write_bytes(encoded)
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
            try:
                canonical = normalize_doi(cast(str, value))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_doi") from exc
            key = ("doi", canonical)
            version = None
        elif namespace in {"arxiv", "arxiv_id", "ar_xiv"}:
            try:
                canonical, version = normalize_arxiv(cast(str, value))
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
        aliases_crossing_versions = bool(
            existing_namespaces & {"doi", "arxiv"}
            and incoming_namespaces
            and existing_namespaces.isdisjoint(incoming_namespaces)
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
    arxiv_version = next(
        (identifier.version for identifier in identifiers if identifier.namespace == "arxiv"), None
    )
    version = record.raw.get("version", arxiv_version)
    if version is not None and not isinstance(version, str):
        raise RecordValidationError("version_invalid")
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


def _create_quarantine(
    session: Session,
    record: _ValidatedRecord,
    payload: dict[str, Any],
    checksum: str,
    artifact_path: Path,
    reason: str,
) -> QuarantineRecord:
    statement = select(QuarantineRecord).where(
        QuarantineRecord.source == record.source,
        QuarantineRecord.source_revision == record.source_revision,
        QuarantineRecord.content_sha256 == checksum,
        QuarantineRecord.reason == reason,
    )
    existing = session.execute(statement).scalar_one_or_none()
    if existing is not None:
        return existing
    quarantine = QuarantineRecord(
        id=uuid4(),
        reason=reason,
        source=record.source,
        source_revision=record.source_revision,
        content_sha256=checksum,
        artifact_path=str(artifact_path),
        raw_json=payload,
    )
    session.add(quarantine)
    session.flush()
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
    conflict = IdentityConflict(
        id=uuid4(),
        reason=reason,
        status="open",
        paper_id=paper_id,
        other_paper_id=other_paper_id,
        source_record_id=source_record.id,
        details=cast(dict[str, Any], _jsonable(dict(details or {}))),
    )
    session.add(conflict)
    session.flush()
    return conflict


def _discard_candidate(session: Session, paper_id: UUID) -> None:
    session.execute(delete(Paper).where(Paper.id == paper_id))
    session.flush()


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
    raw_for_storage = dict(validated.raw)
    if staging_dir is not None:
        raw_for_storage["_staging_dir"] = str(staging_dir)
    payload, checksum, artifact_path = _canonical_payload(raw_for_storage)
    try:
        identifiers = _normalize_external_ids(validated.raw)
    except ValueError as exc:
        reason = str(exc)
        quarantine = _create_quarantine(
            session, validated, payload, checksum, artifact_path, reason
        )
        raise QuarantineError(reason, quarantine.id) from exc

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
                    _discard_candidate(session, paper.id)
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
        insert_statement = (
            pg_insert(PaperIdentifier)
            .values(
                id=uuid4(),
                paper_id=paper.id,
                namespace=identifier.namespace,
                value=identifier.value,
                version=None if identifier.namespace == "arxiv" else identifier.version,
            )
            .on_conflict_do_nothing(index_elements=["namespace", "value"])
        )
        result = cast(CursorResult[Any], session.execute(insert_statement))
        session.flush()
        if result.rowcount == 0:
            existing = session.execute(
                select(PaperIdentifier).where(
                    PaperIdentifier.namespace == identifier.namespace,
                    PaperIdentifier.value == identifier.value,
                )
            ).scalar_one()
            existing_root = _root_paper(session, existing.paper_id)
            if existing_root != paper.id:
                if created:
                    _discard_candidate(session, paper.id)
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
        raise ValueError("cannot_merge_paper_with_itself")
    surviving = session.get(Paper, surviving_paper_id)
    losing = session.get(Paper, losing_paper_id)
    if surviving is None or losing is None:
        raise ValueError("paper_not_found")
    if surviving.merged_into is not None:
        raise ValueError("surviving_paper_already_redirected")
    if losing.merged_into is not None:
        raise ValueError("losing_paper_already_redirected")
    if _root_paper(session, surviving.id) == _root_paper(session, losing.id):
        raise ValueError("merge_would_create_cycle")

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
