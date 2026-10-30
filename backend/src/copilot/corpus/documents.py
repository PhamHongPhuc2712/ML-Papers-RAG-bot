"""Persist parse outcomes and chunks against a paper version."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import delete, true
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..db.models import Chunk, PaperVersion
from .chunk import DEFAULT_CHUNK_NAMESPACE, chunk_id
from .parse import ParseResult, ParseStatus

# 14 bound columns per row against PostgreSQL's 65535-parameter ceiling.
INSERT_BATCH = 2000


def store_parsed_document(
    session: Session,
    version: PaperVersion,
    result: ParseResult,
    chunks: Sequence[Mapping[str, Any]],
    *,
    chunker_version: str,
    namespace: UUID = DEFAULT_CHUNK_NAMESPACE,
) -> list[UUID]:
    """Record parser provenance on the version and upsert its chunks idempotently.

    A version keeps exactly one active chunking: rows produced by a different
    parser or chunker revision are removed before the new rows are inserted, and
    rows with identical identity are left untouched. The caller owns the
    transaction. Returns the chunk identifiers in ordinal order.
    """

    version.parser_version = result.parser_version
    if result.status is ParseStatus.PARSED:
        version.parse_status = ParseStatus.PARSED.value
    else:
        assert result.error_code is not None
        version.parse_status = result.error_code.value
    if result.content_sha256:
        version.content_sha256 = result.content_sha256
    session.flush()

    if result.status is not ParseStatus.PARSED:
        return []

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        identifier = chunk_id(
            version.paper_id,
            result.content_sha256,
            result.parser_version,
            chunker_version,
            int(chunk["section_ordinal"]),
            int(chunk["ordinal"]),
            namespace=namespace,
        )
        rows.append(
            {
                "id": identifier,
                "paper_version_id": version.id,
                "section_path": chunk["section"],
                "section_ordinal": int(chunk["section_ordinal"]),
                "kind": chunk["kind"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "ordinal": int(chunk["ordinal"]),
                "text": chunk["text"],
                "token_count": int(chunk["token_count"]),
                "parser_version": result.parser_version,
                "chunker_version": chunker_version,
                "evidence_default": bool(chunk["evidence_default"]),
                "fragment": chunk.get("fragment"),
            }
        )
    # Remove every chunk of this version that the new chunking does not produce.
    # Matching on the processing revisions alone was not enough: a version whose
    # document checksum changed keeps its ordinals while every chunk ID moves, so
    # the stale rows survived and collided on (paper_version_id, ordinal).
    session.execute(
        delete(Chunk).where(
            Chunk.paper_version_id == version.id,
            Chunk.id.notin_([row["id"] for row in rows]) if rows else true(),
        )
    )
    # PostgreSQL binds at most 65535 parameters per statement, so a document
    # with thousands of chunks has to be written in batches.
    for start in range(0, len(rows), INSERT_BATCH):
        batch = rows[start : start + INSERT_BATCH]
        session.execute(
            pg_insert(Chunk).values(batch).on_conflict_do_nothing(index_elements=["id"])
        )
    session.flush()
    return [row["id"] for row in rows]
