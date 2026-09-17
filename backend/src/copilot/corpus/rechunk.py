"""Re-chunk the stored corpus without the source PDFs.

Changing chunking policy normally means re-parsing, and re-parsing needs the
PDFs — which the venue sweep deletes once a paper's text is safely stored. The
text is still here though, in the chunks themselves: fixed-window chunks are
literal slices of their section that overlap by a known number of tokens, and
they preserve the original line structure. So a section can be stitched back
together from its own chunks and re-chunked in place.

Measured by round trip on 120 random versions: 99.67% of sections reconstruct
byte-for-byte, verified by re-running the *original* policy over the
reconstruction and requiring identical chunk text. The remainder differ by a
token or two at one join, where the exact string overlap could not be located
and the boundary is recovered by re-tokenizing instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from ..db.models import PaperVersion
from .chunk import ChunkerConfig, Section, TokenSpans, chunk_document
from .documents import store_parsed_document
from .parse import ParseResult, ParseStatus

# How much of the next chunk identifies the overlap, and how far back to look.
ANCHOR_CHARS = 80
SEARCH_WINDOW = 8000


@dataclass
class RechunkStats:
    versions: int = 0
    sections: int = 0
    exact_joins: int = 0
    trimmed_joins: int = 0
    chunks_before: int = 0
    chunks_after: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "versions": self.versions,
            "sections": self.sections,
            "exact_joins": self.exact_joins,
            "trimmed_joins": self.trimmed_joins,
            "chunks_before": self.chunks_before,
            "chunks_after": self.chunks_after,
        }


def merge_overlap(
    left: str, right: str, spans: TokenSpans, overlap_tokens: int, stats: RechunkStats
) -> str:
    """Join two consecutive chunks, removing the text they share.

    The overlap is found as an exact string match — the largest suffix of
    ``left`` that is a prefix of ``right`` — because the token boundary cannot
    be recovered: a chunk re-tokenizes slightly differently from the section it
    was cut from. When no exact match is found the boundary is approximated by
    dropping the first ``overlap_tokens`` tokens, which lands within a token.
    """

    if not left:
        return right
    anchor = right[: min(ANCHOR_CHARS, len(right))]
    tail = left[-SEARCH_WINDOW:]
    index = tail.find(anchor)
    while index >= 0:
        size = len(tail) - index  # ascending index == descending overlap
        if size <= len(right) and left.endswith(right[:size]):
            stats.exact_joins += 1
            return left + right[size:]
        index = tail.find(anchor, index + 1)
    tokens = list(spans(right))
    stats.trimmed_joins += 1
    if len(tokens) <= overlap_tokens:
        return left
    return left + right[tokens[overlap_tokens - 1][1] :]


def reconstruct_section(
    rows: Sequence[Any], spans: TokenSpans, overlap_tokens: int, stats: RechunkStats
) -> Section:
    """Rebuild one section from its stored chunks, in ordinal order."""

    first = rows[0]
    if first.fragment:
        # Labelled table fragments repeat the heading and do not overlap.
        body = "\n".join(
            row.text.split("\n", 1)[1] if "\n" in row.text else row.text for row in rows
        )
    else:
        body = rows[0].text
        for row in rows[1:]:
            body = merge_overlap(body, row.text, spans, overlap_tokens, stats)
    return Section(
        name=first.section_path,
        text=body,
        page_start=first.page_start,
        page_end=first.page_end,
        ordinal=first.section_ordinal,
        kind=first.kind,
    )


def reconstruct_document(
    rows: Sequence[Any], spans: TokenSpans, overlap_tokens: int, stats: RechunkStats
) -> list[Section]:
    """Rebuild every section of one paper version, ordered as the parser left them."""

    grouped: dict[int, list[Any]] = {}
    for row in rows:
        grouped.setdefault(row.section_ordinal, []).append(row)
    sections = [
        reconstruct_section(group, spans, overlap_tokens, stats)
        for _, group in sorted(grouped.items())
    ]
    stats.sections += len(sections)
    return sections


_VERSION_CHUNKS = text(
    "select section_ordinal, ordinal, text, fragment, kind, page_start, page_end, section_path"
    " from chunks where paper_version_id = :version order by ordinal"
)


def rechunk_version(
    session: Session,
    version: PaperVersion,
    spans: TokenSpans,
    chunker: ChunkerConfig,
    *,
    source_overlap_tokens: int,
    stats: RechunkStats,
) -> int:
    """Re-chunk one version in place under the configured policy."""

    rows = session.execute(_VERSION_CHUNKS, {"version": version.id}).all()
    if not rows:
        return 0
    stats.chunks_before += len(rows)
    sections = reconstruct_document(rows, spans, source_overlap_tokens, stats)
    result = ParseResult(
        status=ParseStatus.PARSED,
        error_code=None,
        sections=tuple(sections),
        content_sha256=version.content_sha256,
        parser_version=version.parser_version,
        page_count=0,
        quality="low",
    )
    chunks = chunk_document(sections, spans, chunker)
    written = store_parsed_document(
        session,
        version,
        result,
        chunks,
        chunker_version=chunker.chunker_version,
        namespace=chunker.uuid_namespace,
    )
    stats.versions += 1
    stats.chunks_after += len(written)
    return len(written)


def pending_versions(
    engine: Engine,
    chunker_version: str,
    limit: int | None = None,
    *,
    shards: int = 1,
    shard: int = 0,
) -> list[UUID]:
    """Versions whose chunks are not yet at the target chunker revision.

    ``shards`` partitions the work by a hash of the version ID so several
    processes can run the pass at once without contending for the same rows.
    """

    statement = (
        "select distinct c.paper_version_id from chunks c"
        " where c.chunker_version <> :target"
    )
    if shards > 1:
        statement += " and abs(hashtext(c.paper_version_id::text)) % :shards = :shard"
    statement += " order by c.paper_version_id"
    if limit:
        statement += f" limit {int(limit)}"
    params: dict[str, Any] = {"target": chunker_version}
    if shards > 1:
        params.update(shards=shards, shard=shard)
    with engine.connect() as connection:
        return [row[0] for row in connection.execute(text(statement), params)]


def rechunk_corpus(
    engine: Engine,
    spans: TokenSpans,
    chunker: ChunkerConfig,
    *,
    source_overlap_tokens: int,
    session_factory: Callable[[], Session],
    limit: int | None = None,
    shards: int = 1,
    shard: int = 0,
    progress: Callable[[RechunkStats], None] | None = None,
    progress_every: int = 500,
) -> RechunkStats:
    """Re-chunk every version that is not already at the target revision.

    Each version commits on its own, so the pass is resumable: rerunning picks
    up whatever is still on the old revision.
    """

    stats = RechunkStats()
    for version_id in pending_versions(
        engine, chunker.chunker_version, limit, shards=shards, shard=shard
    ):
        with session_factory() as session:
            with session.begin():
                version = session.get(PaperVersion, version_id)
                if version is None:
                    continue
                rechunk_version(
                    session,
                    version,
                    spans,
                    chunker,
                    source_overlap_tokens=source_overlap_tokens,
                    stats=stats,
                )
        if progress and stats.versions % progress_every == 0:
            progress(stats)
    return stats
