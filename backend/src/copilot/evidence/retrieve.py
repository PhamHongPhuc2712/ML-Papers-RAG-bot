"""Bounded evidence retrieval inside a set of papers (spec §9; plan P4.1; ORB plan O4).

Research mode finds papers first, then evidence inside them: BM25 and dense
candidates over the release's chunk collection, restricted to the named papers and
to chunks that are evidence by default, fused by RRF and reranked by the
cross-encoder on the chunk text itself. What reaches the generator is bounded three
ways: at most ``evidence`` pieces, at most ``per_paper`` from one paper, and a token
budget counted in the pinned tokenizer. A paper with no parsed text contributes its
abstract, labelled abstract-only, never silently.

Every piece keeps its provenance: the chunk id, the paper version, the section and
the page span from the stored row, and the stored text itself. The embedding
model's query instruction is never evidence text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text

from ..contracts import Evidence, Reranker
from ..corpus.chunk import TokenizerSpec, token_spans_for
from ..corpus.releases import ReleaseRecord, load_release
from ..models.embeddings import VectorModel, validate_vectors
from ..search.fusion import RRF_K, rrf
from ..search.index import CHUNKS, DENSE, SPARSE, IndexBuildError
from ..search.lexical import BM25Vocabulary
from ..search.rerank import rerank_order, validate_scores

EXPANSIONS = ("none", "section")
ABSTRACT_SECTION = "abstract"


class EvidenceError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class EvidenceConfig:
    papers: int = 10
    chunk_candidates: int = 40
    evidence: int = 12
    per_paper: int = 3
    # None: no floor; every reranked candidate may be evidence, bounded by the caps.
    min_rerank_score: float | None = None
    include_references: bool = False
    token_budget: int = 6000
    tokenizer_kind: str = "model"
    tokenizer_repo: str = ""
    tokenizer_revision: str = ""
    tokenizer_file: str = "tokenizer.json"
    tokenizer_sha256: str = ""
    expansion: str = "none"
    expansion_max_chars: int = 6000
    expansion_min_share: float = 0.2
    rrf_k: int = RRF_K

    def __post_init__(self) -> None:
        for name in ("papers", "chunk_candidates", "evidence", "per_paper", "token_budget"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.expansion not in EXPANSIONS:
            raise ValueError(f"expansion must be one of {', '.join(EXPANSIONS)}")
        if self.expansion_max_chars < 1 or not 0 < self.expansion_min_share <= 1:
            raise ValueError("expansion budget invalid")

    def replace(self, **changes: Any) -> EvidenceConfig:
        return replace(self, **changes)

    @property
    def tokenizer(self) -> TokenizerSpec:
        return TokenizerSpec(
            kind=self.tokenizer_kind,
            repo=self.tokenizer_repo,
            revision=self.tokenizer_revision,
            file=self.tokenizer_file,
            sha256=self.tokenizer_sha256,
        )

    def identity(self) -> dict[str, Any]:
        """Every value that shapes the evidence, for a run's manifest."""

        return {
            "papers": self.papers,
            "chunk_candidates": self.chunk_candidates,
            "evidence": self.evidence,
            "per_paper": self.per_paper,
            "min_rerank_score": self.min_rerank_score,
            "include_references": self.include_references,
            "token_budget": self.token_budget,
            "tokenizer": (
                f"{self.tokenizer_kind}:{self.tokenizer_repo}@{self.tokenizer_revision[:12]}"
            ),
            "expansion": self.expansion,
            "expansion_max_chars": self.expansion_max_chars,
            "expansion_min_share": self.expansion_min_share,
            "rrf_k": self.rrf_k,
        }


def load_evidence_config(path: str | Path) -> EvidenceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    retrieval = raw.get("retrieval") or {}
    context = raw.get("context") or {}
    tokenizer = context.get("tokenizer") or {}
    return EvidenceConfig(
        papers=int(retrieval.get("papers", 10)),
        chunk_candidates=int(retrieval.get("chunk_candidates", 40)),
        evidence=int(retrieval.get("evidence", 12)),
        per_paper=int(retrieval.get("per_paper", 3)),
        min_rerank_score=(
            None
            if retrieval.get("min_rerank_score") is None
            else float(retrieval["min_rerank_score"])
        ),
        include_references=bool(retrieval.get("include_references", False)),
        token_budget=int(context.get("token_budget", 6000)),
        tokenizer_kind=str(tokenizer.get("kind", "model")),
        tokenizer_repo=str(tokenizer.get("repo", "")),
        tokenizer_revision=str(tokenizer.get("revision", "")),
        tokenizer_file=str(tokenizer.get("file", "tokenizer.json")),
        tokenizer_sha256=str(tokenizer.get("sha256", "")),
        expansion=str(context.get("expansion", "none")),
        expansion_max_chars=int(context.get("expansion_max_chars", 6000)),
        expansion_min_share=float(context.get("expansion_min_share", 0.2)),
    )


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    version_id: str
    section: str
    kind: str
    ordinal: int
    page_start: int | None
    page_end: int | None
    text: str
    evidence_default: bool


_CHUNK_ROWS = """
select c.id::text as chunk_id,
       pv.paper_id::text as paper_id,
       c.paper_version_id::text as version_id,
       c.section_path,
       c.kind,
       c.ordinal,
       c.page_start,
       c.page_end,
       c.text,
       c.evidence_default
from chunks c
join paper_versions pv on pv.id = c.paper_version_id
"""
_BY_IDS = _CHUNK_ROWS + "where c.id = any(cast(:ids as uuid[]))\n"
_BY_SECTION = (
    _CHUNK_ROWS
    + "where c.paper_version_id = cast(:version as uuid) and c.section_path = :section\n"
    + "order by c.ordinal\n"
)
_PAPER_ROWS = """
select p.id::text as paper_id,
       p.title,
       p.abstract,
       exists (
           select 1 from paper_versions pv
           where pv.paper_id = p.id and pv.parse_status = 'parsed'
       ) as fulltext
from papers p
where p.id = any(cast(:ids as uuid[]))
"""


def _record(row: Mapping[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(row["chunk_id"]),
        paper_id=str(row["paper_id"]),
        version_id=str(row["version_id"]),
        section=str(row["section_path"]),
        kind=str(row["kind"]),
        ordinal=int(row["ordinal"]),
        page_start=row["page_start"],
        page_end=row["page_end"],
        text=str(row["text"]),
        evidence_default=bool(row["evidence_default"]),
    )


class ChunkRepository:
    """Stored chunk rows and paper metadata, by id, from PostgreSQL."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def chunks(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        with self._engine.connect() as connection:
            rows = connection.execute(text(_BY_IDS), {"ids": list(chunk_ids)}).mappings()
            return {str(row["chunk_id"]): _record(dict(row)) for row in rows}

    def section(self, version_id: str, section: str) -> list[ChunkRecord]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(_BY_SECTION), {"version": version_id, "section": section}
            ).mappings()
            return [_record(dict(row)) for row in rows]

    def papers(self, paper_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not paper_ids:
            return {}
        with self._engine.connect() as connection:
            rows = connection.execute(text(_PAPER_ROWS), {"ids": list(paper_ids)}).mappings()
            return {str(row["paper_id"]): dict(row) for row in rows}


def chunk_filter(paper_ids: Sequence[str] | None, *, include_references: bool) -> models.Filter:
    """Chunks of the named papers (every paper when None), evidence by default unless asked."""

    must: list[models.Condition] = []
    if paper_ids is not None:
        must.append(
            models.FieldCondition(key="paper_id", match=models.MatchAny(any=list(paper_ids)))
        )
    if not include_references:
        must.append(
            models.FieldCondition(key="evidence_default", match=models.MatchValue(value=True))
        )
    return models.Filter(
        must=must or None,
        must_not=[models.FieldCondition(key="deleted", match=models.MatchValue(value=True))],
    )


def build_context(
    evidence: Sequence[Evidence], *, token_budget: int, count_tokens: Callable[[str], int]
) -> tuple[list[Evidence], int]:
    """The pieces that fit the budget in rank order, and how many were dropped."""

    kept: list[Evidence] = []
    used = 0
    dropped = 0
    for item in evidence:
        tokens = count_tokens(item.text)
        if used + tokens > token_budget:
            dropped += 1
            continue
        used += tokens
        kept.append(item)
    return kept, dropped


def expand_to_section(
    evidence: Evidence, *, chunks: Sequence[ChunkRecord], max_chars: int, min_share: float
) -> Evidence:
    """haiku.rag's rule: the whole section when it fits and is worth it, else grow outward.

    ``chunks`` is the hit's section in order. When the joined section fits
    ``max_chars`` and is at least ``min_share`` of it, the section replaces the chunk.
    When it does not fit, neighbours are added around the hit, next then previous,
    while the text stays within the budget. The hit's own text is never clipped; a
    section too small to be worth it leaves the chunk as it was.
    """

    if evidence.chunk_id is None:
        return evidence
    ordered = list(chunks)
    position = next(
        (i for i, chunk in enumerate(ordered) if chunk.chunk_id == str(evidence.chunk_id)), None
    )
    if position is None:
        return evidence
    separator = "\n\n"
    whole = separator.join(chunk.text for chunk in ordered)
    if len(whole) <= max_chars:
        if len(whole) < min_share * max_chars or whole == evidence.text:
            return evidence
        return _widened(evidence, ordered, whole)
    low = high = position
    length = len(ordered[position].text)
    while True:
        grew = False
        for step in ("next", "previous"):
            index = high + 1 if step == "next" else low - 1
            if not 0 <= index < len(ordered):
                continue
            added = len(ordered[index].text) + len(separator)
            if length + added > max_chars:
                continue
            length += added
            grew = True
            if step == "next":
                high = index
            else:
                low = index
        if not grew:
            break
    window = separator.join(chunk.text for chunk in ordered[low : high + 1])
    if window == evidence.text:
        return evidence
    return _widened(evidence, ordered[low : high + 1], window)


def _widened(evidence: Evidence, chunks: Sequence[ChunkRecord], text: str) -> Evidence:
    starts = [c.page_start for c in chunks if c.page_start is not None]
    ends = [c.page_end for c in chunks if c.page_end is not None]
    return evidence.model_copy(
        update={
            "text": text,
            "expanded": True,
            "page_start": min(starts) if starts else evidence.page_start,
            "page_end": max(ends) if ends else evidence.page_end,
        }
    )


@dataclass
class EvidenceRetriever:
    """Chunk evidence inside named papers for one release, bounded as configured."""

    engine: Engine
    client: QdrantClient
    embedder: VectorModel
    reranker: Reranker | None
    config: EvidenceConfig
    data_dir: Path
    max_tokens: int | None = None
    last_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._repository = ChunkRepository(self.engine)
        self._releases: dict[str, tuple[ReleaseRecord, BM25Vocabulary]] = {}
        self._count = self._counter()

    def _counter(self) -> Callable[[str], int]:
        spans = token_spans_for(self.config.tokenizer, self.data_dir)
        return lambda text: len(spans(text))

    def _release(self, release_id: str) -> tuple[ReleaseRecord, BM25Vocabulary]:
        cached = self._releases.get(release_id)
        if cached is None:
            release = load_release(self.engine, release_id)
            if release.model_revision != self.embedder.identity:
                raise IndexBuildError(
                    "model_revision_mismatch",
                    f"{self.embedder.identity} != {release.model_revision}",
                )
            build = release.counts.get(CHUNKS)
            if not isinstance(build, dict) or not isinstance(build.get("vocabulary"), dict):
                raise IndexBuildError("collection_not_built", f"{release_id}:{CHUNKS}")
            vocabulary = BM25Vocabulary.load(
                Path(str(build["vocabulary"]["path"])), sha256=str(build["vocabulary"]["sha256"])
            )
            cached = (release, vocabulary)
            self._releases[release_id] = cached
        return cached

    def section_chunks(self, evidence: Evidence) -> list[ChunkRecord]:
        return self._repository.section(evidence.version_id, evidence.section)

    def candidates(
        self, query: str, paper_ids: Sequence[str] | None, release_id: str
    ) -> list[tuple[str, float]]:
        """Chunk ids by RRF of the BM25 and dense lists inside the papers (or everywhere)."""

        release, vocabulary = self._release(release_id)
        collection = release.collection(CHUNKS)
        query_filter = chunk_filter(paper_ids, include_references=self.config.include_references)
        limit = self.config.chunk_candidates
        rankings: list[list[str]] = []
        encoded = vocabulary.encode_query(query)
        if encoded.indices:
            sparse = models.SparseVector(indices=list(encoded.indices), values=list(encoded.values))
            hits = self.client.query_points(
                collection,
                query=sparse,
                using=SPARSE,
                query_filter=query_filter,
                limit=limit,
                with_payload=False,
            ).points
            rankings.append([str(hit.id) for hit in hits])
        prefix = str(getattr(self.embedder, "query_prefix", "") or "")
        vectors = self.embedder.encode_array([prefix + query], max_tokens=self.max_tokens)
        validate_vectors(vectors.vectors, self.embedder.dimensions)
        dense = [float(value) for value in vectors.vectors[0]]
        hits = self.client.query_points(
            collection,
            query=dense,
            using=DENSE,
            query_filter=query_filter,
            limit=limit,
            with_payload=False,
        ).points
        rankings.append([str(hit.id) for hit in hits])
        return rrf(rankings, self.config.rrf_k)[:limit]

    def retrieve(
        self, query: str, paper_ids: Sequence[str] | None, release_id: str
    ) -> list[Evidence]:
        """Evidence for a question inside ``paper_ids`` (every paper when None)."""

        config = self.config
        stats: dict[str, Any] = {
            "candidates": 0,
            "reranked": 0,
            "floor_dropped": 0,
            "cap_dropped": 0,
            "budget_dropped": 0,
            "abstract_only": 0,
            "expanded": 0,
        }
        self.last_stats = stats
        if paper_ids is not None and not paper_ids:
            return []
        ranked = self.candidates(query, paper_ids, release_id)
        stats["candidates"] = len(ranked)
        rows = self._repository.chunks([chunk_id for chunk_id, _ in ranked])
        ordered = [rows[chunk_id] for chunk_id, _ in ranked if chunk_id in rows]
        scores: dict[str, float] = {}
        if ordered and self.reranker is not None:
            values = validate_scores(
                self.reranker.score(query, [chunk.text for chunk in ordered]), len(ordered)
            )
            reranked = rerank_order([chunk.chunk_id for chunk in ordered], values)
            stats["reranked"] = len(reranked)
            kept_ids = []
            for chunk_id, value in reranked:
                if config.min_rerank_score is None or value > config.min_rerank_score:
                    kept_ids.append(chunk_id)
                    scores[chunk_id] = value
                else:
                    stats["floor_dropped"] += 1
            ordered = [rows[chunk_id] for chunk_id in kept_ids]
        chosen: list[ChunkRecord] = []
        per_paper: dict[str, int] = {}
        for chunk in ordered:
            if len(chosen) >= config.evidence:
                stats["cap_dropped"] += 1
                continue
            if per_paper.get(chunk.paper_id, 0) >= config.per_paper:
                stats["cap_dropped"] += 1
                continue
            per_paper[chunk.paper_id] = per_paper.get(chunk.paper_id, 0) + 1
            chosen.append(chunk)
        pieces = [
            Evidence(
                id=f"C{index}",
                source_kind="paper",
                source_id=UUID(chunk.paper_id),
                version_id=chunk.version_id,
                chunk_id=UUID(chunk.chunk_id),
                text=chunk.text,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
            for index, chunk in enumerate(chosen)
        ]
        # A named paper with no parsed text: its abstract, labelled, after the body evidence.
        if paper_ids is not None and len(pieces) < config.evidence:
            papers = self._repository.papers(list(paper_ids))
            for paper_id in paper_ids:
                row = papers.get(paper_id)
                if row is None or row["fulltext"] or not row["abstract"]:
                    continue
                if len(pieces) >= config.evidence:
                    break
                pieces.append(
                    Evidence(
                        id=f"A{paper_id}",
                        source_kind="paper",
                        source_id=UUID(paper_id),
                        version_id="",
                        chunk_id=None,
                        text=str(row["abstract"]),
                        section=ABSTRACT_SECTION,
                        abstract_only=True,
                    )
                )
                stats["abstract_only"] += 1
        if config.expansion == "section":
            widened = []
            for item in pieces:
                if item.chunk_id is None:
                    widened.append(item)
                    continue
                grown = expand_to_section(
                    item,
                    chunks=self.section_chunks(item),
                    max_chars=config.expansion_max_chars,
                    min_share=config.expansion_min_share,
                )
                stats["expanded"] += int(grown.expanded)
                widened.append(grown)
            pieces = widened
        kept, dropped = build_context(
            pieces, token_budget=config.token_budget, count_tokens=self._count
        )
        stats["budget_dropped"] = dropped
        return [item.model_copy(update={"id": f"E{index}"}) for index, item in enumerate(kept, 1)]


def evidence_by_id(evidence: Sequence[Evidence]) -> dict[str, Evidence]:
    return {item.id: item for item in evidence}
