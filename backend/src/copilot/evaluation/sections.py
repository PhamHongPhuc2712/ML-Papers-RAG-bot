"""Locate a benchmark's gold section inside our chunks (ORB plan, O3).

Open RAG Bench labels each query with a gold document *and* a gold section, in
its own Mistral-OCR text. Our chunks come from pypdf over the same PDF, so the
section has to be found again: a chunk belongs to the gold section when a run of
the section's sentences appears in it. Three consecutive sentences, or 40% of the
section's sentences when that is fewer, is the bar; a single shared sentence is
not, because abstracts and introductions repeat each other.

What comes out is ``section_hit@1``: of the queries whose gold section mapped to
at least one chunk, the share whose rank-1 paper is the gold paper *and* whose
best chunk in some branch lies in that section. It is always reported with its
coverage, the share of queries that mapped at all; a section that did not map
is unknown, never a miss.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..search.lexical import tokenize

MIN_RUN = 3
MIN_SHARE = 0.4
# A "sentence" of fewer tokens is a heading or a label; one of more is a table row,
# a formula dump or a base64 blob. Neither locates anything.
MIN_SENTENCE_TOKENS = 3
MAX_SENTENCE_TOKENS = 120
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ChunkText:
    chunk_id: str
    text: str


def normalized_sentences(text: str) -> list[str]:
    """Sentences as the lexical tokenizer sees them, headings and blobs dropped."""

    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for sentence in _SENTENCE_END.split(stripped):
            tokens = tokenize(sentence)
            if MIN_SENTENCE_TOKENS <= len(tokens) <= MAX_SENTENCE_TOKENS:
                out.append(" ".join(tokens))
    return out


def required_run(sentence_count: int) -> int:
    """The run of consecutive sentences a chunk must contain to belong to the section."""

    if sentence_count <= 0:
        return 0
    return max(1, min(MIN_RUN, math.ceil(MIN_SHARE * sentence_count)))


def _longest_run(sentences: Sequence[str], chunk_normalized: str) -> int:
    haystack = f" {chunk_normalized} "
    best = run = 0
    for sentence in sentences:
        if f" {sentence} " in haystack:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def locate_section(section_text: str, chunks: Sequence[ChunkText]) -> set[str] | None:
    """The chunk ids the gold section overlaps, or ``None`` when it is not found."""

    sentences = normalized_sentences(section_text)
    needed = required_run(len(sentences))
    if needed == 0:
        return None
    found = {
        chunk.chunk_id
        for chunk in chunks
        if _longest_run(sentences, " ".join(tokenize(chunk.text))) >= needed
    }
    return found or None


def section_hit(best_chunk_ids: Iterable[str], gold_chunks: set[str] | None) -> bool | None:
    """Whether any best chunk lies in the gold section; ``None`` when the section is unmapped."""

    if gold_chunks is None:
        return None
    return any(chunk_id in gold_chunks for chunk_id in best_chunk_ids)


# --- Over a recorded run ------------------------------------------------------------------


def chunks_for_paper(engine: Any, paper_id: str) -> list[ChunkText]:
    """The parsed version's chunks, in order."""

    from uuid import UUID

    from sqlalchemy import select

    from ..db.models import Chunk, PaperVersion
    from ..db.session import session_scope

    with session_scope(engine) as session:
        rows = session.execute(
            select(Chunk.id, Chunk.text)
            .join(PaperVersion, PaperVersion.id == Chunk.paper_version_id)
            .where(PaperVersion.paper_id == UUID(paper_id), PaperVersion.parse_status == "parsed")
            .order_by(Chunk.ordinal)
        ).all()
    return [ChunkText(str(chunk_id), str(text)) for chunk_id, text in rows]


def gold_chunks(
    sections: Mapping[str, tuple[str, int]],
    paper_ids: Mapping[str, str],
    section_text: Mapping[tuple[str, int], str],
    engine: Any,
) -> dict[str, set[str] | None]:
    """Query id to the gold section's chunk ids, one lookup per (document, section)."""

    located: dict[tuple[str, int], set[str] | None] = {}
    chunks_cache: dict[str, list[ChunkText]] = {}
    out: dict[str, set[str] | None] = {}
    for query_id, (doc_id, section_id) in sections.items():
        key = (doc_id, section_id)
        if key not in located:
            paper_id = paper_ids.get(doc_id)
            text = section_text.get(key)
            if paper_id is None or text is None:
                located[key] = None
            else:
                if paper_id not in chunks_cache:
                    chunks_cache[paper_id] = chunks_for_paper(engine, paper_id)
                located[key] = locate_section(text, chunks_cache[paper_id])
        out[query_id] = located[key]
    return out


def score_run(
    rows: Iterable[Mapping[str, Any]],
    golds: Mapping[str, set[str] | None],
    gold_paper: Mapping[str, str],
) -> dict[str, Any]:
    """Per variant: coverage, paper@1 and section_hit@1, with a facet by query set."""

    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    out: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in items:
            facets = ("all", f"query_set:{row['query_set']}")
            query_id = str(row["query_id"])
            ranked = list(row["ranked"] or [])
            gold = golds.get(query_id)
            top_is_gold = bool(ranked) and ranked[0] == gold_paper.get(query_id)
            best = json.loads(row["best_chunks"] or "{}") if "best_chunks" in row else {}
            hit = (
                section_hit(best.values(), gold)
                if top_is_gold
                else (None if gold is None else False)
            )
            for facet in facets:
                counts[facet]["queries"] += 1
                counts[facet]["paper_at_1"] += int(top_is_gold)
                if gold is not None:
                    counts[facet]["mapped"] += 1
                    counts[facet]["section_hit"] += int(bool(hit))
                    counts[facet]["paper_at_1_mapped"] += int(top_is_gold)
        out[variant] = {
            facet: {
                "queries": c["queries"],
                "mapped": c["mapped"],
                "coverage": c["mapped"] / c["queries"] if c["queries"] else None,
                "paper_at_1": c["paper_at_1"] / c["queries"] if c["queries"] else None,
                "section_hit_at_1": c["section_hit"] / c["mapped"] if c["mapped"] else None,
                "section_hit_given_paper_at_1": (
                    c["section_hit"] / c["paper_at_1_mapped"] if c["paper_at_1_mapped"] else None
                ),
            }
            for facet, c in sorted(counts.items())
        }
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def render_sections(scores: Mapping[str, Any], *, split: str, mapped_pairs: int, pairs: int) -> str:
    lines = [
        f"# Gold sections — {split}",
        "",
        "Rendered by `eval orb-sections` from the recorded `per_query.parquet`; nothing was "
        "rerun. A gold section maps to our chunks when a chunk holds a run of its sentences "
        f"(at least {MIN_RUN}, or {int(MIN_SHARE * 100)}% of the section when that is fewer); "
        f"{mapped_pairs} of {pairs} distinct (document, section) pairs mapped. `section_hit@1` "
        "is over mapped queries only: the rank-1 paper is the gold paper and its best chunk in "
        "some branch lies in the gold section. `paper@1` is over every query.",
        "",
        "| Variant | Facet | Queries | Mapped | Coverage | paper@1 | section_hit@1 | "
        "section hit given paper@1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant, facets in scores.items():
        for facet, cell in facets.items():
            lines.append(
                f"| `{variant}` | {facet} | {cell['queries']} | {cell['mapped']} | "
                f"{_pct(cell['coverage'])} | {_pct(cell['paper_at_1'])} | "
                f"{_pct(cell['section_hit_at_1'])} | {_pct(cell['section_hit_given_paper_at_1'])} |"
            )
    return "\n".join(lines) + "\n"


def orb_sections(
    run: Path,
    *,
    split: str,
    dataset: Path,
    corpus_dir: Path,
    engine: Any,
) -> Path:
    """Score a recorded run's rank-1 hits against the gold sections; writes beside the run."""

    import pyarrow.parquet as pq

    from .datasets import read_dataset
    from .orb import read_sections

    sections = read_sections(dataset)
    frozen = read_dataset(dataset)
    gold_paper = {
        record.query_id: str(record.paper_id) for record in frozen.qrels if record.paper_id
    }
    paper_ids = {
        record.corpusid: str(record.paper_id) for record in frozen.qrels if record.paper_id
    }
    pairs = sorted({pair for pair in sections.values()})
    section_text: dict[tuple[str, int], str] = {}
    for doc_id, section_id in pairs:
        path = corpus_dir / f"{doc_id}.json"
        if not path.exists():
            continue
        paper = json.loads(path.read_text(encoding="utf-8"))
        for section in paper.get("sections") or []:
            if int(section.get("section_id", -1)) == section_id:
                section_text[(doc_id, section_id)] = str(section.get("text") or "")
                break
    golds = gold_chunks(sections, paper_ids, section_text, engine)
    rows = pq.read_table(run / split / "per_query.parquet").to_pylist()
    scores = score_run(rows, golds, gold_paper)
    mapped_pairs = sum(
        1 for pair in pairs if any(golds.get(q) for q, p in sections.items() if p == pair)
    )
    payload = {
        "run": str(run),
        "split": split,
        "rule": {"min_run": MIN_RUN, "min_share": MIN_SHARE},
        "pairs": len(pairs),
        "mapped_pairs": mapped_pairs,
        "variants": scores,
    }
    (run / "sections.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out = run / "sections.md"
    out.write_text(
        render_sections(scores, split=split, mapped_pairs=mapped_pairs, pairs=len(pairs))
    )
    return out
