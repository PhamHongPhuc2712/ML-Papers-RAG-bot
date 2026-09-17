"""Measure two chunking policies over the same real documents.

The question a chunker has to answer is not "what size are the chunks" but
"where do they cut". A window that expires mid-sentence hands the retriever half
a claim and the reader a fragment, so the metric that matters here is how often
a chunk begins or ends in the middle of a sentence.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .chunk import ChunkerConfig, Section, TokenSpans, chunk_sections
from .chunk_paragraph import pack_paragraphs, split_paragraphs
from .parse import ParseStatus, parse_pdf_result

_ENDS = (".", "!", "?", '."', ".)", ".”", '?"', '!"')
_OPENS = "\"'(“["
PROSE_KINDS = frozenset({"body", "appendix", "abstract"})


def ends_mid_sentence(text: str) -> bool:
    return not text.rstrip().endswith(_ENDS)


def starts_mid_sentence(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    return not (stripped[0].isupper() or stripped[0].isdigit() or stripped[0] in _OPENS)


def _summarize(chunks: Sequence[dict[str, Any]], papers: int) -> dict[str, Any]:
    tokens = [int(c["token_count"]) for c in chunks]
    prose = [c for c in chunks if c["kind"] in PROSE_KINDS]
    prose_tokens = [int(c["token_count"]) for c in prose] or [0]
    ordered = sorted(tokens) or [0]
    return {
        "chunks": len(chunks),
        "chunks_per_paper": round(len(chunks) / papers, 1) if papers else 0.0,
        "token_median": statistics.median(ordered),
        "token_mean": round(statistics.mean(ordered), 1),
        "token_p90": ordered[int(0.9 * (len(ordered) - 1))],
        "token_max": max(ordered),
        "prose_chunks": len(prose),
        "prose_token_median": statistics.median(sorted(prose_tokens)),
        "pct_ending_mid_sentence": round(
            100 * sum(1 for c in prose if ends_mid_sentence(str(c["text"]))) / max(1, len(prose)), 1
        ),
        "pct_starting_mid_sentence": round(
            100 * sum(1 for c in prose if starts_mid_sentence(str(c["text"]))) / max(1, len(prose)),
            1,
        ),
    }


def compare_policies(
    pdfs: Sequence[Path],
    spans: TokenSpans,
    chunker: ChunkerConfig,
    *,
    max_pdf_bytes: int,
    sample: int = 40,
    seed: int = 2026,
) -> dict[str, Any]:
    """Chunk the same documents both ways and report where each policy cuts."""

    chosen = list(pdfs)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:sample]

    fixed: list[dict[str, Any]] = []
    packed: list[dict[str, Any]] = []
    paragraphs: list[int] = []
    papers = 0
    failures = 0
    for pdf in chosen:
        result = parse_pdf_result(pdf, max_bytes=max_pdf_bytes)
        if result.status is not ParseStatus.PARSED:
            failures += 1
            continue
        papers += 1
        prose: list[Section] = [s for s in result.sections if s.kind in PROSE_KINDS]
        paragraphs.append(sum(len(split_paragraphs(s.text)) for s in prose))
        fixed.extend(
            chunk_sections(
                result.sections,
                spans,
                target=chunker.target_tokens,
                overlap=chunker.overlap_tokens,
                hard_cap=chunker.hard_cap_tokens,
            )
        )
        packed.extend(
            pack_paragraphs(
                result.sections,
                spans,
                target_tokens=chunker.target_tokens,
                max_tokens=chunker.max_tokens,
                paragraph_max_tokens=chunker.paragraph_max_tokens,
                overlap_sentences=chunker.overlap_sentences,
            )
        )
    return {
        "papers": papers,
        "parse_failures": failures,
        "seed": seed,
        "paragraphs_per_paper_median": statistics.median(sorted(paragraphs)) if paragraphs else 0,
        "fixed_window": _summarize(fixed, papers),
        "paragraph_pack": _summarize(packed, papers),
    }
