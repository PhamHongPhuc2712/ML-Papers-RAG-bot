"""Reciprocal rank fusion, computed here rather than inside the vector store (spec §7).

``sum(1/(k+rank))`` over one-based ranks, with k = 60. It is implemented in
application code so the constant is ours and the result reproducible: a
server's built-in fusion may use another constant, normalize scores, or change
between versions. A list that repeats an ID counts it once, at its best rank,
and ties sort by ID so the same inputs always give the same order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

RRF_K = 60


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse rankings by reciprocal rank; highest first, ties broken by ID."""

    if k < 1:
        raise ValueError("invalid_rrf_k")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
