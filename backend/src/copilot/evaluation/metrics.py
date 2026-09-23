"""Rank metrics, and the judged coverage that has to travel with them.

One rule shapes this module. A result the dataset never labelled is
*unjudged*, not irrelevant — scoring it as a miss understates a system against
a sparsely labelled benchmark, and quietly so. LitSearch labels one gold paper
per query out of 64,183; our own corpus holds many more papers that would
answer the same question. So ``evaluate`` returns coverage alongside every
score and the caller cannot obtain one without the other (spec §11).

Gains are graded 0–3 where the dataset supports it and binary where it does
not; ``ndcg_at_k`` handles both, because 2**1-1 == 1 makes a binary label a
graded label with one grade.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

MAX_GRADE = 3


class MetricError(ValueError):
    """A metric cannot be computed from these arguments."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One query's scores, with the coverage they were computed at."""

    ndcg_at_k: float
    recall_at_k: float
    mrr_at_k: float
    k: int
    returned: int
    judged: int
    judged_coverage: float | None


def _check_k(k: int) -> None:
    if k < 1:
        raise MetricError("invalid_k", str(k))


def _check_grades(grades: Mapping[str, int]) -> None:
    for item, grade in grades.items():
        if grade < 0 or grade > MAX_GRADE:
            raise MetricError("invalid_grade", f"{item}={grade}")


def _cut(ranked: Sequence[str], k: int) -> list[str]:
    """Deduplicate preserving order, then take the first k.

    A ranking that repeats an ID is a bug in the retriever, not a second
    opportunity to score the same paper.
    """

    return list(dict.fromkeys(ranked))[:k]


def _dcg(gains: Iterable[int]) -> float:
    total = 0.0
    for index, grade in enumerate(gains):
        total += (2**grade - 1) / math.log2(index + 2)
    return total


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Normalized discounted cumulative gain over graded or binary labels."""

    _check_k(k)
    _check_grades(grades)
    cut = _cut(ranked, k)
    ideal = sorted(grades.values(), reverse=True)[:k]
    denominator = _dcg(ideal)
    if not denominator:
        return 0.0
    return _dcg(grades.get(item, 0) for item in cut) / denominator


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the *known* relevant items retrieved by rank k.

    Recall against the judged pool, never against every paper that would have
    answered the query. The denominator is what the dataset labelled.
    """

    _check_k(k)
    wanted = set(relevant)
    if not wanted:
        raise MetricError("no_relevant")
    found = wanted.intersection(_cut(ranked, k))
    return len(found) / len(wanted)


def mrr_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Reciprocal rank of the first relevant item, 0.0 if none by rank k."""

    _check_k(k)
    wanted = set(relevant)
    if not wanted:
        raise MetricError("no_relevant")
    for index, item in enumerate(_cut(ranked, k), start=1):
        if item in wanted:
            return 1 / index
    return 0.0


def evaluate(*, ranked: Sequence[str], grades: Mapping[str, int], k: int) -> MetricResult:
    """Score one query and report the coverage the scores were computed at.

    ``judged_coverage`` is the share of returned results the dataset had an
    opinion about. It is ``None`` for an empty ranking — nothing was returned,
    so coverage is unknown rather than perfect.
    """

    _check_k(k)
    _check_grades(grades)
    cut = _cut(ranked, k)
    relevant = {item for item, grade in grades.items() if grade > 0}
    judged = sum(1 for item in cut if item in grades)
    return MetricResult(
        ndcg_at_k=ndcg_at_k(ranked, grades, k),
        recall_at_k=recall_at_k(ranked, relevant, k) if relevant else 0.0,
        mrr_at_k=mrr_at_k(ranked, relevant, k) if relevant else 0.0,
        k=k,
        returned=len(cut),
        judged=judged,
        judged_coverage=judged / len(cut) if cut else None,
    )
