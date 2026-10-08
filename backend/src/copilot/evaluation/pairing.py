"""Pairing two answering runs on identical cases, and calibrating the judge (ORB plan O5).

Exact McNemar on verdicts over the cases both runs judged, a sign test on the cited
MAP, the Wilson interval for an accuracy, and Cohen's kappa between the judge and
blind hand labels. Every test here is exact: no normal approximation at 400 cases.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class PairingError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def _binomial_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial test of ``k`` successes in ``n`` against one half."""

    if n == 0:
        return 1.0
    low = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar: ``b`` cases only the treated run got right, ``c`` only the baseline."""

    if b < 0 or c < 0:
        raise PairingError("invalid_counts")
    return _binomial_two_sided(b, b + c)


def smallest_rejected_split(n: int, alpha: float = 0.05) -> tuple[int, int] | None:
    """The most even (majority, minority) split of ``n`` discordant cases the test rejects."""

    best: tuple[int, int] | None = None
    for minority in range(n // 2 + 1):
        if _binomial_two_sided(minority, n) < alpha:
            best = (n - minority, minority)
    return best


def sign_test(differences: Sequence[float]) -> dict[str, Any]:
    positive = sum(1 for value in differences if value > 0)
    negative = sum(1 for value in differences if value < 0)
    ties = len(differences) - positive - negative
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "p": _binomial_two_sided(positive, positive + negative),
    }


def wilson(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float:
    if len(first) != len(second) or not first:
        raise PairingError("kappa_requires_pairs")
    n = len(first)
    agree = sum(1 for a, b in zip(first, second, strict=True) if a == b) / n
    yes_a = sum(first) / n
    yes_b = sum(second) / n
    chance = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    if chance == 1.0:
        return 1.0
    return (agree - chance) / (1 - chance)


def read_cases(path: str | Path) -> dict[str, dict[str, Any]]:
    """Per-case rows keyed by query id, from JSONL or the text-free Parquet."""

    target = Path(path)
    if target.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(target).to_pylist()
    else:
        rows = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["query_id"])
        if key in cases:
            raise PairingError("duplicate_case", key)
        cases[key] = row
    if not cases:
        raise PairingError("no_cases", str(target))
    return cases


@dataclass
class PairReport:
    treated: str
    baseline: str
    cases: int
    judged: int
    both: int
    neither: int
    treated_only: int
    baseline_only: int
    mcnemar_p: float
    smallest_rejected_split: tuple[int, int] | None
    accuracy: dict[str, float]
    cited_map: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def pair(treated: str | Path, baseline: str | Path) -> PairReport:
    """Exact McNemar on verdicts and a sign test on cited MAP over the common cases."""

    left = read_cases(treated)
    right = read_cases(baseline)
    common = sorted(set(left) & set(right))
    if not common:
        raise PairingError("no_common_cases")
    judged = [
        key
        for key in common
        if isinstance(left[key].get("equivalent"), bool)
        and isinstance(right[key].get("equivalent"), bool)
    ]
    both = sum(1 for key in judged if left[key]["equivalent"] and right[key]["equivalent"])
    neither = sum(
        1 for key in judged if not left[key]["equivalent"] and not right[key]["equivalent"]
    )
    b = sum(1 for key in judged if left[key]["equivalent"] and not right[key]["equivalent"])
    c = sum(1 for key in judged if right[key]["equivalent"] and not left[key]["equivalent"])
    maps = [
        (float(left[key].get("cited_map") or 0.0), float(right[key].get("cited_map") or 0.0))
        for key in common
    ]
    return PairReport(
        treated=str(treated),
        baseline=str(baseline),
        cases=len(common),
        judged=len(judged),
        both=both,
        neither=neither,
        treated_only=b,
        baseline_only=c,
        mcnemar_p=mcnemar_exact(b, c),
        smallest_rejected_split=smallest_rejected_split(b + c),
        accuracy={
            "treated": (both + b) / len(judged) if judged else 0.0,
            "baseline": (both + c) / len(judged) if judged else 0.0,
        },
        cited_map={
            "treated": sum(t for t, _ in maps) / len(maps),
            "baseline": sum(r for _, r in maps) / len(maps),
            "sign": sign_test([t - r for t, r in maps]),
        },
    )


def calibration(labels: str | Path, cases: str | Path, *, floor: float = 0.6) -> dict[str, Any]:
    """Agreement between blind hand labels and the judge over the cases both have."""

    human = read_cases(labels)
    judged = read_cases(cases)
    keys = sorted(
        key
        for key in human
        if key in judged
        and isinstance(human[key].get("equivalent"), bool)
        and isinstance(judged[key].get("equivalent"), bool)
    )
    if not keys:
        raise PairingError("no_common_cases")
    a = [bool(human[key]["equivalent"]) for key in keys]
    b = [bool(judged[key]["equivalent"]) for key in keys]
    kappa = cohen_kappa(a, b)
    return {
        "labels": len(human),
        "judged": len(keys),
        "agreement": sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(keys),
        "kappa": kappa,
        "kappa_floor": floor,
        "calibrated": kappa >= floor,
        "confusion": {
            "both_yes": sum(1 for x, y in zip(a, b, strict=True) if x and y),
            "human_yes_judge_no": sum(1 for x, y in zip(a, b, strict=True) if x and not y),
            "human_no_judge_yes": sum(1 for x, y in zip(a, b, strict=True) if not x and y),
            "both_no": sum(1 for x, y in zip(a, b, strict=True) if not x and not y),
        },
        "disagreements": [key for key, x, y in zip(keys, a, b, strict=True) if x != y],
    }
