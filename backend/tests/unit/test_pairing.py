"""Exact pairing of two answering runs, the sign test, Wilson intervals and judge calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.evaluation.pairing import (
    PairingError,
    calibration,
    cohen_kappa,
    mcnemar_exact,
    pair,
    sign_test,
    smallest_rejected_split,
    wilson,
)


def test_mcnemar_exact_matches_hand_computed_cases():
    assert mcnemar_exact(8, 2) == pytest.approx(0.109, abs=0.001)
    assert mcnemar_exact(10, 1) == pytest.approx(0.012, abs=0.001)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0


def test_the_smallest_discordance_split_the_test_would_reject():
    # Of 11 discordant cases, a 10-1 split is rejected at 0.05 and 9-2 is not.
    assert smallest_rejected_split(11) == (10, 1)
    assert mcnemar_exact(9, 2) > 0.05
    assert smallest_rejected_split(3) is None


def test_the_sign_test_counts_only_nonzero_differences():
    result = sign_test([0.5, 0.2, 0.0, -0.1, 0.3, 0.0])
    assert (result["positive"], result["negative"], result["ties"]) == (3, 1, 2)
    assert result["p"] == pytest.approx(0.625, abs=0.001)


def test_wilson_interval_for_an_accuracy():
    low, high = wilson(90, 100)
    assert 0.82 < low < 0.84 and 0.94 < high < 0.95
    assert wilson(0, 0) == (0.0, 1.0)


def _cases(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pair_joins_on_query_id_and_reports_both_tests(tmp_path):
    baseline = _cases(
        tmp_path / "flat.jsonl",
        [
            {"query_id": "a", "equivalent": True, "cited_map": 1.0},
            {"query_id": "b", "equivalent": False, "cited_map": 0.0},
            {"query_id": "c", "equivalent": False, "cited_map": 0.5},
            {"query_id": "d", "equivalent": True, "cited_map": 1.0},
            {"query_id": "e", "equivalent": None, "cited_map": 1.0},
        ],
    )
    treated = _cases(
        tmp_path / "section.jsonl",
        [
            {"query_id": "a", "equivalent": True, "cited_map": 1.0},
            {"query_id": "b", "equivalent": True, "cited_map": 1.0},
            {"query_id": "c", "equivalent": True, "cited_map": 0.5},
            {"query_id": "d", "equivalent": False, "cited_map": 0.5},
            {"query_id": "e", "equivalent": True, "cited_map": 1.0},
        ],
    )
    report = pair(treated, baseline)
    # The judge-error case `e` is left out of the verdict pairing, not scored as a loss.
    assert report.cases == 5 and report.judged == 4
    assert (report.treated_only, report.baseline_only) == (2, 1)
    assert report.mcnemar_p == pytest.approx(mcnemar_exact(2, 1))
    assert report.accuracy == {"treated": 0.75, "baseline": 0.5}
    assert report.cited_map["treated"] == pytest.approx(0.8)
    assert report.cited_map["sign"]["positive"] == 1 and report.cited_map["sign"]["negative"] == 1


def test_pair_refuses_disjoint_files_and_duplicate_keys(tmp_path):
    a = _cases(tmp_path / "a.jsonl", [{"query_id": "x", "equivalent": True, "cited_map": 1.0}])
    b = _cases(tmp_path / "b.jsonl", [{"query_id": "y", "equivalent": True, "cited_map": 1.0}])
    with pytest.raises(PairingError, match="no_common_cases"):
        pair(a, b)
    dup = _cases(
        tmp_path / "dup.jsonl",
        [
            {"query_id": "x", "equivalent": True, "cited_map": 1.0},
            {"query_id": "x", "equivalent": False, "cited_map": 0.0},
        ],
    )
    with pytest.raises(PairingError, match="duplicate_case"):
        pair(dup, a)


def test_cohen_kappa_and_calibration_against_hand_labels(tmp_path):
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    labels = _cases(
        tmp_path / "labels.jsonl",
        [
            {"query_id": q, "equivalent": v}
            for q, v in zip("abcde", [True, True, False, False, True], strict=False)
        ],
    )
    cases = _cases(
        tmp_path / "cases.jsonl",
        [
            {"query_id": q, "equivalent": v, "cited_map": 1.0}
            for q, v in zip("abcdef", [True, False, False, False, True, True], strict=False)
        ],
    )
    report = calibration(labels, cases)
    assert report["labels"] == 5 and report["judged"] == 5
    assert report["agreement"] == pytest.approx(0.8)
    assert report["confusion"] == {
        "both_yes": 2,
        "human_yes_judge_no": 1,
        "human_no_judge_yes": 0,
        "both_no": 2,
    }
    assert 0.5 < report["kappa"] < 0.7
    assert report["calibrated"] is (report["kappa"] >= 0.6)
