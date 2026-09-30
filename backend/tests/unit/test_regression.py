"""Regression thresholds, run manifests, split discipline and the frozen smoke set."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from copilot.evaluation.datasets import read_dataset
from copilot.evaluation.regression import (
    REQUIRED_FIELDS,
    ManifestError,
    comparable,
    compare_runs,
    regressed,
    validate_manifest,
)
from copilot.evaluation.retrieval import (
    ExperimentError,
    QueryResult,
    Variant,
    bootstrap_interval,
    check_smoke,
    decide,
    load_experiment,
    run_query,
    run_retrieval,
    run_smoke,
    select_queries,
    smoke_experiment,
    summarize,
)

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / "data" / "fixtures" / "retrieval-smoke"
EXPERIMENT = ROOT / "configs" / "experiments" / "retrieval.yaml"


def test_absolute_threshold_and_boundary():
    assert regressed(0.70, 0.66)
    assert not regressed(0.70, 0.67)
    assert not regressed(0.01, 0.00)
    assert not regressed(0.60, 0.65)


def _manifest(**changes):
    manifest = {
        "run_id": "run-1",
        "created_at": "2026-09-30T00:00:00+00:00",
        "code": {"git_sha": "a" * 40, "dirty": False},
        "corpus": {
            "release_id": "rel-a",
            "manifest_sha256": "b" * 64,
            "parser_version": "pypdf-text-v2",
            "chunker_version": "paragraph-pack-v1",
        },
        "dataset": {"digest": "c" * 64, "split": "validation", "slice": "in_domain", "queries": 52},
        "experiment": {"config_sha256": "d" * 64},
        "search": {"config_sha256": "e" * 64},
        "seed": 42,
        "hardware": {"cpu": "cpu", "gpu": "none", "torch": "2.14"},
        "timing": {"methodology": "sequential"},
        "variants": {
            "bm25": {"mode": "bm25", "release_id": "rel-a", "embedding": "m@1", "reranker": "none"}
        },
    }
    for path, value in changes.items():
        target = manifest
        *parents, last = path.split(".")
        for part in parents:
            target = target[part]
        target[last] = value
    return manifest


def _drop(manifest, path):
    target = manifest
    *parents, last = path.split(".")
    for part in parents:
        target = target[part]
    del target[last]
    return manifest


def test_the_threshold_refuses_scores_outside_zero_to_one():
    with pytest.raises(ValueError, match="invalid_metric"):
        regressed(1.2, 0.5)
    with pytest.raises(ValueError, match="invalid_metric"):
        regressed(0.5, -0.1)


@pytest.mark.parametrize("path", REQUIRED_FIELDS)
def test_a_run_missing_any_manifest_field_is_rejected(path):
    validate_manifest(_manifest())
    with pytest.raises(ManifestError, match=f"manifest_incomplete:{path}"):
        validate_manifest(_drop(_manifest(), path))
    with pytest.raises(ManifestError, match="manifest_incomplete"):
        validate_manifest(_manifest(**{path: ""}))


def test_a_variant_without_its_model_versions_is_rejected():
    manifest = _manifest()
    del manifest["variants"]["bm25"]["embedding"]
    with pytest.raises(ManifestError, match="variants.bm25.embedding"):
        validate_manifest(manifest)
    with pytest.raises(ManifestError, match="manifest_incomplete:variants"):
        validate_manifest(_manifest(variants={}))


@pytest.mark.parametrize(
    ("path", "value", "what"),
    [
        ("corpus.release_id", "rel-b", "corpus"),
        ("corpus.manifest_sha256", "f" * 64, "corpus"),
        ("dataset.digest", "0" * 64, "dataset"),
        ("dataset.split", "development", "split"),
        ("dataset.slice", "all", "split"),
    ],
)
def test_runs_over_different_corpora_or_splits_are_not_compared(path, value, what):
    comparable(_manifest(), _manifest(run_id="run-2"))
    with pytest.raises(ManifestError, match=f"incomparable_runs:{what}"):
        comparable(_manifest(), _manifest(**{path: value}))


def _metrics_doc(manifest, recall, ndcg):
    summary = {"recall@10": {"mean": recall}, "ndcg@10": {"mean": ndcg}}
    return {"manifest": manifest, "variants": {"bm25": {"summary": summary}}}


def test_a_regression_report_flags_only_drops_beyond_the_threshold():
    baseline = _metrics_doc(_manifest(), 0.70, 0.60)
    held = compare_runs(baseline, _metrics_doc(_manifest(run_id="run-2"), 0.67, 0.62))
    assert held["regressed"] is False
    dropped = compare_runs(baseline, _metrics_doc(_manifest(run_id="run-3"), 0.66, 0.62))
    assert dropped["regressed"] is True
    assert dropped["variants"]["bm25"]["recall@10"]["regressed"] is True
    with pytest.raises(ManifestError, match="incomparable_runs"):
        compare_runs(baseline, _metrics_doc(_manifest(**{"dataset.split": "test"}), 0.7, 0.6))


def _result(name, family, metrics, *, failure=None, seconds=0.1):
    return QueryResult(
        variant=name,
        query_id=family,
        family_id=family,
        query_set="s",
        ranked=[],
        relevant=["x"],
        metrics={**metrics, "judged@10": None if failure else 0.1, "judged@50": None},
        candidate_recall=None,
        rerank_pool_recall=None,
        seconds=seconds,
        failure=failure,
    )


def _zeros(value=0.0):
    return {"recall@10": value, "recall@50": value, "ndcg@10": value, "mrr@10": value}


def test_a_failed_query_stays_in_every_denominator():
    experiment = smoke_experiment(SMOKE)
    results = [
        _result("bm25", "f1", _zeros(1.0)),
        _result("bm25", "f2", _zeros(1.0)),
        _result("bm25", "f3", _zeros(0.0), failure="candidates_unavailable", seconds=3.0),
    ]
    summary = summarize(results, experiment)
    assert summary["queries"] == 3
    assert summary["ndcg@10"]["mean"] == pytest.approx(2 / 3)
    assert summary["failures"] == {"count": 1, "codes": {"candidates_unavailable": 1}}
    assert summary["latency"]["max_ms"] == 3000.0
    assert summary["judged@10"]["unknown"] == 1


def test_a_query_whose_search_fails_is_scored_zero_not_dropped():
    class Down:
        def search(self, *args):
            raise ConnectionError("qdrant went away")

    from copilot.corpus.releases import ReleaseRecord
    from copilot.evaluation.retrieval import EvalQuery, InlineRunner
    from copilot.search.service import SearchConfig, SearchService

    release = ReleaseRecord("r", "0" * 64, "p", "c", "m", "ready", {})
    service = SearchService(
        engine=None,
        lexical=Down(),
        dense=Down(),
        reranker=None,
        config=SearchConfig(),
        runner=InlineRunner(),
        papers=_Papers(),
        capture=lambda: release,
    )
    query = EvalQuery("q1", "f1", "validation", "s", "graph learning", {"x": 1})
    result = run_query(
        service, Variant("hybrid", "hybrid"), query, release, smoke_experiment(SMOKE)
    )
    assert result.failure == "candidates_unavailable"
    assert result.metrics["ndcg@10"] == 0.0 and result.metrics["judged@10"] is None


class _Papers:
    def load(self, paper_ids):
        return {}


def test_the_smoke_set_is_reproducible_and_matches_its_frozen_outputs():
    first, second = run_smoke(SMOKE), run_smoke(SMOKE)
    assert first == second
    expected = json.loads((SMOKE / "expected.json").read_text(encoding="utf-8"))
    check = check_smoke(first, expected)
    assert check == {"passed": True, "regressions": [], "rankings_changed": []}


def test_the_smoke_gate_fails_a_drop_beyond_the_threshold():
    expected = json.loads((SMOKE / "expected.json").read_text(encoding="utf-8"))
    worse = copy.deepcopy(expected)
    worse["hybrid"]["metrics"]["ndcg@10"] -= 0.031
    worse["hybrid"]["rankings_sha256"] = "moved"
    check = check_smoke(worse, expected)
    assert check["passed"] is False
    assert check["regressions"][0]["variant"] == "hybrid"
    assert check["rankings_changed"] == ["hybrid"]


def test_the_bootstrap_is_fixed_by_its_seed():
    values = np.linspace(0, 1, 52)
    first = bootstrap_interval(values, resamples=1000, seed=42, confidence=0.95)
    assert first == bootstrap_interval(values, resamples=1000, seed=42, confidence=0.95)
    assert first != bootstrap_interval(values, resamples=1000, seed=7, confidence=0.95)
    assert first[0] < values.mean() < first[1]


@pytest.mark.parametrize("split", ["test", "development"])
def test_only_the_validation_split_may_choose(split):
    experiment = smoke_experiment(SMOKE)
    with pytest.raises(ExperimentError, match="held_out_split" if split == "test" else "not_the"):
        decide(split, {}, {}, experiment)


def test_the_test_split_does_not_run_without_the_lock(tmp_path):
    with pytest.raises(ExperimentError, match="locked_test_required"):
        run_retrieval(
            EXPERIMENT,
            "test",
            tmp_path,
            data_dir=tmp_path,
            database_url="postgresql+psycopg://unused:unused@127.0.0.1:1/none",
            qdrant_url="http://127.0.0.1:1",
        )


def _variant_results(name, scores):
    return [_result(name, f"f{i:02d}", _zeros(value)) for i, value in enumerate(scores)]


def _summaries(results, p95_seconds):
    return {name: {"latency": {"p95_ms": p95_seconds[name] * 1000}} for name in results}


def test_a_costlier_mode_is_promoted_only_on_a_supported_gain_within_budget():
    experiment = smoke_experiment(SMOKE)
    n = 40
    results = {
        "bm25": _variant_results("bm25", [0.5] * n),
        # A real, consistent gain.
        "dense": _variant_results("dense", [0.8] * n),
        # Better still, but too slow.
        "hybrid": _variant_results("hybrid", [0.9] * n),
        # No better than dense: noise around it.
        "hybrid_rerank": _variant_results(
            "hybrid_rerank", [0.8 + (0.1 if i % 2 else -0.1) for i in range(n)]
        ),
    }
    decision = decide(
        "validation",
        results,
        _summaries(results, {"bm25": 0.1, "dense": 0.1, "hybrid": 5.0, "hybrid_rerank": 0.5}),
        experiment,
    )
    steps = {(step["from"], step["to"]): step for step in decision["steps"]}
    assert steps[("bm25", "dense")]["promoted"] is True
    assert steps[("dense", "hybrid")]["gain_supported"] is True
    assert steps[("dense", "hybrid")]["promoted"] is False
    assert steps[("dense", "hybrid_rerank")]["promoted"] is False
    assert decision["chosen"]["variant"] == "dense"


def test_the_pilot_config_loads_and_its_ablations_change_one_thing():
    experiment = load_experiment(EXPERIMENT)
    by_name = {variant.name: variant for variant in experiment.variants}
    assert [v.name for v in experiment.variants][:4] == ["bm25", "dense", "hybrid", "hybrid_rerank"]
    for variant in experiment.variants:
        if variant.baseline:
            base = by_name[variant.baseline]
            assert variant.mode == base.mode
            changed = [
                field
                for field in ("models", "pair_max_tokens")
                if getattr(variant, field) != getattr(base, field)
            ]
            assert len(changed) == 1, variant.name


def test_a_config_naming_an_unknown_baseline_is_refused(tmp_path):
    text = EXPERIMENT.read_text(encoding="utf-8").replace("baseline: dense", "baseline: nothing")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ExperimentError, match="dense_bge_small.baseline"):
        load_experiment(path)


def test_the_in_domain_slice_is_the_frozen_150_52_48():
    dataset = read_dataset(ROOT / "data" / "fixtures" / "retrieval")
    counts = {
        split: len(select_queries(dataset, split, "in_domain"))
        for split in ("development", "validation", "test")
    }
    assert counts == {"development": 150, "validation": 52, "test": 48}
    everything = select_queries(dataset, "validation", "all")
    # Queries with no gold paper in our corpus have nothing to recall and are left out.
    assert all(query.relevant for query in everything)


def test_the_small_model_pin_puts_its_query_instruction_in_its_identity():
    from copilot.models.embeddings import load_embedding_spec

    large = load_embedding_spec(ROOT / "configs" / "models.yaml")
    small = load_embedding_spec(ROOT / "configs" / "experiments" / "models-bge-small.yaml")
    # Unchanged, so the active release's recorded model revision still matches.
    assert large.identity == "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181#cls-l2-v1"
    assert small.query_prefix.startswith("Represent this sentence")
    assert small.identity.startswith("BAAI/bge-small-en-v1.5@5c38ec7c")
    assert "+query:" in small.identity and small.dimensions == 384


def test_the_dense_branch_prefixes_queries_and_only_queries():
    from copilot.corpus.releases import ReleaseRecord
    from copilot.models.embeddings import EncodedBatch
    from copilot.search.dense import DenseRetriever

    class Instructed:
        identity = "fixture/instructed"
        dimensions = 2
        query_prefix = "Represent: "

        def __init__(self) -> None:
            self.seen: list[str] = []

        def encode_array(self, texts, *, max_tokens=None):
            self.seen.extend(texts)
            return EncodedBatch(np.array([[1.0, 0.0]] * len(texts), dtype=np.float32), 0)

    class Client:
        def query_points(self, *args, **kwargs):
            return type("Result", (), {"points": []})()

    model = Instructed()
    retriever = DenseRetriever(None, Client(), model)
    retriever._releases["r"] = ReleaseRecord("r", "0" * 64, "p", "c", model.identity, "ready", {})
    retriever.search("graph learning", None, 10, "r")
    assert model.seen == ["Represent: graph learning"]
