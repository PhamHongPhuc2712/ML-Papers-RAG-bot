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
    # Why it failed travels with the row: each branch's error, never silently dropped.
    assert result.warnings == ["lexical:ConnectionError,dense:ConnectionError"]


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


def test_a_degraded_candidate_is_never_promoted_when_the_rule_requires_a_clean_run():
    from dataclasses import replace

    n = 40
    results = {
        "bm25": _variant_results("bm25", [0.5] * n),
        "dense": _variant_results("dense", [0.8] * n),
    }
    summaries = _summaries(results, {"bm25": 0.1, "dense": 0.1})
    # One query fell back to an earlier stage: the run measured a timeout, not the mode.
    summaries["dense"]["degraded"] = {"count": 1, "warnings": {"rerank_timeout": 1}}
    rules = {"primary": "ndcg@10", "order": ["bm25", "dense"], "require_undegraded": True}
    strict = replace(smoke_experiment(SMOKE), decision=rules)
    step = decide("validation", results, summaries, strict)["steps"][0]
    assert (step["gain_supported"], step["clean"], step["promoted"]) == (True, False, False)
    assert decide("validation", results, summaries, strict)["chosen"]["variant"] == "bm25"
    # The rule is opt-in, so earlier experiments decide exactly as they did.
    lenient = replace(strict, decision={**rules, "require_undegraded": False})
    assert decide("validation", results, summaries, lenient)["chosen"]["variant"] == "dense"
    summaries["dense"]["degraded"] = {"count": 0, "warnings": {}}
    assert decide("validation", results, summaries, strict)["chosen"]["variant"] == "dense"
    # A degraded baseline flatters the candidate, so it blocks the step too.
    summaries["bm25"]["degraded"] = {"count": 2, "warnings": {"dense_timeout": 2}}
    assert decide("validation", results, summaries, strict)["steps"][0]["clean"] is False


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


# --- P2.6: variants that change how candidates are gathered and reranked ---------------


def _config_with(tmp_path, variants):
    import yaml

    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    raw["variants"] = [{"name": "hybrid", "mode": "hybrid"}, *variants]
    raw["decision"]["order"] = ["hybrid"]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_a_variant_may_override_the_three_knobs_p26_names(tmp_path):
    path = _config_with(
        tmp_path,
        [
            {
                "name": "pool200",
                "mode": "hybrid",
                "baseline": "hybrid",
                "search": {"candidates": {"per_branch": 200}},
            },
            {
                "name": "depth100",
                "mode": "hybrid_rerank",
                "search": {"rerank": {"depth": 100}, "deadlines_seconds": {"rerank": 2.5}},
            },
        ],
    )
    by_name = {variant.name: variant for variant in load_experiment(path).variants}
    assert by_name["hybrid"].search == ()
    assert by_name["pool200"].search == (("candidates_per_branch", 200),)
    assert dict(by_name["depth100"].search) == {"rerank_depth": 100, "rerank_seconds": 2.5}


@pytest.mark.parametrize(
    ("mode", "search", "why"),
    [
        ("hybrid", {"fusion": {"rrf_k": 30}}, "search.fusion.rrf_k"),
        ("hybrid", {"rerank": {"depth": 100}}, "search.rerank.depth"),
        ("dense", {"deadlines_seconds": {"rerank": 2.0}}, "search.deadlines_seconds.rerank"),
        ("hybrid", {"candidates": {"per_branch": 0}}, "search.candidates.per_branch"),
        ("hybrid", {"candidates": {"per_branch": 150.5}}, "search.candidates.per_branch"),
        ("hybrid_rerank", {"deadlines_seconds": {"rerank": True}}, "search.deadlines_seconds"),
        ("hybrid", {"candidates": 200}, "search.candidates"),
    ],
)
def test_an_override_outside_those_knobs_or_their_modes_is_refused(tmp_path, mode, search, why):
    path = _config_with(tmp_path, [{"name": "changed", "mode": mode, "search": search}])
    with pytest.raises(ExperimentError, match=f"changed.{why}"):
        load_experiment(path)


def test_an_override_replaces_only_its_own_values_and_must_fit_the_total_deadline():
    from copilot.evaluation.retrieval import search_config_for
    from copilot.search.service import SearchConfig

    base = SearchConfig()
    deeper = Variant(
        name="deeper",
        mode="hybrid_rerank",
        search=(("rerank_depth", 100), ("rerank_seconds", 2.5)),
    )
    config = search_config_for(deeper, base)
    assert (config.rerank_depth, config.rerank_seconds) == (100, 2.5)
    assert (config.candidates_per_branch, config.total_seconds) == (100, 3.0)
    assert search_config_for(Variant(name="same", mode="hybrid"), base) == base
    late = Variant(name="late", mode="hybrid_rerank", search=(("rerank_seconds", 3.5),))
    with pytest.raises(ExperimentError, match="late.*total"):
        search_config_for(late, base)


def test_a_comparison_says_when_a_variant_searched_differently():
    baseline = _metrics_doc(_manifest(), 0.70, 0.60)
    same = compare_runs(baseline, _metrics_doc(_manifest(run_id="run-2"), 0.70, 0.60))
    assert same["search_changed"] == []
    deeper = _manifest(run_id="run-3", **{"variants.bm25.search": {"candidates_per_branch": 200}})
    changed = compare_runs(baseline, _metrics_doc(deeper, 0.70, 0.60))
    assert changed["search_changed"] == ["bm25"]


# --- LLM reranking variants (LLM reranking plan, L5) ---------------------------------------


@pytest.mark.parametrize(
    ("variant", "why"),
    [
        ({"name": "deep", "mode": "hybrid_rerank_llm"}, "deep.llm"),
        ({"name": "deep", "mode": "hybrid_rerank", "llm": "openai-gpt-6-luna"}, "deep.llm"),
        ({"name": "deep", "mode": "hybrid_rerank_llm", "llm": "no-such-model"}, "deep.llm"),
    ],
)
def test_an_llm_variant_names_a_configured_model_and_only_the_llm_mode_may(tmp_path, variant, why):
    with pytest.raises(ExperimentError, match=why):
        load_experiment(_config_with(tmp_path, [variant]))


def test_an_llm_variant_loads_with_its_model_and_may_rerank_deeper(tmp_path):
    path = _config_with(
        tmp_path,
        [
            {
                "name": "deep",
                "mode": "hybrid_rerank_llm",
                "llm": "openai-gpt-6-luna",
                "search": {"rerank": {"depth": 100}, "deadlines_seconds": {"rerank": 2.5}},
            }
        ],
    )
    deep = load_experiment(path).variant("deep")
    assert deep.llm == "openai-gpt-6-luna"
    assert deep.search == (("rerank_depth", 100), ("rerank_seconds", 2.5))


def test_the_preflight_prices_every_query_at_its_worst_case():
    from copilot.evaluation.retrieval import llm_preflight_usd
    from copilot.models.spend import estimate_usd, load_llm_config
    from copilot.search.service import SearchConfig

    model = load_llm_config("configs/llm.yaml").models["openai-gpt-6-luna"]
    deep = Variant(name="deep", mode="hybrid_rerank_llm", llm="openai-gpt-6-luna")
    plain = Variant(name="plain", mode="hybrid_rerank")
    configs = {"deep": SearchConfig(rerank_depth=100), "plain": SearchConfig()}
    worst = estimate_usd(model.price, 300 * 7 * 100, model.max_output_tokens)
    assert llm_preflight_usd([deep, plain], configs, queries=359) == pytest.approx(359 * worst)
    assert llm_preflight_usd([plain], configs, queries=359) == 0.0


def test_a_comparison_says_when_a_variant_s_llm_moved():
    llm = {"served_models": ["gpt-6-luna"], "prompt_sha256": "a" * 64}
    baseline = _metrics_doc(_manifest(**{"variants.bm25.llm": llm}), 0.70, 0.60)
    same = compare_runs(
        baseline, _metrics_doc(_manifest(run_id="run-2", **{"variants.bm25.llm": llm}), 0.70, 0.60)
    )
    assert same["llm_changed"] == []
    moved = {**llm, "served_models": ["gpt-6-luna-2026-11-01"]}
    changed = compare_runs(
        baseline,
        _metrics_doc(_manifest(run_id="run-3", **{"variants.bm25.llm": moved}), 0.70, 0.60),
    )
    assert changed["llm_changed"] == ["bm25"]
    plain = compare_runs(
        _metrics_doc(_manifest(), 0.70, 0.60), _metrics_doc(_manifest(run_id="r"), 0.70, 0.60)
    )
    assert plain["llm_changed"] == []


# --- Candidate source (P2.6 step 4) --------------------------------------------------------


def test_a_variant_may_find_its_candidates_through_chunks(tmp_path):
    path = _config_with(
        tmp_path,
        [
            {"name": "hybrid_chunks", "mode": "hybrid", "candidates": "chunks"},
            {"name": "hybrid_both", "mode": "hybrid_rerank", "candidates": "both"},
        ],
    )
    experiment = load_experiment(path)
    by_name = {variant.name: variant for variant in experiment.variants}
    assert by_name["hybrid"].candidates is None
    assert by_name["hybrid_chunks"].candidates == "chunks"
    assert by_name["hybrid_both"].candidates == "both"
    # Unset follows the shared file over our corpus, and stays at paper level over a
    # packaged snapshot corpus, which was never chunked.
    from dataclasses import replace

    from copilot.evaluation.retrieval import candidate_source
    from copilot.search.service import SearchConfig, load_search_config

    shipped = load_search_config("configs/search.yaml")
    assert shipped.candidates_source == "chunks"
    assert candidate_source(experiment, by_name["hybrid"], shipped) == "chunks"
    assert candidate_source(experiment, by_name["hybrid"], SearchConfig()) == "papers"
    snapshot = replace(experiment, papers="snapshot")
    assert candidate_source(snapshot, by_name["hybrid"], shipped) == "papers"
    assert candidate_source(snapshot, by_name["hybrid_chunks"], shipped) == "chunks"


def test_a_candidate_source_outside_the_three_is_refused(tmp_path):
    path = _config_with(tmp_path, [{"name": "odd", "mode": "hybrid", "candidates": "sections"}])
    with pytest.raises(ExperimentError, match="odd.candidates"):
        load_experiment(path)


def test_the_shipped_candidate_source_leaves_every_recorded_digest_unchanged():
    from copilot.evaluation.retrieval import search_digest, search_values
    from copilot.search.service import SearchConfig

    base = SearchConfig()
    assert search_values(base, "hybrid") == search_values(base, "hybrid", "papers")
    assert "candidates_source" not in search_values(base, "hybrid")
    assert search_digest(base, "hybrid") == search_digest(base, "hybrid", "papers")
    # Another source is part of what was searched: recorded, and its own digest.
    assert search_values(base, "hybrid", "chunks")["candidates_source"] == "chunks"
    digests = {search_digest(base, "hybrid", source) for source in ("papers", "chunks", "both")}
    assert len(digests) == 3


# --- Guards on a decision step (P2.6 step 4) ------------------------------------------------


def _mixed_results(name, primary, guarded):
    """Per-family rows where recall@50 is ``primary`` and ndcg@10 is ``guarded``."""

    return [
        _result(name, f"f{i:02d}", {**_zeros(p), "ndcg@10": g})
        for i, (p, g) in enumerate(zip(primary, guarded, strict=True))
    ]


def test_a_guard_blocks_a_recall_gain_that_loses_the_top_of_the_page():
    from dataclasses import replace

    n = 40
    rules = {
        "primary": "recall@50",
        "order": ["hybrid", "hybrid_rerank"],
        "guards": [{"metric": "ndcg@10", "max_drop": 0.03}],
    }
    experiment = replace(smoke_experiment(SMOKE), decision=rules)
    base = _mixed_results("hybrid", [0.6] * n, [0.5] * n)
    # Clearly more recall, clearly worse nDCG: the guard must refuse it.
    worse_top = _mixed_results("hybrid_rerank", [0.8] * n, [0.4] * n)
    summaries = _summaries(
        {"hybrid": base, "hybrid_rerank": worse_top},
        {
            "hybrid": 0.5,
            "hybrid_rerank": 0.6,
        },
    )
    decision = decide(
        "validation", {"hybrid": base, "hybrid_rerank": worse_top}, summaries, experiment
    )
    step = decision["steps"][0]
    assert step["gain_supported"] is True
    assert step["guards"][0]["metric"] == "ndcg@10"
    assert step["guards"][0]["holds"] is False
    assert (step["guards_hold"], step["promoted"]) == (False, False)
    assert decision["chosen"]["variant"] == "hybrid"
    # The same recall gain with the top of the page held within the guard is promoted.
    held_top = _mixed_results(
        "hybrid_rerank", [0.8] * n, [0.5 + (0.01 if i % 2 else -0.01) for i in range(n)]
    )
    decision = decide(
        "validation", {"hybrid": base, "hybrid_rerank": held_top}, summaries, experiment
    )
    assert decision["steps"][0]["guards_hold"] is True
    assert decision["chosen"]["variant"] == "hybrid_rerank"
    # Without guards the rule decides exactly as before.
    lenient = replace(experiment, decision={k: v for k, v in rules.items() if k != "guards"})
    decision = decide(
        "validation", {"hybrid": base, "hybrid_rerank": worse_top}, summaries, lenient
    )
    assert decision["steps"][0]["guards"] == []
    assert decision["chosen"]["variant"] == "hybrid_rerank"


@pytest.mark.parametrize(
    "guards",
    [[{"metric": "precision@5"}], [{"metric": "ndcg@10", "max_drop": 0}], ["ndcg@10"]],
)
def test_a_guard_must_name_a_measured_metric_and_a_positive_drop(tmp_path, guards):
    import yaml

    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    raw["decision"]["guards"] = guards
    path = tmp_path / "guarded.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentError, match="decision.guards"):
        load_experiment(path)


def test_the_first_stage_config_pre_registers_a_guarded_recall_rule():
    experiment = load_experiment(ROOT / "configs" / "experiments" / "m2-first-stage.yaml")
    assert experiment.decision["primary"] == "recall@50"
    assert experiment.decision["order"] == [
        "hybrid_rerank",
        "hybrid_rerank_chunks",
        "hybrid_rerank_both",
    ]
    assert experiment.decision["guards"] == [{"metric": "ndcg@10", "max_drop": 0.03}]
    assert experiment.decision["require_undegraded"] is True
    by_name = {variant.name: variant for variant in experiment.variants}
    assert by_name["hybrid_rerank_both"].candidates == "both"
    assert by_name["hybrid_rerank_both"].baseline is None


def test_the_search_file_names_its_candidate_source_and_refuses_an_unknown_one(tmp_path):
    import yaml

    from copilot.search.service import load_search_config

    raw = yaml.safe_load(Path("configs/search.yaml").read_text(encoding="utf-8"))
    raw["candidates"]["source"] = "sections"
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="candidates.source"):
        load_search_config(path)
    del raw["candidates"]["source"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_search_config(path).candidates_source == "papers"
