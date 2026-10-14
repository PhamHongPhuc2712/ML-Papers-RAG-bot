"""The retrieval harness end to end, over the seeded test corpus and its release."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import SecretStr
from sqlalchemy import text

from copilot.corpus.export import export_snapshot
from copilot.db.session import assert_safe_test_database
from copilot.evaluation.datasets import Dataset as FrozenDataset
from copilot.evaluation.datasets import QrelRecord, QueryRecord, SplitRecord, write_dataset
from copilot.evaluation.regression import validate_manifest
from copilot.evaluation.retrieval import ExperimentError, run_retrieval
from copilot.models.embeddings import FixtureEmbedding
from copilot.models.llm import ChatClient
from copilot.models.spend import SpendLedger, load_llm_config
from copilot.search.index import (
    CHUNKS,
    PAPERS,
    CollectionSettings,
    IndexConfig,
    build_index,
    collection_names,
)
from copilot.search.llm_rerank import LlmListwiseReranker, load_prompt
from copilot.search.rerank import FixtureReranker

pytestmark = pytest.mark.integration

# A second release of the seeded corpus with its paper collection only.
PAPERS_ONLY = "search-a-papers-only"
# (query id, split, query set, specificity, text, gold titles in the seeded corpus). The
# seeded chunks hold each abstract, and BM25 does not stem: queries use the abstracts' words.
QUERIES = (
    (
        "q-0001",
        "development",
        "set_a",
        1,
        "message passing molecular graphs",
        ("Graph neural networks for molecules",),
    ),
    (
        "q-0002",
        "development",
        "set_b",
        0,
        "sentence pairs contrasted",
        ("Contrastive sentence embeddings",),
    ),
    (
        "q-0003",
        "development",
        "set_c",
        1,
        "retrieve passages then generate",
        ("Retrieval augmented generation",),
    ),
    (
        "q-0004",
        "development",
        "set_d",
        0,
        "attention over graph nodes on molecular graphs",
        ("Graph transformers at scale", "Graph neural networks for molecules"),
    ),
    (
        "q-0005",
        "validation",
        "set_a",
        1,
        "denoising generates images",
        ("Diffusion models for image synthesis",),
    ),
    (
        "q-0006",
        "validation",
        "set_b",
        0,
        "loss falls as a power law",
        ("Scaling laws for language models",),
    ),
)
SOURCE = "fixture@seeded-corpus"


@dataclass
class Bench:
    root: Path
    config: Path
    data_dir: Path


def _write_dataset(directory: Path, data_dir: Path, ids: dict[str, str]) -> None:
    """The frozen format: ids and labels in the dataset, text under DATA_DIR."""

    queries = tuple(
        QueryRecord(
            query_id=qid,
            query=query,
            query_set=qset,
            workflow="known_item",
            specificity=spec,
            quality=2,
            in_domain=True,
            source=SOURCE,
            author="fixture",
        )
        for qid, _, qset, spec, query, _ in QUERIES
    )
    qrels = tuple(
        QrelRecord(query_id=qid, corpusid=ids[title], grade=1, paper_id=ids[title])
        for qid, _, _, _, _, golds in QUERIES
        for title in golds
    )
    splits = tuple(
        SplitRecord(query_id=qid, family_id=qid, split=split) for qid, split, *_ in QUERIES
    )
    write_dataset(FrozenDataset(queries, qrels, splits), directory, include_query_text=False)
    text_file = data_dir / "benchmarks" / "fixture" / "queries.jsonl"
    text_file.parent.mkdir(parents=True)
    text_file.write_text(
        "".join(json.dumps({"query_id": q.query_id, "query": q.query}) + "\n" for q in queries)
    )


LLM_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "openai_chat.json"


def _reversing_llm(calls: list[str], *, completion_tokens: int):
    """A stand-in OpenAI endpoint: it answers every head in reverse, and counts the calls."""

    template = json.loads(LLM_FIXTURE.read_text())

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload["model"])
        count = int(re.search(r"Rank all (\d+) candidates", payload["messages"][1]["content"])[1])
        body = copy.deepcopy(template)
        body["model"] = "fixture-llm"
        body["choices"][0]["message"]["content"] = " > ".join(
            f"[{number}]" for number in range(count, 0, -1)
        )
        body["usage"] = {
            "prompt_tokens": 900,
            "completion_tokens": completion_tokens,
            "total_tokens": 900 + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        return httpx.Response(200, json=body)

    return handler


class FixtureModels:
    """The run's model factory with fixture models; one path names a mismatched embedder."""

    def __init__(self, *, completion_tokens: int = 30) -> None:
        self.llm_calls: list[str] = []
        self.completion_tokens = completion_tokens

    def embedder(self, models: Path):
        if models.name == "broken.yaml":
            return FixtureEmbedding(identity="fixture/not-the-release-model"), None
        return FixtureEmbedding(), None

    def reranker(self, models: Path):
        return FixtureReranker()

    def listwise(self, model_key: str, ledger, cache):
        config = load_llm_config("configs/llm.yaml")
        model = config.models[model_key]
        client = ChatClient(
            model,
            config.providers[model.provider],
            SecretStr("sk-test-fixture"),
            ledger,
            transport=httpx.MockTransport(
                _reversing_llm(self.llm_calls, completion_tokens=self.completion_tokens)
            ),
            cache=cache,
            sleep=lambda _seconds: None,
        )
        return LlmListwiseReranker(client, load_prompt("prompts/rerank/listwise-v1.yaml"))


@pytest.fixture(scope="module")
def bench(search_corpus, test_settings, tmp_path_factory) -> Iterator[Bench]:
    assert_safe_test_database(test_settings)
    root = tmp_path_factory.mktemp("eval")
    data_dir = root / "data"
    _write_dataset(root / "dataset", data_dir, search_corpus.ids)
    # The paper-only release, for the chunk-level refusal: same papers, no chunk collection.
    snapshot = root / "exports" / PAPERS_ONLY
    export_snapshot(PAPERS_ONLY, snapshot, engine=search_corpus.engine)
    build_index(
        snapshot / "manifest.json",
        FixtureEmbedding(),
        engine=search_corpus.engine,
        client=search_corpus.client,
        prefix=test_settings.qdrant_collection_prefix,
        data_dir=data_dir,
        config=IndexConfig(
            collections={
                PAPERS: CollectionSettings(max_tokens=64),
                CHUNKS: CollectionSettings(max_tokens=64),
            },
            batch_points=4,
            canary_queries=2,
            canary_limit=3,
            canary_terms=3,
        ),
        kinds=[PAPERS],
        release_id=PAPERS_ONLY,
    )
    config = root / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "eval-mini",
                "dataset": {
                    "path": str(root / "dataset"),
                    "query_text": "benchmarks/fixture/queries.jsonl",
                    "slice": "all",
                    "labels": "paper_id",
                },
                "corpus": {"release": search_corpus.release, "papers": "database"},
                "search": "configs/search.yaml",
                "models": str(root / "fixture.yaml"),
                "metrics": {"recall_at": [10, 50], "ndcg_at": [10], "mrr_at": [10], "depth": 50},
                "bootstrap": {"resamples": 1000, "seed": 42, "confidence": 0.95},
                "variants": [
                    {"name": "bm25", "mode": "bm25"},
                    {"name": "dense", "mode": "dense"},
                    {"name": "hybrid", "mode": "hybrid"},
                    {"name": "hybrid_rerank", "mode": "hybrid_rerank"},
                    # Deliberately broken: an embedder the release was not built with.
                    {
                        "name": "dense_broken",
                        "mode": "dense",
                        "baseline": "dense",
                        "changes": "an embedder the release was not built with",
                        "models": str(root / "broken.yaml"),
                    },
                ],
                "decision": {
                    "primary": "ndcg@10",
                    "order": ["bm25", "dense", "hybrid", "hybrid_rerank"],
                },
            }
        )
    )
    try:
        yield Bench(root=root, config=config, data_dir=data_dir)
    finally:
        with search_corpus.engine.begin() as connection:
            connection.execute(
                text("delete from corpus_releases where id = :id"), {"id": PAPERS_ONLY}
            )
        for name in collection_names(test_settings.qdrant_collection_prefix, PAPERS_ONLY).values():
            if search_corpus.client.collection_exists(name):
                search_corpus.client.delete_collection(name)


def _run(bench: Bench, test_settings, out: Path, split: str = "development"):
    return run_retrieval(
        bench.config,
        split,
        out,
        data_dir=bench.data_dir,
        database_url=test_settings.database_url,
        qdrant_url=test_settings.qdrant_url,
        models=FixtureModels(),
    )


def test_a_run_writes_a_complete_manifest(bench, search_corpus, test_settings, tmp_path):
    metrics = _run(bench, test_settings, tmp_path)
    manifest = metrics["manifest"]
    validate_manifest(manifest)
    assert manifest["corpus"]["release_id"] == search_corpus.release
    assert manifest["corpus"]["papers"] == "database"
    assert manifest["dataset"]["labels"] == "paper_id"
    assert manifest["dataset"]["sources"] == [SOURCE]
    assert manifest["dataset"]["queries"] == 4
    summary = metrics["variants"]["bm25"]["summary"]
    # Exact BM25 finds every development gold by its title words.
    assert summary["recall@10"]["mean"] == 1.0
    assert set(summary["by_specificity"]) == {"0", "1"}
    assert set(summary["by_query_set"]) == {"set_a", "set_b", "set_c", "set_d"}
    rows = pq.read_table(tmp_path / "development" / "per_query.parquet").to_pylist()
    assert {row["specificity"] for row in rows} == {0, 1}


def test_a_rerun_reproduces_every_metric_and_ranking(bench, test_settings, tmp_path):
    first = _run(bench, test_settings, tmp_path / "a")
    second = _run(bench, test_settings, tmp_path / "b")
    for name, entry in first["variants"].items():
        for metric in ("recall@10", "recall@50", "ndcg@10", "mrr@10"):
            assert entry["summary"][metric] == second["variants"][name]["summary"][metric]

    def rankings(directory):
        rows = pq.read_table(directory / "development" / "per_query.parquet").to_pylist()
        return [(row["variant"], row["query_id"], row["ranked"]) for row in rows]

    assert rankings(tmp_path / "a") == rankings(tmp_path / "b")


def test_a_broken_baseline_is_a_counted_failure_not_a_quiet_zero(bench, test_settings, tmp_path):
    metrics = _run(bench, test_settings, tmp_path)
    broken = metrics["variants"]["dense_broken"]["summary"]
    assert broken["queries"] == 4
    assert broken["failures"] == {"count": 4, "codes": {"candidates_unavailable": 4}}
    assert broken["ndcg@10"]["mean"] == 0.0
    rows = pq.read_table(tmp_path / "development" / "per_query.parquet").to_pylist()
    failed = [row for row in rows if row["variant"] == "dense_broken"]
    assert len(failed) == 4 and all(row["failure"] == "candidates_unavailable" for row in failed)
    assert metrics["variants"]["dense"]["summary"]["failures"]["count"] == 0


def test_the_validation_run_decides_and_the_test_split_stays_locked(bench, test_settings, tmp_path):
    metrics = _run(bench, test_settings, tmp_path, split="validation")
    assert metrics["decision"]["chosen"]["variant"] in {"bm25", "dense", "hybrid", "hybrid_rerank"}
    from copilot.evaluation.retrieval import ExperimentError

    with pytest.raises(ExperimentError, match="locked_test_required"):
        _run(bench, test_settings, tmp_path, split="test")


def test_a_variant_overriding_search_knobs_runs_with_them_and_records_them(
    bench, test_settings, tmp_path
):
    """P2.6: rerank depth, candidates per branch and the rerank deadline, per variant."""

    config = yaml.safe_load(bench.config.read_text())
    config["variants"] = [
        {"name": "hybrid", "mode": "hybrid"},
        {
            "name": "hybrid_pool2",
            "mode": "hybrid",
            "baseline": "hybrid",
            "changes": "candidates per branch 100 -> 2",
            "search": {"candidates": {"per_branch": 2}},
        },
        {"name": "hybrid_rerank", "mode": "hybrid_rerank"},
        {
            "name": "hybrid_rerank_depth3",
            "mode": "hybrid_rerank",
            "baseline": "hybrid_rerank",
            "changes": "rerank depth 50 -> 3, rerank deadline 1.5 s -> 2.5 s",
            "search": {"rerank": {"depth": 3}, "deadlines_seconds": {"rerank": 2.5}},
        },
    ]
    config["decision"]["order"] = ["hybrid", "hybrid_rerank"]
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump(config))
    metrics = run_retrieval(
        path,
        "development",
        tmp_path / "out",
        data_dir=bench.data_dir,
        database_url=test_settings.database_url,
        qdrant_url=test_settings.qdrant_url,
        models=FixtureModels(),
    )
    variants = metrics["manifest"]["variants"]
    assert variants["hybrid"]["search"]["candidates_per_branch"] == 100
    assert variants["hybrid_pool2"]["search"]["candidates_per_branch"] == 2
    assert variants["hybrid_pool2"]["search_overrides"] == {"candidates_per_branch": 2}
    deeper = variants["hybrid_rerank_depth3"]["search"]
    assert (deeper["rerank_depth"], deeper["rerank_seconds"]) == (3, 2.5)
    # Same knobs, same digest; any override, its own digest.
    assert variants["hybrid"]["search_sha256"] == variants["hybrid_rerank"]["search_sha256"]
    assert len({entry["search_sha256"] for entry in variants.values()}) == 3
    # The run-level record still names the file every variant started from.
    assert metrics["manifest"]["search"]["values"]["rerank_depth"] == 50

    rows = pq.read_table(tmp_path / "out" / "development" / "per_query.parquet").to_pylist()

    def longest(name):
        return max(len(row["ranked"]) for row in rows if row["variant"] == name)

    # Two candidates from each branch fuse to at most four; the corpus has eight.
    assert longest("hybrid_pool2") <= 4 < longest("hybrid")
    assert longest("hybrid_rerank_depth3") == 3 < longest("hybrid_rerank")


# --- LLM reranking variants (LLM reranking plan, L5) ---------------------------------------


@pytest.fixture
def llm_config(bench, migrated_database, tmp_path) -> Path:
    with migrated_database.begin() as connection:
        connection.execute(text("delete from llm_calls"))
    config = yaml.safe_load(bench.config.read_text())
    config["variants"] = [
        {"name": "hybrid_rerank", "mode": "hybrid_rerank"},
        {
            "name": "deep",
            "mode": "hybrid_rerank_llm",
            "llm": "openai-gpt-6-luna",
            "changes": "the cross-encoder head reordered by a listwise LLM",
        },
    ]
    config["decision"]["order"] = ["hybrid_rerank", "deep"]
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _run_llm(path, bench, test_settings, out, models, *, max_spend=1.0, cache=None):
    return run_retrieval(
        path,
        "development",
        out,
        data_dir=bench.data_dir,
        database_url=test_settings.database_url,
        qdrant_url=test_settings.qdrant_url,
        models=models,
        max_spend_usd=max_spend,
        llm_daily_cap_usd=Decimal("5.00"),
        llm_cache=cache,
    )


def test_an_llm_variant_records_its_model_prompt_and_real_cost(
    llm_config, bench, test_settings, migrated_database, tmp_path
):
    models = FixtureModels()
    metrics = _run_llm(
        llm_config, bench, test_settings, tmp_path / "a", models, cache=tmp_path / "cache"
    )
    manifest = metrics["manifest"]
    validate_manifest(manifest)
    llm = manifest["variants"]["deep"]["llm"]
    assert llm["model_key"] == "openai-gpt-6-luna" and llm["model"] == "gpt-6-luna"
    assert llm["served_models"] == ["fixture-llm"]
    assert llm["prompt_sha256"] == load_prompt("prompts/rerank/listwise-v1.yaml").sha256
    assert llm["words"] == 300
    # Four development queries, one call each: the warm-up never reached the LLM.
    assert llm["calls"] == 4 and len(models.llm_calls) == 4 and llm["cache_hits"] == 0
    assert llm["input_tokens"] == 4 * 900 and llm["output_tokens"] == 4 * 30
    ledger = SpendLedger(migrated_database, daily_cap_usd=Decimal("5.00"))
    assert llm["cost_usd"] > 0
    assert llm["cost_usd"] == pytest.approx(ledger.run_spend(llm["ledger_run_id"]))
    assert manifest["cost"]["metered_usd"] == pytest.approx(llm["cost_usd"])
    assert "hybrid_rerank" not in manifest["cost"]["by_variant"]

    rows = pq.read_table(tmp_path / "a" / "development" / "per_query.parquet").to_pylist()
    ranked = {(row["variant"], row["query_id"]): row["ranked"] for row in rows}
    development = [query_id for query_id, split, *_ in QUERIES if split == "development"]
    for query_id in development:
        assert ranked[("deep", query_id)] == ranked[("hybrid_rerank", query_id)][::-1]

    from copilot.evaluation.report import render_report

    report = render_report(tmp_path / "a").read_text()
    assert "| LLM cost USD | LLM calls (cached) |" in report
    assert f"| {llm['cost_usd']:.4f} | 4 (0) |" in report
    assert "USD metered (development " in report


def test_a_rerun_over_the_same_cache_is_free_and_identical(
    llm_config, bench, test_settings, tmp_path
):
    models = FixtureModels()
    _run_llm(llm_config, bench, test_settings, tmp_path / "a", models, cache=tmp_path / "cache")
    again = _run_llm(
        llm_config, bench, test_settings, tmp_path / "b", models, cache=tmp_path / "cache"
    )
    llm = again["manifest"]["variants"]["deep"]["llm"]
    assert llm["cache_hits"] == llm["calls"] == 4 and len(models.llm_calls) == 4
    assert llm["cost_usd"] == 0.0 and again["manifest"]["cost"]["metered_usd"] == 0.0

    def rankings(directory):
        rows = pq.read_table(directory / "development" / "per_query.parquet").to_pylist()
        return [(row["variant"], row["query_id"], row["ranked"]) for row in rows]

    assert rankings(tmp_path / "a") == rankings(tmp_path / "b")


def test_a_budget_below_the_preflight_refuses_before_any_query(
    llm_config, bench, test_settings, tmp_path
):
    models = FixtureModels()
    with pytest.raises(ExperimentError, match="run_budget_exceeded"):
        _run_llm(llm_config, bench, test_settings, tmp_path / "out", models, max_spend=1e-6)
    assert models.llm_calls == [] and not (tmp_path / "out").exists()
    with pytest.raises(ExperimentError, match="run_budget_required"):
        _run_llm(llm_config, bench, test_settings, tmp_path / "out", models, max_spend=None)


def test_spend_past_the_budget_stops_the_run_and_writes_nothing(
    llm_config, bench, test_settings, tmp_path
):
    # Each answer reports 400,000 output tokens, $0.20 at luna's price: far past its
    # reservation, so only the mid-run guard can stop it.
    models = FixtureModels(completion_tokens=400_000)
    with pytest.raises(ExperimentError, match="run_budget_exceeded"):
        _run_llm(
            llm_config,
            bench,
            test_settings,
            tmp_path / "out",
            models,
            max_spend=0.05,
            cache=tmp_path / "cache",
        )
    assert len(models.llm_calls) == 1 and not (tmp_path / "out").exists()


def test_the_guard_sums_every_llm_variant_in_a_run(
    bench, test_settings, migrated_database, tmp_path
):
    with migrated_database.begin() as connection:
        connection.execute(text("delete from llm_calls"))
    config = yaml.safe_load(bench.config.read_text())
    config["variants"] = [
        {"name": "hybrid_rerank", "mode": "hybrid_rerank"},
        {"name": "deep", "mode": "hybrid_rerank_llm", "llm": "openai-gpt-6-luna"},
        {"name": "deep_low", "mode": "hybrid_rerank_llm", "llm": "openai-gpt-6-luna-low"},
    ]
    config["decision"]["order"] = ["hybrid_rerank", "deep", "deep_low"]
    path = tmp_path / "two.yaml"
    path.write_text(yaml.safe_dump(config))
    # Each answer bills 40,000 output tokens, about $0.0201, so four queries cost about
    # $0.080 per variant. A $0.12 budget passes the preflight and the first variant, and
    # the second variant's second query takes the run past it.
    models = FixtureModels(completion_tokens=40_000)
    with pytest.raises(ExperimentError, match="run_budget_exceeded"):
        _run_llm(
            path,
            bench,
            test_settings,
            tmp_path / "out",
            models,
            max_spend=0.12,
            cache=tmp_path / "cache",
        )
    assert len(models.llm_calls) == 4 + 2 and not (tmp_path / "out").exists()


def test_a_chunk_level_variant_is_refused_when_the_release_has_no_chunk_collection(
    bench, test_settings, tmp_path
):
    """P2.6 step 4: a release built with its paper collection only cannot answer through chunks."""

    config = yaml.safe_load(bench.config.read_text())
    config["corpus"]["release"] = PAPERS_ONLY
    config["variants"] = [
        {"name": "hybrid", "mode": "hybrid"},
        {"name": "hybrid_chunks", "mode": "hybrid", "candidates": "chunks"},
    ]
    config["decision"]["order"] = ["hybrid"]
    path = tmp_path / "chunks.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ExperimentError, match="chunks_not_built"):
        run_retrieval(
            path,
            "development",
            tmp_path / "out",
            data_dir=bench.data_dir,
            database_url=test_settings.database_url,
            qdrant_url=test_settings.qdrant_url,
            models=FixtureModels(),
        )
    assert not (tmp_path / "out").exists()
