"""The answering run on fixtures: split discipline, the budget stop, outputs and the smoke set."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq
import pytest

from copilot.contracts import Evidence
from copilot.evaluation.qa import (
    QA_SMOKE,
    QaError,
    check_qa_smoke,
    load_qa_experiment,
    run_qa_smoke,
    summarize_qa,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "configs" / "experiments" / "orb-text-qa.yaml"


def test_the_experiment_names_its_models_prompts_variants_and_rule():
    experiment = load_qa_experiment(EXPERIMENT)
    assert experiment.generator == "openai-gpt-6-luna"
    assert experiment.judge == "openai-gpt-6-luna-low"
    assert [v.name for v in experiment.variants] == ["flat", "section"]
    assert experiment.variant("section").evidence == {"expansion": "section"}
    assert experiment.budget_usd == 2.5
    assert (
        experiment.decision["mcnemar_p"] == 0.05
        and experiment.decision["cited_map_max_drop"] == 0.03
    )
    with pytest.raises(QaError, match="unknown_variant"):
        experiment.variant("deep")


def test_the_frozen_split_needs_the_flag(tmp_path):
    from copilot.evaluation.qa import run_qa

    with pytest.raises(QaError, match="frozen_flag_required"):
        run_qa(EXPERIMENT, "frozen", tmp_path, max_spend_usd=1.0, frozen=False)
    with pytest.raises(QaError, match="unknown_split"):
        run_qa(EXPERIMENT, "retrieval_only", tmp_path, max_spend_usd=1.0)


def test_the_smoke_run_scores_every_case_and_is_reproducible(tmp_path):
    first = run_qa_smoke(QA_SMOKE, out=tmp_path / "a")
    second = run_qa_smoke(QA_SMOKE, out=tmp_path / "b")
    assert first == second
    rows = pq.read_table(tmp_path / "a" / "smoke" / "cases.parquet").to_pylist()
    by_id = {row["query_id"]: row for row in rows}
    assert len(rows) == 6
    # s1: judged equivalent, gold cited first.
    assert by_id["s1"]["equivalent"] is True and by_id["s1"]["cited_map"] == 1.0
    # s2: the fake cites E1 (not gold) first, then E2 (gold): MAP 0.5; judged wrong.
    assert by_id["s2"]["cited_map"] == 0.5 and by_id["s2"]["equivalent"] is False
    # s4: a malformed verdict twice is a judge error, excluded from the denominator.
    assert by_id["s4"]["judge_error"] == "judge_error" and by_id["s4"]["equivalent"] is None
    # s5: no evidence is an abstention, scored wrong without a judge call.
    assert by_id["s5"]["abstained"] is True and by_id["s5"]["equivalent"] is False
    assert by_id["s5"]["judge_calls"] == 0
    # s6: the fabricated id was repaired: the answer counts, provenance does not.
    assert by_id["s6"]["repair_attempted"] is True and by_id["s6"]["provenance_valid"] is False
    assert by_id["s6"]["equivalent"] is True
    # The repository copy carries no question, reference or answer text.
    assert not {"question", "reference", "answer"} & set(rows[0])
    metrics = first["metrics"]
    assert metrics["cases"] == 6 and metrics["judged"] == 5 and metrics["judge_errors"] == 1
    assert metrics["correct"] == 3 and metrics["accuracy"]["mean"] == pytest.approx(0.6)
    assert metrics["abstention_rate"] == pytest.approx(1 / 6)
    assert metrics["provenance_validity"] == pytest.approx(5 / 6)
    assert 0 < metrics["accuracy"]["wilson"][0] < 0.6 < metrics["accuracy"]["wilson"][1] < 1
    assert metrics["cost_usd"] == 0.0
    # Per-case text goes under DATA_DIR-like storage the caller names, never under reports.
    assert (tmp_path / "a" / "text" / "cases.jsonl").exists()


def test_the_smoke_gate_matches_the_frozen_outputs():
    actual = run_qa_smoke(QA_SMOKE)
    expected = json.loads((QA_SMOKE / "expected.json").read_text())
    assert check_qa_smoke(actual, expected)["passed"], check_qa_smoke(actual, expected)


def test_summaries_bootstrap_over_families_and_count_what_they_say():
    rows = [
        {
            "query_id": "a",
            "family_id": "f1",
            "type": "extractive",
            "equivalent": True,
            "cited_map": 1.0,
            "provenance_valid": True,
            "abstained": False,
            "judge_error": None,
            "generation_cost_usd": 0.001,
            "judge_cost_usd": 0.0005,
        },
        {
            "query_id": "b",
            "family_id": "f1",
            "type": "abstractive",
            "equivalent": False,
            "cited_map": 0.0,
            "provenance_valid": False,
            "abstained": False,
            "judge_error": None,
            "generation_cost_usd": 0.001,
            "judge_cost_usd": 0.0005,
        },
        {
            "query_id": "c",
            "family_id": "f2",
            "type": "extractive",
            "equivalent": None,
            "cited_map": 1.0,
            "provenance_valid": True,
            "abstained": False,
            "judge_error": "judge_error",
            "generation_cost_usd": 0.001,
            "judge_cost_usd": 0.001,
        },
    ]
    metrics = summarize_qa(rows, resamples=50, seed=42, confidence=0.95)
    assert metrics["judged"] == 2 and metrics["correct"] == 1 and metrics["judge_errors"] == 1
    assert metrics["accuracy"]["mean"] == 0.5 and len(metrics["accuracy"]["interval"]) == 2
    assert metrics["cited_map"]["mean"] == pytest.approx(2 / 3)
    assert metrics["by_type"]["extractive"]["accuracy"] == 1.0
    assert metrics["cost_usd"] == pytest.approx(0.005)


def test_an_experiment_may_override_the_first_stage_by_the_harness_rule(tmp_path):
    from copilot.evaluation.qa import qa_search_config

    shipped = load_qa_experiment(EXPERIMENT)
    assert shipped.search_overrides == ()
    # No overrides: the shared file's value, `title_abstract_chunk` since 2026-10-21.
    assert qa_search_config(shipped).rerank_text == "title_abstract_chunk"
    c256 = load_qa_experiment(ROOT / "configs" / "experiments" / "orb-c256-qa.yaml")
    assert c256.release == "orb-c256"
    assert c256.search_overrides == (("rerank_text", "title_chunk"),)
    assert qa_search_config(c256).rerank_text == "title_chunk"
    # Only a knob the retrieval harness may turn is accepted.
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        EXPERIMENT.read_text(encoding="utf-8") + "search_overrides:\n  fusion: {rrf_k: 10}\n",
        encoding="utf-8",
    )
    with pytest.raises(QaError, match="experiment_invalid"):
        load_qa_experiment(bad)


def test_an_experiment_names_its_generator_client(tmp_path):
    from copilot.evaluation.qa import load_qa_experiment

    source = Path("configs/experiments/orb-text-qa-v2.yaml").read_text()
    shipped = tmp_path / "v2.yaml"
    shipped.write_text(source)
    assert load_qa_experiment(shipped).generator_client == "http"
    structured = tmp_path / "v3.yaml"
    structured.write_text(source + "\ngenerator_client: langchain\n")
    assert load_qa_experiment(structured).generator_client == "langchain"
    wrong = tmp_path / "bad.yaml"
    wrong.write_text(source + "\ngenerator_client: other\n")
    with pytest.raises(QaError, match="generator_client"):
        load_qa_experiment(wrong)


def test_the_overview_experiment_differs_from_the_chat_s_generator_only_in_its_prompt():
    """U2's rule compares two arms that search alike and answer alike but for the prompt."""

    from dataclasses import fields

    from copilot.evaluation.qa import qa_search_config
    from copilot.models.generation import FakeGenerator, load_generation_prompt

    base = load_qa_experiment(ROOT / "configs/experiments/orb-text-qa-v3-structured.yaml")
    overview = load_qa_experiment(ROOT / "configs/experiments/orb-text-qa-u2-overview.yaml")
    assert qa_search_config(overview) == qa_search_config(base)
    differing = {
        f.name
        for f in fields(base)
        if getattr(base, f.name) != getattr(overview, f.name)
        and f.name not in {"path", "sha256", "name", "search_overrides", "budget_usd"}
    }
    assert differing == {"generate_prompt"}
    prompt = load_generation_prompt(overview.generate_prompt)
    assert prompt.name == "grounded-overview-v1"
    # The same validation path: an invented id is repaired, and the answer cites real ids.
    evidence = [
        Evidence(
            id="E1",
            source_kind="paper",
            source_id=uuid4(),
            version_id="v",
            chunk_id=uuid4(),
            text="t",
            section="s",
        )
    ]
    outcome = FakeGenerator(prompt).generate("FABRICATE papers on graphs", evidence)
    assert outcome.repair_attempted and outcome.answer.claims[0].evidence_ids == ["E1"]
