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


# --- The research agent as a variant (agent experiment, 2026-10-25) ------------------------

AGENT_EXPERIMENT = ROOT / "configs" / "experiments" / "orb-text-qa-agent.yaml"


class _Scripted:
    identity = "scripted"

    def __init__(self, replies) -> None:
        self.replies = list(replies)

    def step(self, messages, tools, *, request_id=None, run_id=None):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 0.001


def _call(name, args, call_id="c1"):
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _gold_evidence(paper_id: str) -> Evidence:
    return Evidence(
        id="E1",
        source_kind="paper",
        source_id=paper_id,
        version_id="v1",
        chunk_id=uuid4(),
        text="The gold passage.",
        section="Method",
    )


class _FixedJudge:
    identity = "fixed"

    def judge(self, question, reference, answer, *, request_id=None, run_id=None):
        from copilot.evaluation.judge import JudgeOutcome, JudgeVerdict

        return JudgeOutcome(JudgeVerdict(reason="same", equivalent=True), calls=1, cost_usd=0.0002)


def _agent_case(model):
    from copilot.agent.tools import AgentLimits
    from copilot.agent.turn import AgentTurn, load_agent_prompt
    from copilot.chat.retriever import CorpusRetriever, PaperHits
    from copilot.evaluation.qa import QaCase
    from copilot.models.generation import FakeGenerator

    gold = str(uuid4())
    turn = AgentTurn(
        model,
        CorpusRetriever(
            search=lambda query: PaperHits("rel", [gold, str(uuid4())], []),
            evidence=lambda query, papers, release: [_gold_evidence(gold)],
        ),
        lambda item: [],
        FakeGenerator(),
        load_agent_prompt(ROOT / "prompts" / "agent" / "research-v1.yaml"),
        limits=AgentLimits(),
        max_chars=6000,
        token_budget=6000,
        count_tokens=lambda text: len(text.split()),
    )
    case = QaCase("q1", "f1", "extractive", "What is gold?", "Gold.", gold)
    return case, turn


def test_an_agent_case_is_scored_like_a_pipeline_case_plus_its_trace():
    from copilot.evaluation.qa import QaVariant, answer_case_agent

    case, turn = _agent_case(
        _Scripted(
            [
                _call("search_papers", {"query": "gold"}),
                _call("answer", {"question": "What is gold?", "evidence_ids": ["S1E1"]}, "c2"),
            ]
        )
    )
    result = answer_case_agent(
        case, turn, _FixedJudge(), QaVariant(name="agent", answerer="agent"), run_id="r"
    )
    row = result.row()
    assert row["variant"] == "agent" and row["evidence_ids"] == ["E1"]
    assert row["gold_in_papers"] is True and row["cited_map"] == 1.0 and row["equivalent"] is True
    assert row["agent_searches"] == 1 and row["agent_model_calls"] == 2
    assert row["agent_stop"] == "answered" and row["agent_chosen_ids"] == ["S1E1"]
    # The agent's two steps and the answer, together.
    assert row["generation_cost_usd"] == pytest.approx(0.002)
    assert row["agent_cost_usd"] == pytest.approx(0.002)
    # The plan's size in the repository's row; queries and plan text only in the DATA_DIR copy.
    assert row["agent_plan_parts"] == 0 and "agent_text" not in row
    assert result.text_row()["agent_text"] == {"queries": ["gold"], "plan": []}


def test_a_spend_cap_inside_an_agent_case_is_a_typed_failure():
    from copilot.evaluation.qa import QaVariant, answer_case_agent
    from copilot.models.llm import LlmError

    case, turn = _agent_case(_Scripted([LlmError("llm_spend_cap")]))
    result = answer_case_agent(
        case, turn, _FixedJudge(), QaVariant(name="agent", answerer="agent"), run_id="r"
    )
    assert result.failure == "agent:llm_spend_cap" and result.equivalent is None


def test_a_pipeline_row_has_no_agent_columns(tmp_path):
    run_qa_smoke(QA_SMOKE, out=tmp_path)
    rows = pq.read_table(tmp_path / "smoke" / "cases.parquet").to_pylist()
    assert rows and not any(key.startswith("agent_") for key in rows[0])


def test_the_agent_experiment_loads_with_both_variants():
    experiment = load_qa_experiment(AGENT_EXPERIMENT)
    assert [(v.name, v.answerer) for v in experiment.variants] == [
        ("flat", "pipeline"),
        ("agent", "agent"),
    ]
    assert experiment.variant("agent").agent["max_searches"] == 3
    assert experiment.generator_client == "langchain" and experiment.search_overrides == ()


def test_an_unknown_answerer_is_refused(tmp_path):
    text = AGENT_EXPERIMENT.read_text().replace("answerer: agent", "answerer: oracle")
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(QaError, match="answerer"):
        load_qa_experiment(path)


def test_the_preflight_adds_the_agent_s_steps():
    from copilot.evaluation.qa import qa_preflight_usd

    experiment = load_qa_experiment(AGENT_EXPERIMENT)
    plain = qa_preflight_usd(experiment, cases=10, variants=2, evidence_tokens=6000)
    with_agent = qa_preflight_usd(
        experiment, cases=10, variants=2, evidence_tokens=6000, agents=[{"max_model_calls": 8}]
    )
    assert with_agent > plain


def test_the_research_v3_orb_guard_differs_from_the_scored_one_only_in_the_prompt():
    scored = load_qa_experiment(ROOT / "configs" / "experiments" / "orb-text-qa-agent.yaml")
    guard = load_qa_experiment(ROOT / "configs" / "experiments" / "orb-text-qa-agent-v3.yaml")
    assert guard.name == "orb-text-qa-agent-v3"
    assert guard.decision == scored.decision
    old, new = (
        dict(next(v for v in e.variants if v.name == "agent").agent) for e in (scored, guard)
    )
    assert old.pop("prompt").endswith("research-v2.yaml")
    assert new.pop("prompt").endswith("research-v3.yaml")
    assert old == new


def test_a_checked_agent_reserves_four_more_calls():
    from copilot.agent.tools import AgentLimits
    from copilot.evaluation.qa import agent_turn_worst_usd
    from copilot.models.spend import load_llm_config

    model = load_llm_config("configs/llm.yaml").models["openai-gpt-6-luna"]
    plain = agent_turn_worst_usd(model.price, model.max_output_tokens, 6000, AgentLimits())
    checked = agent_turn_worst_usd(
        model.price, model.max_output_tokens, 6000, AgentLimits(), checked=True
    )
    assert checked > plain


def test_frozen2_needs_the_frozen_flag(tmp_path):
    from copilot.evaluation.qa import QaError, run_qa

    with pytest.raises(QaError, match="frozen_flag_required"):
        run_qa(
            "configs/experiments/orb-text-qa-agent-v3.yaml",
            "frozen2",
            tmp_path,
            max_spend_usd=1.0,
        )


def test_the_self_check_orb_config_registers_its_rule():
    from copilot.evaluation.qa import load_qa_experiment

    experiment = load_qa_experiment("configs/experiments/orb-text-qa-agent-check.yaml")
    assert [v.name for v in experiment.variants] == ["flat", "agent", "agent-check"]
    agent, checked = experiment.variant("agent").agent, experiment.variant("agent-check").agent
    assert {k: v for k, v in checked.items() if not k.startswith(("check", "rewrite"))} == agent
    assert checked["check"] is True and agent["prompt"] == "prompts/agent/research-v2.yaml"
    assert experiment.faithfulness["model"] == "openai-gpt-6-luna-low"
    assert experiment.decision["primary"] == "agent-check"
    assert experiment.decision["baseline"] == "agent"
    assert experiment.decision["accuracy_max_drop"] == 0.02
