"""MTRAG answering on fixtures: scoring by answerability, live and gold-prefix replay, failures,
summaries and the paired comparison (agent experiment, Task 7)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from copilot.chat.turns import INSUFFICIENT_TEXT
from copilot.contracts import Claim, Evidence, GroundedAnswer
from copilot.evaluation.judge import JudgeOutcome, JudgeVerdict
from copilot.evaluation.mtrag import MtragConversation, MtragTurn
from copilot.evaluation.mtrag_run import (
    MtragAnswer,
    MtragRunError,
    TurnContext,
    average_precision,
    judge_question,
    load_mtrag_experiment,
    mtrag_preflight_usd,
    pair_systems,
    run_gold_prefix,
    run_live,
    score_turn,
    select_conversations,
    summarize_mtrag,
)
from copilot.models.generation import GenerationOutcome
from copilot.models.llm import LlmError

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "configs" / "experiments" / "mtrag-agent.yaml"


class FixedJudge:
    identity = "fixed"

    def __init__(self, equivalent: bool = True) -> None:
        self.equivalent = equivalent
        self.questions: list[str] = []

    def judge(self, question, reference, answer, *, request_id=None, run_id=None):
        self.questions.append(question)
        return JudgeOutcome(
            JudgeVerdict(reason="r", equivalent=self.equivalent), calls=1, cost_usd=0.0001
        )


def _turn(label: str = "ANSWERABLE", relevant=("p1",), turn: int = 1, task: str = "c1<::>1"):
    return MtragTurn(
        task_id=task,
        turn=turn,
        question=f"question {turn}",
        history=[("user", "earlier"), ("agent", "reference answer")] if turn > 1 else [],
        reference="ref",
        answerability=label,
        multi_turn_type="Follow-up" if turn > 1 else "N/A",
        relevant=list(relevant),
    )


def _evidence(paper: str) -> Evidence:
    return Evidence(
        id="E1",
        source_kind="paper",
        source_id=paper,
        version_id="v",
        chunk_id=uuid4(),
        text="passage",
        section="Passage",
    )


ABSTAINED = GenerationOutcome(GroundedAnswer(claims=[], insufficient_evidence=True))


def _answered(label: str = "E1") -> GenerationOutcome:
    return GenerationOutcome(
        GroundedAnswer(
            claims=[Claim(text="an answer", evidence_ids=[label])], insufficient_evidence=False
        )
    )


def test_scoring_follows_the_answerability_label():
    judge = FixedJudge()
    assert (
        score_turn(_turn("UNANSWERABLE", ()), ABSTAINED, [], judge, run_id="r")["correct"] is True
    )
    assert (
        score_turn(_turn("UNANSWERABLE", ()), _answered(), ["p1"], judge, run_id="r")["correct"]
        is False
    )
    assert score_turn(_turn("ANSWERABLE"), ABSTAINED, [], judge, run_id="r")["correct"] is False
    assert score_turn(_turn("PARTIAL"), ABSTAINED, [], judge, run_id="r")["correct"] is False
    assert judge.questions == []
    row = score_turn(_turn("ANSWERABLE"), _answered(), ["p1"], judge, run_id="r")
    assert row["correct"] is True and row["judge_calls"] == 1 and row["cited_map"] == 1.0


def test_average_precision_over_several_relevant_passages():
    assert average_precision(["a", "x", "b"], ["a", "b"]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(["x"], []) is None


def test_the_judge_reads_the_human_rewrite_when_there_is_one():
    turn = _turn(turn=2, task="c1<::>2")
    assert judge_question(turn, {"c1<::>2": "standalone rewrite"}) == "standalone rewrite"
    assert judge_question(turn, {}) == "earlier / question 2"


class RecordingSystem:
    name = "agent"

    def __init__(self, fail_on_turn: int | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on_turn = fail_on_turn
        self.error = error
        self.paper = str(uuid4())

    def answer(self, message, history, carried, *, run_id):
        self.calls.append({"message": message, "history": list(history), "carried": list(carried)})
        if self.fail_on_turn == len(self.calls):
            raise self.error
        evidence = [_evidence(self.paper)]
        return MtragAnswer(
            outcome=_answered(),
            evidence=evidence,
            query=message,
            paper_ids=[self.paper],
            searched=not carried,
            searches=0 if carried else 1,
            model_calls=2,
            stop="answered",
            cost_usd=0.001,
            seconds=1.0,
        )


def _conversation() -> MtragConversation:
    return MtragConversation(
        "c1", "development", [_turn(turn=1, task="c1<::>1"), _turn(turn=2, task="c1<::>2")]
    )


def _context(system: RecordingSystem, judge=None) -> TurnContext:
    return TurnContext(
        judge=judge or FixedJudge(),
        run_id="r",
        rewrites={},
        passages=lambda ids: {system.paper: "p1"} if system.paper in ids else {},
    )


def test_live_mode_feeds_each_system_its_own_answers_and_cited_passages():
    system = RecordingSystem()
    rows = run_live(_conversation(), system, _context(system))
    first, second = system.calls
    assert first["history"] == [] and first["carried"] == []
    assert [(h.role, h.text) for h in second["history"]] == [
        ("user", "question 1"),
        ("assistant", "an answer"),
    ]
    assert [e.source_id for e in second["carried"]] == [first_carried(system)]
    assert [r["turn"] for r in rows] == [1, 2]
    assert rows[1]["searched"] is False and rows[1]["cited_map"] == 1.0
    assert "text" in rows[0] and rows[0]["text"]["answer"] == "an answer"


def first_carried(system: RecordingSystem):
    from uuid import UUID

    return UUID(system.paper)


def test_gold_prefix_uses_the_reference_conversation_and_carries_nothing():
    system = RecordingSystem()
    run_gold_prefix(_conversation(), system, _context(system))
    second = system.calls[1]
    assert [(h.role, h.text) for h in second["history"]] == [
        ("user", "earlier"),
        ("assistant", "reference answer"),
    ]
    assert second["carried"] == []


def test_a_failed_live_turn_is_recorded_and_the_conversation_goes_on():
    system = RecordingSystem(fail_on_turn=1, error=LlmError("llm_timeout"))
    rows = run_live(_conversation(), system, _context(system))
    assert rows[0]["failure"] == "llm_timeout" and rows[0]["correct"] is False
    assert system.calls[1]["history"][-1].text == INSUFFICIENT_TEXT
    assert system.calls[1]["carried"] == [] and len(rows) == 2


def _row(conversation: str, task: str, turn: int, correct, label="ANSWERABLE", **extra):
    base = {
        "conversation_id": conversation,
        "task_id": task,
        "turn": turn,
        "answerability": label,
        "correct": correct,
        "abstained": False,
        "failure": None,
        "judge_error": None,
        "cited_map": None,
        "evidence_recall@10": None,
        "searched": True,
        "searches": 1,
        "model_calls": 2,
        "cost_usd": 0.001,
        "seconds": 1.0,
    }
    return {**base, **extra}


def test_the_summary_counts_labels_refusals_and_skips():
    rows = [
        _row("a", "a1", 1, True),
        _row("a", "a2", 2, False, searched=False, searches=0),
        _row("b", "b1", 1, True, label="UNANSWERABLE", abstained=True),
        _row("b", "b2", 2, True, searched=False, searches=0),
    ]
    metrics = summarize_mtrag(rows, resamples=100, seed=1, confidence=0.95)
    assert metrics["accuracy"] == pytest.approx(0.75) and metrics["judged"] == 4
    assert metrics["by_answerability"]["UNANSWERABLE"]["accuracy"] == 1.0
    assert metrics["refusal"] == {
        "unanswerable_turns": 1,
        "recall": 1.0,
        "precision": 1.0,
        "abstentions": 1,
    }
    assert metrics["followups"]["skip_rate"] == 1.0
    assert metrics["followups"]["accuracy_when_skipped"] == pytest.approx(0.5)


def test_pairing_counts_discordant_turns():
    treated = [_row("a", "a1", 1, True), _row("a", "a2", 2, True), _row("b", "b1", 1, None)]
    baseline = [_row("a", "a1", 1, False), _row("a", "a2", 2, True), _row("b", "b1", 1, True)]
    pair = pair_systems(treated, baseline, resamples=100, seed=1, confidence=0.95)
    assert (pair["turns"], pair["only_treated"], pair["only_baseline"]) == (2, 1, 0)
    assert pair["difference"] == pytest.approx(0.5)


def test_the_experiment_loads_and_its_preflight_covers_both_systems():
    experiment = load_mtrag_experiment(EXPERIMENT)
    assert experiment.modes == ("live", "gold_prefix")
    assert [s["name"] for s in experiment.systems] == ["pipeline", "agent"]
    worst = mtrag_preflight_usd(
        experiment, turns=40, evidence_tokens=6000, chat_model="openai-gpt-6-luna"
    )
    assert worst > 0


def test_a_bad_experiment_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(EXPERIMENT.read_text().replace("modes: [live, gold_prefix]", "modes: [replay]"))
    with pytest.raises(MtragRunError, match="modes"):
        load_mtrag_experiment(path)


def _in_split(conversation_id: str, split: str) -> MtragConversation:
    return MtragConversation(conversation_id, split, [_turn(task=f"{conversation_id}<::>1")])


def test_the_all_split_reads_every_conversation_and_is_never_frozen():
    conversations = [_in_split("a", "development"), _in_split("b", "test")]
    assert [
        c.conversation_id for c in select_conversations(conversations, "all", frozen=False)
    ] == ["a", "b"]
    assert [
        c.conversation_id for c in select_conversations(conversations, "development", frozen=False)
    ] == ["a"]
    with pytest.raises(MtragRunError, match="frozen_not_allowed"):
        select_conversations(conversations, "all", frozen=True)
    with pytest.raises(MtragRunError, match="frozen_flag_required"):
        select_conversations(conversations, "test", frozen=False)


def test_named_agents_load_with_a_secondary_comparison_and_preflight_as_agents():
    experiment = load_mtrag_experiment(ROOT / "configs" / "experiments" / "mtrag-agent-v3-dev.yaml")
    assert [s["name"] for s in experiment.systems] == ["pipeline", "agent-v2", "agent-v3"]
    assert experiment.pairs() == {
        "decision": ("agent-v3", "pipeline"),
        "secondary": ("agent-v3", "agent-v2"),
    }
    two = mtrag_preflight_usd(
        load_mtrag_experiment(EXPERIMENT),
        turns=10,
        evidence_tokens=6000,
        chat_model="openai-gpt-6-luna",
    )
    three = mtrag_preflight_usd(
        experiment, turns=10, evidence_tokens=6000, chat_model="openai-gpt-6-luna"
    )
    assert three > two * 1.4  # a second agent costs an agent's worst case, not a pipeline's


def test_a_comparison_naming_an_unknown_system_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(EXPERIMENT.read_text().replace("treated: agent", "treated: agent-v9"))
    with pytest.raises(MtragRunError, match="decision"):
        load_mtrag_experiment(path)


def test_the_govt_experiment_reads_its_own_domain_and_registers_both_comparisons():
    experiment = load_mtrag_experiment(ROOT / "configs" / "experiments" / "mtrag-govt-agent.yaml")
    assert experiment.domain.name == "govt" and experiment.release == "mtrag-govt-v1"
    assert experiment.pairs() == {
        "decision": ("agent-v3", "pipeline"),
        "secondary": ("agent-v3", "agent-v2"),
    }
    assert experiment.decision["mode"] == "live" and experiment.decision["mcnemar_p"] == 0.05


def test_the_self_check_fiqa_config_registers_its_rule():
    from copilot.evaluation.mtrag_run import load_mtrag_experiment

    for path in (
        "configs/experiments/mtrag-fiqa-agent-check.yaml",
        "configs/experiments/mtrag-govt-agent-check-dev.yaml",
    ):
        experiment = load_mtrag_experiment(path)
        assert [s["name"] for s in experiment.systems] == ["pipeline", "agent", "agent-check"]
        assert experiment.pairs()["secondary"] == ("agent-check", "agent")
        assert experiment.faithfulness["model"] == "openai-gpt-6-luna-low"
    assert (
        load_mtrag_experiment("configs/experiments/mtrag-fiqa-agent-check.yaml").domain.name
        == "fiqa"
    )
