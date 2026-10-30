"""The follow-up harness: three conditions over the same turn 1, judged against B's reference."""

from __future__ import annotations

from decimal import Decimal

from copilot.evaluation.chat_eval import CONDITIONS, Conversation, condition_inputs


def test_each_condition_sends_what_it_names():
    conv = Conversation(
        conversation_id="c1",
        paper_id="p",
        first_id="a",
        first="What does SimCSE train with?",
        second_id="b",
        second="What batch size does SimCSE use?",
        followup="What batch size does it use?",
        reference="64",
    )
    history_answer = "SimCSE trains with a contrastive loss."
    inputs = {c: condition_inputs(conv, c, history_answer) for c in CONDITIONS}
    assert inputs["standalone"] == ("What batch size does SimCSE use?", [], False)
    message, history, condense = inputs["raw"]
    assert message == "What batch size does it use?" and not condense and len(history) == 2
    assert inputs["condensed"][2] is True and inputs["condensed"][0] == conv.followup


def _conversation(cid: str = "c1", paper: str = "p1") -> Conversation:
    return Conversation(
        conversation_id=cid,
        paper_id=paper,
        first_id=f"{cid}-a",
        first="What does SimCSE train with?",
        second_id=f"{cid}-b",
        second="What batch size does SimCSE use?",
        followup="What batch size does it use?",
        reference="64",
    )


class _Model:
    """The condenser's model: always the same standalone query."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, purpose, schema=None, request_id=None, run_id=None):
        from copilot.models.llm import ChatResult, Usage

        self.calls += 1
        return ChatResult("SimCSE batch size", Usage(10, 0, 5), "m", 0.0001, False, 5)


class _Judge:
    identity = "fixture/judge"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, question, reference, answer, *, request_id=None, run_id=None):
        from copilot.evaluation.judge import JudgeOutcome, JudgeVerdict

        self.calls += 1
        return JudgeOutcome(JudgeVerdict("same", True), calls=1, cost_usd=0.0001)


class _Counting:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.messages: list[str] = []

    def run(self, message, history, *, run_id=None):
        self.messages.append(message)
        return self.pipeline.run(message, history, run_id=run_id)


def _pipelines(paper: str = "p1", *, abstain_on: str | None = None):
    from uuid import uuid4

    from copilot.chat.condense import Condenser, load_condense_prompt
    from copilot.chat.pipeline import TurnPipeline
    from copilot.chat.retriever import CorpusRetriever, PaperHits
    from copilot.contracts import Evidence
    from copilot.models.generation import FakeGenerator

    paper_id = str(uuid4())

    def evidence(query, papers, release):
        if abstain_on is not None and abstain_on in query:
            return []
        return [
            Evidence(
                id="E1",
                source_kind="paper",
                source_id=paper_id,
                version_id="v1",
                chunk_id=uuid4(),
                text="The batch size is 64.",
                section="Method",
            )
        ]

    retriever = CorpusRetriever(
        search=lambda query: PaperHits("rel", [paper], []), evidence=evidence
    )
    model = _Model()
    condenser = Condenser(
        model, load_condense_prompt("prompts/chat/condense-v1.yaml"), max_chars=6000
    )
    generator = FakeGenerator()
    pipelines = {
        flag: _Counting(
            TurnPipeline(
                condenser=condenser,
                retriever=retriever,
                generator=generator,
                condense=flag,
            )
        )
        for flag in (True, False)
    }
    return pipelines, model


def test_run_conversation_answers_turn_one_once_and_each_condition():
    from copilot.evaluation.chat_eval import run_conversation

    pipelines, model = _pipelines(abstain_on="it use")
    judge = _Judge()
    rows = run_conversation(_conversation(), pipelines=pipelines, judge=judge, run_id="r")
    assert set(rows) == set(CONDITIONS)
    # turn 1 and the two uncondensed turn 2s go through pipelines[False]
    assert pipelines[False].messages.count("What does SimCSE train with?") == 1
    assert pipelines[True].messages == ["What batch size does it use?"]
    assert rows["raw"]["query"] == "What batch size does it use?"
    assert rows["condensed"]["query"] == "SimCSE batch size" and model.calls == 1
    assert rows["standalone"]["query"] == "What batch size does SimCSE use?"
    # the raw follow-up finds no evidence: an abstention, scored wrong, never judged
    assert rows["raw"]["abstained"] and rows["raw"]["equivalent"] is False
    assert rows["condensed"]["equivalent"] is True and rows["standalone"]["equivalent"] is True
    assert judge.calls == 2
    assert all(row["gold_in_papers"] for row in rows.values())


def _write_conversations(tmp_path, ids):
    import hashlib
    import json

    texts = tmp_path / "data" / "eval" / "chat" / "conversations-v2.jsonl"
    texts.parent.mkdir(parents=True)
    fixture = tmp_path / "conversations.jsonl"
    rows = []
    for cid in ids:
        c = _conversation(cid)
        rows.append({**c.__dict__, "split": "development", "doc_id": "d"})
    texts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fixture.write_text(
        "".join(
            json.dumps(
                {
                    "conversation_id": r["conversation_id"],
                    "split": "development",
                    "paper_id": r["paper_id"],
                    "first_id": r["first_id"],
                    "second_id": r["second_id"],
                    "rewrite_sha256": hashlib.sha256(r["followup"].encode()).hexdigest(),
                }
            )
            + "\n"
            for r in rows
        )
    )
    return fixture


def _experiment(tmp_path, fixture):
    import yaml

    raw = yaml.safe_load(open("configs/experiments/orb-chat-followups.yaml"))
    raw["dataset"]["path"] = str(fixture)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_the_run_writes_three_paired_conditions(tmp_path):
    import json

    import pyarrow.parquet as pq

    from copilot.evaluation.chat_eval import run_chat_eval
    from copilot.models.spend import MemoryLedger

    fixture = _write_conversations(tmp_path, ["c1", "c2", "c3"])
    config = _experiment(tmp_path, fixture)
    pipelines, _ = _pipelines()
    result = run_chat_eval(
        config,
        "development",
        tmp_path / "reports",
        max_spend_usd=1.0,
        data_dir=tmp_path / "data",
        pipelines=pipelines,
        judge=_Judge(),
        ledger=MemoryLedger(Decimal("1")),
    )
    assert pipelines[False].messages.count("What does SimCSE train with?") == 3
    ids = []
    for condition in CONDITIONS:
        directory = tmp_path / "reports" / "development" / f"{result['run_id']}.{condition}"
        table = pq.read_table(directory / "cases.parquet").to_pylist()
        assert "answer" not in table[0] and "query" not in table[0]
        ids.append([row["query_id"] for row in table])
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["cases"] == 3 and len(metrics["accuracy"]["wilson"]) == 2
    assert ids[0] == ids[1] == ids[2] == ["c1", "c2", "c3"]
    text = tmp_path / "data" / "eval" / "chat" / result["run_id"] / "raw" / "cases.jsonl"
    assert "answer" in json.loads(text.read_text().splitlines()[0])


def test_the_run_refuses_what_it_must(tmp_path):
    import pytest

    from copilot.evaluation.chat_eval import ChatEvalError, run_chat_eval
    from copilot.models.spend import MemoryLedger

    fixture = _write_conversations(tmp_path, ["c1"])
    config = _experiment(tmp_path, fixture)
    common = {
        "data_dir": tmp_path / "data",
        "judge": _Judge(),
        "ledger": MemoryLedger(Decimal("1")),
    }
    with pytest.raises(ChatEvalError, match="frozen_flag_required"):
        run_chat_eval(config, "frozen", tmp_path, max_spend_usd=1.0, **common)
    with pytest.raises(ChatEvalError, match="run_budget_exceeded"):
        run_chat_eval(
            config,
            "development",
            tmp_path,
            max_spend_usd=0.0001,
            pipelines=_pipelines()[0],
            **common,
        )
