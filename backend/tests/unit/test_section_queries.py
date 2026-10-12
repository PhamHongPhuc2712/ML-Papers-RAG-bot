"""Section-level questions: answerable from one passage, not by the paper's name or its words."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
from pydantic import SecretStr

from copilot.evaluation.datasets import read_dataset, validate_dataset
from copilot.evaluation.section_queries import (
    ChunkSample,
    acceptable,
    assign,
    build_section_queries,
    load_question_prompt,
    longest_copied_run,
    write_section_queries,
)
from copilot.models.llm import ChatClient
from copilot.models.spend import MemoryLedger, load_llm_config

SAMPLE = ChunkSample(
    chunk_id="c1",
    paper_id="p1",
    title="SimCSE: Simple Contrastive Learning of Sentence Embeddings",
    venue="EMNLP",
    year=2021,
    section="Training details",
    text=(
        "We train with a batch size of 64 and a learning rate of 3e-5 for one epoch"
        " on Wikipedia sentences."
    ),
)
CONFIG = load_llm_config("configs/llm.yaml")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"


def test_a_copied_run_is_counted_in_words():
    assert longest_copied_run("batch size of 64 and a learning rate of 3e-5", SAMPLE.text) == 10
    assert longest_copied_run("What batch size was used?", SAMPLE.text) == 2


def test_questions_naming_the_paper_or_copying_it_are_refused():
    assert acceptable("What batch size does SimCSE train with?", SAMPLE) == "names_paper"
    assert (
        acceptable("What batch size does simple contrastive learning use?", SAMPLE) == "names_paper"
    )
    copied = "Do they train with a batch size of 64 and a learning rate of 3e-5?"
    assert acceptable(copied, SAMPLE) == "copies_passage"
    assert acceptable("", SAMPLE) == "empty"
    assert (
        acceptable(
            "What batch size and learning rate were used to train the sentence encoder?", SAMPLE
        )
        is None
    )


def test_assignment_is_deterministic_and_by_paper():
    samples = [ChunkSample(f"c{i}", f"p{i}", "t", "ICLR", 2024, "Method", "x") for i in range(30)]
    first = assign(samples, development=10, validation=10, test=10, seed=42)
    assert first == assign(samples, development=10, validation=10, test=10, seed=42)
    assert sorted(first.values()).count("test") == 10 and len(first) == 30


def _client(replies: list[str]) -> tuple[ChatClient, list[dict]]:
    """A ChatClient whose model replies with ``replies`` in turn, through a mock transport."""

    calls: list[dict] = []
    template = json.loads((FIXTURES / "openai_chat.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        body = json.loads(json.dumps(template))
        body["choices"][0]["message"]["content"] = replies.pop(0)
        return httpx.Response(200, json=body)

    model = CONFIG.models["openai-gpt-6-luna"]
    client = ChatClient(
        model,
        CONFIG.providers[model.provider],
        SecretStr("sk-test"),
        MemoryLedger(daily_cap_usd=Decimal("1.00")),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    return client, calls


def _sample(index: int, title: str) -> ChunkSample:
    return ChunkSample(
        f"c{index}",
        f"p{index}",
        title,
        "ICLR",
        2024,
        "Experiments",
        "We evaluate on three benchmarks and report accuracy averaged over five seeds.",
    )


def test_one_accepted_one_accepted_on_retry_one_dropped(tmp_path):
    samples = [
        _sample(1, "Robust Graph Learning"),
        _sample(2, "FlashSearch: Fast Retrieval"),
        _sample(3, "FlashSearch: Fast Retrieval Again"),
    ]
    client, calls = _client(
        [
            json.dumps({"question": "How many random seeds are results averaged over?"}),
            json.dumps({"question": "How fast is FlashSearch?"}),
            json.dumps({"question": "Over how many runs is reported accuracy averaged?"}),
            json.dumps({"question": "Does FlashSearch beat baselines?"}),
            json.dumps({"answer": "a reply without a question"}),
        ]
    )

    def complete(messages):
        return client.complete(messages, timeout=5.0, purpose="evaluation", run_id="t")

    built = build_section_queries(
        samples, complete, load_question_prompt("prompts/eval/section-question-v1.yaml")
    )

    assert [sample.chunk_id for sample, _ in built.kept] == ["c1", "c2"]
    assert built.rejected == {"names_paper": 2, "unparsable": 1}
    assert built.dropped == 1 and built.calls == 5
    retry = calls[2]["messages"]
    assert retry[-1]["content"].startswith("That question was rejected (names_paper).")
    assert "Experiments" in calls[0]["messages"][1]["content"]

    fixture, texts = tmp_path / "fixture", tmp_path / "data" / "queries.jsonl"
    write_section_queries(
        built.kept, {"c1": "development", "c2": "test"}, fixture, texts, source="copilot_v2@r1"
    )
    assert '"query"' not in (fixture / "queries.jsonl").read_text()
    assert validate_dataset(fixture)["queries"] == 2
    dataset = read_dataset(fixture)
    assert {q.query_set for q in dataset.queries} == {"sections_v1"}
    assert {(r.query_id, r.paper_id) for r in dataset.qrels} == {("c1", "p1"), ("c2", "p2")}
    lines = [json.loads(line) for line in texts.read_text().splitlines()]
    assert lines[0] == {
        "query_id": "c1",
        "query": "How many random seeds are results averaged over?",
    }


def test_a_bare_one_line_question_is_taken_as_the_question():
    from copilot.evaluation.section_queries import _question

    assert _question('"Over how many seeds is accuracy averaged?"\n') == (
        "Over how many seeds is accuracy averaged?"
    )
    assert _question(json.dumps({"question": " Why  this? "})) == "Why this?"
    assert _question("First line.\nSecond line.") is None
    assert _question('{"question": broken') is None
    assert _question(json.dumps({"q": "x"})) is None
