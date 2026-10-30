"""MultiHop-RAG: sampling, splits, evidence checks and chunk headers (agent follow-ups, Task 5)."""

from __future__ import annotations

import re
from collections import Counter

import pytest
from sqlalchemy import create_engine

from copilot.corpus.chunk import ChunkerConfig, TokenizerSpec
from copilot.evaluation.multihop import (
    MultihopError,
    _guard_database,
    article_header,
    check_evidence,
    chunk_article,
    query_id,
    sample_queries,
)


def _row(kind: str, n: int, urls=("https://a",)) -> dict:
    return {
        "query": f"{kind} question {n}?",
        "answer": "Yes",
        "question_type": kind,
        "evidence_list": [{"url": url, "fact": "f"} for url in urls],
    }


def _rows() -> list[dict]:
    rows = []
    for kind, count in (
        ("comparison_query", 200),
        ("temporal_query", 200),
        ("inference_query", 50),
        ("null_query", 50),
    ):
        rows += [_row(kind, n) for n in range(count)]
    return rows


def test_only_comparison_and_temporal_questions_are_sampled_and_splits_are_stable():
    rows = _rows()
    first, second = sample_queries(rows), sample_queries(list(reversed(rows)))
    assert {q.type for q in first} == {"comparison_query", "temporal_query"}
    assert Counter((q.type, q.split) for q in first) == {
        ("comparison_query", "development"): 15,
        ("comparison_query", "test"): 75,
        ("temporal_query", "development"): 15,
        ("temporal_query", "test"): 75,
    }
    assert first == second


def test_a_repeated_question_is_kept_once_and_its_urls_deduplicated():
    rows = _rows() + [_row("comparison_query", 3), _row("comparison_query", 200)]
    rows.append(_row("temporal_query", 999, urls=("https://a", "https://a", "https://b")))
    sampled = sample_queries(rows, development=1, test=200)  # 201 distinct of each type
    ids = [q.id for q in sampled]
    assert len(ids) == len(set(ids))
    repeated = next(q for q in sampled if q.id == query_id("temporal_query question 999?"))
    assert repeated.evidence_urls == ("https://a", "https://b")


def test_too_few_questions_of_a_type_is_refused():
    with pytest.raises(MultihopError, match="too_few_questions"):
        sample_queries([_row("comparison_query", n) for n in range(10)])


def test_evidence_outside_the_corpus_is_refused():
    (query,) = [
        q for q in sample_queries(_rows(), development=1, test=0) if q.type == "temporal_query"
    ]
    check_evidence([query], {"https://a"})
    with pytest.raises(MultihopError, match="evidence_outside_corpus"):
        check_evidence([query], {"https://elsewhere"})


def _spans(text: str):
    return [m.span() for m in re.finditer(r"\S+", text)]


CHUNKER = ChunkerConfig(
    policy="paragraph-pack",
    chunker_version="paragraph-pack-v1",
    target_tokens=40,
    hard_cap_tokens=60,
    overlap_tokens=0,
    references_in_default_evidence=False,
    uuid_namespace=__import__("uuid").uuid4(),
    tokenizer=TokenizerSpec("whitespace"),
    max_tokens=50,
    paragraph_max_tokens=60,
    overlap_sentences=0,
)


def test_each_chunk_starts_with_source_and_date():
    article = {
        "source": "The Verge",
        "published_at": "2023-10-02T12:00:00+00:00",
        "body": " ".join(f"Sentence number {i} ends here." for i in range(40)),
    }
    chunks = chunk_article(article, _spans, CHUNKER)
    assert len(chunks) > 1
    assert all(c["text"].startswith("The Verge, published 2023-10-02.\n") for c in chunks)
    assert all(c["token_count"] == len(_spans(c["text"])) for c in chunks)
    assert article_header({"source": None, "published_at": None}) == "unknown source.\n"


def test_only_a_multihop_or_test_database_is_loaded():
    for name in ("copilot_multihop", "test_copilot"):
        _guard_database(create_engine(f"postgresql+psycopg://u@localhost/{name}"))
    with pytest.raises(MultihopError, match="database_refused"):
        _guard_database(create_engine("postgresql+psycopg://u@localhost/copilot_v2"))


def test_the_planning_experiment_loads_and_differs_only_in_the_plan():
    from pathlib import Path

    from copilot.evaluation.qa import load_qa_experiment

    experiment = load_qa_experiment(Path("configs/experiments/multihop-plan.yaml"))
    assert experiment.benchmark == "multihop" and experiment.release == "multihop-rag-v1"
    agent, planned = (experiment.variant(name).agent for name in ("agent", "agent-plan"))
    planned = dict(planned)
    assert (planned.pop("plan"), planned.pop("plan_prompt"), planned.pop("max_subquestions")) == (
        True,
        "prompts/agent/plan-v1.yaml",
        3,
    )
    assert planned == dict(agent) and agent["max_searches"] == 4
    assert experiment.decision["primary"] == "agent-plan"


def test_multihop_cases_follow_the_frozen_fixtures(tmp_path, monkeypatch):
    from pathlib import Path

    from copilot.evaluation import multihop
    from copilot.evaluation.qa import QaError, load_multihop_cases, load_qa_experiment

    experiment = load_qa_experiment(Path("configs/experiments/multihop-plan.yaml"))
    sampled = sample_queries(_rows())
    monkeypatch.setattr(multihop, "load_queries", lambda data_dir: sampled)
    monkeypatch.setattr(multihop, "assignment", lambda path: {q.id: q.split for q in sampled})
    cases = load_multihop_cases(experiment, "frozen", tmp_path)
    assert len(cases) == 150 and all(case.gold_paper_id is None for case in cases)
    assert len(load_multihop_cases(experiment, "development", tmp_path)) == 30
    monkeypatch.setattr(multihop, "assignment", lambda path: {})
    with pytest.raises(QaError, match="dataset_mismatch"):
        load_multihop_cases(experiment, "frozen", tmp_path)
