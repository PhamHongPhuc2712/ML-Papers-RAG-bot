"""MTRAG ClapNQ: parsing, grouping, splits, load checks and passage metrics (agent experiment)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from copilot.evaluation.mtrag import (
    CLAPNQ,
    DOMAINS,
    MTRAG_COMMIT,
    MtragError,
    _guard_database,
    assign_splits,
    check_qrels_resolve,
    domain_for_release,
    file_path,
    load_conversations,
    paper_id_for,
    parse_qrels,
    passage_metrics,
    strip_nul,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mtrag"


def _qrels() -> dict[str, list[str]]:
    return parse_qrels((FIXTURES / "dev.tsv").read_text().splitlines())


def test_qrels_skip_the_header_and_keep_file_order():
    assert _qrels() == {"c1<::>2": ["p2", "p3"], "c2<::>1": ["p1"]}


def test_conversations_group_by_id_order_by_turn_and_drop_other_collections():
    by_id = {
        c.conversation_id: c for c in load_conversations(FIXTURES / "reference.jsonl", _qrels())
    }
    assert set(by_id) == {"c1", "c2"}
    first, second = by_id["c1"].turns
    assert [first.turn, second.turn] == [1, 2]
    assert second.question == "When did they finish it?"
    assert second.history == [
        ("user", "Who built the Glass Bridge?"),
        ("agent", "The Ferro company."),
    ]
    assert second.relevant == ["p2", "p3"] and second.reference == "In 1911."
    assert second.multi_turn_type == "Follow-up"


def test_labels_are_kept_and_an_unjudged_turn_has_no_relevant_passages():
    by_id = {
        c.conversation_id: c for c in load_conversations(FIXTURES / "reference.jsonl", _qrels())
    }
    first = by_id["c1"].turns[0]
    assert first.answerability == "UNANSWERABLE" and first.relevant == []
    assert by_id["c2"].turns[0].answerability == "PARTIAL"


def test_splits_are_stable_and_sized():
    ids = [f"conv-{i}" for i in range(29)]
    first = assign_splits(ids, development=5)
    assert first == assign_splits(list(reversed(ids)), development=5)
    assert sum(1 for s in first.values() if s == "development") == 5
    assert sum(1 for s in first.values() if s == "test") == 24


def test_a_qrel_outside_the_corpus_is_refused():
    with pytest.raises(MtragError) as raised:
        check_qrels_resolve({"p1", "p2"}, {"c1<::>1": ["p1"], "c2<::>1": ["p9"]})
    assert raised.value.code == "qrels_unresolved" and "p9" in raised.value.detail


def test_only_an_mtrag_or_test_database_is_loaded():
    for name in ("copilot_mtrag", "copilot_mtrag_govt", "test_copilot"):
        _guard_database(create_engine(f"postgresql+psycopg://u@localhost/{name}"))
    with pytest.raises(MtragError, match="database_refused"):
        _guard_database(create_engine("postgresql+psycopg://u@localhost/copilot_v2"))


def test_passage_ids_map_to_stable_paper_ids():
    assert paper_id_for("822086267_7384-8758-0-1374") == paper_id_for("822086267_7384-8758-0-1374")
    assert paper_id_for("a") != paper_id_for("b")


def test_passage_metrics_on_a_known_ranking():
    m = passage_metrics(["a", "x", "b", "y", "z", "q", "r", "s", "t", "u"], ["a", "b"])
    assert m["recall@5"] == 1.0 and m["recall@10"] == 1.0
    ideal = 1 + 1 / math.log2(3)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_a_query_with_nothing_relevant_found_scores_zero():
    m = passage_metrics(["x"] * 10, ["a"])
    assert m == {"recall@5": 0.0, "ndcg@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}


def test_clapnq_keeps_its_files_release_and_splits():
    root = Path("/data") / "benchmarks" / "mtrag" / MTRAG_COMMIT
    assert file_path(Path("/data"), "qrels") == root / "dev.tsv"
    assert file_path(Path("/data"), "corpus", CLAPNQ) == root / "clapnq.jsonl.zip"
    assert (CLAPNQ.release, CLAPNQ.development) == ("mtrag-clapnq-v1", 5)
    assert CLAPNQ.fixtures == Path("data/fixtures/mtrag")
    assert domain_for_release("mtrag-clapnq-v1") is CLAPNQ


def test_govt_files_live_in_their_own_directory_and_every_conversation_is_test():
    govt = DOMAINS["govt"]
    root = Path("/data") / "benchmarks" / "mtrag" / MTRAG_COMMIT / "govt"
    assert file_path(Path("/data"), "qrels", govt) == root / "dev.tsv"
    assert file_path(Path("/data"), "lastturn", govt) == root / "govt_lastturn.jsonl"
    assert domain_for_release("mtrag-govt-v1") is govt
    conversations = load_conversations(FIXTURES / "reference.jsonl", {}, govt)
    assert [c.conversation_id for c in conversations] == ["c9"]
    assert {c.split for c in conversations} == {"test"}


def test_fiqa_is_a_domain_with_its_own_release_and_no_development():
    fiqa = DOMAINS["fiqa"]
    root = Path("/data") / "benchmarks" / "mtrag" / MTRAG_COMMIT / "fiqa"
    assert file_path(Path("/data"), "corpus", fiqa) == root / "fiqa.jsonl.zip"
    assert domain_for_release("mtrag-fiqa-v1") is fiqa
    assert (fiqa.collection, fiqa.development) == ("mt-rag-fiqa-beir-elser-512-100-20240501", 0)
    assert fiqa.fixtures == Path("data/fixtures/mtrag-fiqa")


def test_an_unknown_release_is_refused():
    with pytest.raises(MtragError, match="unknown_release"):
        domain_for_release("mtrag-cloud-v1")


def test_nul_bytes_are_stripped_before_storage():
    assert strip_nul("a\x00b") == "ab" and strip_nul("plain") == "plain"
