"""Building follow-up conversations: deterministic, split by paper, and leak-checked."""

from __future__ import annotations

from copilot.evaluation.conversations import (
    distinctive_terms,
    leaks,
    pair_by_paper,
    split_by_paper,
)

QUESTIONS = [(f"p{p}", f"q{p}-{i}") for p in range(10) for i in range(3)] + [("lonely", "q-x")]


def test_pairs_are_one_per_paper_with_two_questions_and_deterministic():
    pairs = pair_by_paper(QUESTIONS, seed=42)
    assert pairs == pair_by_paper(QUESTIONS, seed=42)
    assert len(pairs) == 10 and all(a != b for _, a, b in pairs)
    assert "lonely" not in {paper for paper, _, _ in pairs}


def test_splits_never_share_a_paper():
    splits = split_by_paper(pair_by_paper(QUESTIONS, seed=42), development=4, frozen=6, seed=42)
    development = {paper for paper, _, _ in splits["development"]}
    frozen = {paper for paper, _, _ in splits["frozen"]}
    assert len(development) == 4 and len(frozen) == 6 and not development & frozen


def test_a_rewrite_naming_the_method_leaks():
    title = "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
    first = "What loss does SimCSE use with BERT-base?"
    second = "What batch size does SimCSE use with BERT-base on STS-B?"
    terms = distinctive_terms(title, first, second)
    # What turn 2 shares with turn 1 names the referent; STS-B is only turn 2's own subject.
    assert {"simcse", "bert-base"} <= terms and "sts-b" not in terms
    assert leaks("What batch size does SimCSE use?", terms, title)
    assert leaks("What batch size for simple contrastive learning?", terms, title)
    assert not leaks("What batch size does it use?", terms, title)


def _orb_fixture(tmp_path):
    """Four papers with three retrieval_only text questions each, one with an unloaded paper."""

    import json

    root = tmp_path / "data" / "benchmarks" / "orb" / "pdf" / "arxiv"
    (root / "corpus").mkdir(parents=True)
    queries, qrels, answers, urls, qa = {}, {}, {}, {}, []
    for p in range(5):
        doc = f"2301.0000{p}"
        urls[doc] = f"https://arxiv.org/pdf/{doc}"
        (root / "corpus" / f"{doc}.json").write_text(
            json.dumps({"title": f"FooNet{p}: Simple Widgets for Gadget Learning"})
        )
        for i in range(3):
            qid = f"q{p}-{i}"
            queries[qid] = {
                "query": f"How does FooNet{p} handle case {i}?",
                "type": "extractive",
                "source": "text",
            }
            qrels[qid] = {"doc_id": doc, "section_id": i}
            answers[qid] = f"answer {p}-{i}"
            qa.append({"query_id": qid, "qa_split": "retrieval_only"})
    qa[0]["qa_split"] = "development"  # an answering split's question is never reused
    for name, body in (
        ("queries", queries),
        ("qrels", qrels),
        ("answers", answers),
        ("pdf_urls", urls),
    ):
        (root / f"{name}.json").write_text(json.dumps(body))
    dataset = tmp_path / "fixtures"
    dataset.mkdir()
    (dataset / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qa))
    paper_ids = {f"2301.0000{p}": f"paper-{p}" for p in range(4)}  # paper 4 is not loaded
    return tmp_path / "data", dataset, paper_ids


class _Scripted:
    """Leaks once for every other pair, then complies; leaks twice for paper 3."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        from types import SimpleNamespace

        self.calls.append(messages)
        user = messages[1]["content"]
        retried = len(messages) > 2
        if "FooNet3" in user:
            return SimpleNamespace(text="What about FooNet3 then?")
        if "FooNet1" in user and not retried:
            return SimpleNamespace(text="And how does FooNet1 do it?")
        return SimpleNamespace(text="And how does it handle the other case?\nextra line")


def test_the_builder_keeps_rewrites_that_need_the_conversation(tmp_path):
    import hashlib
    import json

    from copilot.evaluation.conversations import TEXTS, build_conversations, read_conversations

    data_dir, dataset, paper_ids = _orb_fixture(tmp_path)
    fixture = tmp_path / "repo" / "conversations.jsonl"
    complete = _Scripted()
    result = build_conversations(
        data_dir=data_dir,
        dataset=dataset,
        paper_ids=paper_ids,
        complete=complete,
        out_fixture=fixture,
        development=2,
        frozen=2,
    )
    counts = result["counts"]
    assert counts["development"]["pairs"] + counts["frozen"]["pairs"] == 4
    assert sum(c["kept"] for c in counts.values()) == 3
    assert sum(c["dropped"] for c in counts.values()) == 1
    # paper 1 and paper 3 each took a retry
    assert result["calls"] == 6 and len(complete.calls) == 6
    retry = next(m for m in complete.calls if len(m) > 2 and "FooNet1" in m[1]["content"])
    assert "foonet1" in retry[-1]["content"]
    repo_text = fixture.read_text()
    assert "FooNet" not in repo_text and "handle" not in repo_text and "answer" not in repo_text
    texts = data_dir / TEXTS
    rows = [json.loads(line) for line in texts.read_text().splitlines()]
    assert {r["paper_id"] for r in rows} == {"paper-0", "paper-1", "paper-2"}
    assert all(r["followup"] == "And how does it handle the other case?" for r in rows)
    assert all("q0-0" not in r["conversation_id"] for r in rows)
    split = rows[0]["split"]
    assert [r["conversation_id"] for r in read_conversations(fixture, texts, split)] == sorted(
        r["conversation_id"] for r in rows if r["split"] == split
    )
    entry = json.loads(repo_text.splitlines()[0])
    assert (
        entry["rewrite_sha256"]
        == hashlib.sha256(b"And how does it handle the other case?").hexdigest()
    )


def test_a_changed_rewrite_is_refused(tmp_path):
    import json

    import pytest

    from copilot.evaluation.conversations import TEXTS, build_conversations, read_conversations

    data_dir, dataset, paper_ids = _orb_fixture(tmp_path)
    fixture = tmp_path / "conversations.jsonl"
    build_conversations(
        data_dir=data_dir,
        dataset=dataset,
        paper_ids=paper_ids,
        complete=_Scripted(),
        out_fixture=fixture,
        development=2,
        frozen=2,
    )
    texts = data_dir / TEXTS
    rows = [json.loads(line) for line in texts.read_text().splitlines()]
    rows[0]["followup"] = "Something else entirely?"
    texts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="conversation_text_changed"):
        read_conversations(fixture, texts, rows[0]["split"])


def test_the_builder_stops_at_its_budget(tmp_path):
    from copilot.evaluation.conversations import build_conversations

    data_dir, dataset, paper_ids = _orb_fixture(tmp_path)
    complete = _Scripted()
    result = build_conversations(
        data_dir=data_dir,
        dataset=dataset,
        paper_ids=paper_ids,
        complete=complete,
        out_fixture=tmp_path / "c.jsonl",
        development=2,
        frozen=2,
        spent=lambda: 0.02 * len(complete.calls),
        max_spend_usd=0.01,
    )
    assert result["stopped"].startswith("budget:")
    assert sum(c["kept"] + c["dropped"] for c in result["counts"].values()) == 1


def test_the_command_refuses_a_budget_below_its_worst_case(capsys):
    from copilot.cli import main

    argv = ["eval", "chat-conversations", "--database-url", "unused", "--max-spend-usd", "0.01"]
    assert main(argv) == 1
    assert '"error": "run_budget_exceeded"' in capsys.readouterr().out
