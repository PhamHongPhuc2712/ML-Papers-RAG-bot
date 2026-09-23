"""Pinning the LitSearch benchmark and matching it to our corpus."""

from __future__ import annotations

import pytest

from copilot.evaluation import litsearch
from copilot.evaluation.litsearch import (
    CorpusPaper,
    LitSearchDoc,
    LitSearchQuery,
    UnpinnedRevisionError,
    acl_id_from_url,
    gold_coverage,
    match_to_corpus,
    normalize_title,
)


def test_fetch_pins_the_revision(tmp_path, monkeypatch):
    seen: list[str] = []

    def fake(repo, revision, member, root):
        seen.append(revision)
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(litsearch, "hf_fetch", fake)
    litsearch.fetch_litsearch(tmp_path)
    assert seen and set(seen) == {litsearch.LITSEARCH_REVISION}


def test_a_floating_revision_is_refused(tmp_path):
    for revision in ("main", "refs/convert/parquet", ""):
        with pytest.raises(UnpinnedRevisionError):
            litsearch.fetch_litsearch(tmp_path, revision=revision)


def test_title_normalization_ignores_case_punctuation_and_spacing():
    assert normalize_title("Attention Is All You Need!") == normalize_title(
        "attention is all-you-need"
    )
    assert (
        normalize_title("BERT: Pre-training  of Deep\nBidirectional")
        == "bertpretrainingofdeepbidirectional"
    )
    assert normalize_title("") == ""


def test_acl_ids_come_out_of_anthology_urls():
    assert acl_id_from_url("https://aclanthology.org/2024.acl-long.123.pdf") == "2024.acl-long.123"
    assert acl_id_from_url("https://aclanthology.org/2023.emnlp-main.45/") == "2023.emnlp-main.45"
    assert acl_id_from_url("https://openreview.net/pdf/abc.pdf") is None
    assert acl_id_from_url(None) is None


def _doc(cid, title, acl=None):
    return LitSearchDoc(corpusid=cid, title=title, acl_id=acl, arxiv_id=None, doi=None)


def _paper(pid, title, url=None):
    return CorpusPaper(paper_id=pid, title=title, pdf_url=url, year=2024)


def test_a_strong_id_match_beats_a_title_match():
    """An ACL id is evidence; a matching title is a guess that usually works."""

    docs = [_doc("c1", "Some Reformatted Title", acl="2024.acl-long.1")]
    papers = [
        _paper("p-acl", "A Completely Different Title", "https://aclanthology.org/2024.acl-long.1.pdf"),
        _paper("p-title", "Some Reformatted Title"),
    ]
    report = match_to_corpus(docs, papers)
    assert report.matches["c1"].paper_id == "p-acl"
    assert report.matches["c1"].method == "acl_id"
    assert report.matched_by_strong_id == 1
    assert report.matched_by_title == 0


def test_title_matching_is_the_fallback_and_is_labelled_as_such():
    docs = [_doc("c1", "Attention Is All You Need")]
    papers = [_paper("p1", "attention is all you need")]
    report = match_to_corpus(docs, papers)
    assert report.matches["c1"].method == "title"
    assert report.matched_by_title == 1


def test_an_ambiguous_title_is_recorded_not_guessed():
    """Two corpus papers share a normalized title: pick neither, count it."""

    docs = [_doc("c1", "A Study")]
    papers = [_paper("p1", "A Study"), _paper("p2", "a study!")]
    report = match_to_corpus(docs, papers)
    assert "c1" not in report.matches
    assert report.ambiguous_titles == 1
    assert report.unmatched == 1


def test_documents_with_no_counterpart_are_unmatched():
    report = match_to_corpus([_doc("c1", "Nothing Like It")], [_paper("p1", "Different")])
    assert report.unmatched == 1
    assert report.matches == {}


def test_gold_coverage_counts_full_partial_and_missing_separately():
    queries = [
        LitSearchQuery("q1", "?", "manual_acl", 1, 2, ("c1",)),
        LitSearchQuery("q2", "?", "manual_acl", 1, 2, ("c1", "c2")),
        LitSearchQuery("q3", "?", "inline_acl", 0, 1, ("c9",)),
    ]
    matched = {"c1"}
    coverage = gold_coverage(queries, matched)
    assert coverage.covered == 1          # q1: every gold present
    assert coverage.partial == 1          # q2: one of two
    assert coverage.missing == 1          # q3: none
    assert coverage.total == 3
    assert coverage.fraction == pytest.approx(1 / 3)


def test_partial_coverage_is_never_counted_as_covered():
    """Recall against a partly-present gold set understates the system."""

    queries = [LitSearchQuery("q", "?", "manual_acl", 1, 2, ("a", "b", "c"))]
    coverage = gold_coverage(queries, {"a", "b"})
    assert coverage.covered == 0
    assert coverage.partial == 1
    assert coverage.fraction == 0.0


def test_coverage_is_reported_per_query_set():
    queries = [
        LitSearchQuery("q1", "?", "manual_acl", 1, 2, ("c1",)),
        LitSearchQuery("q2", "?", "inline_acl", 1, 2, ("c2",)),
    ]
    coverage = gold_coverage(queries, {"c1"})
    assert coverage.by_query_set["manual_acl"].fraction == pytest.approx(1.0)
    assert coverage.by_query_set["inline_acl"].fraction == 0.0
