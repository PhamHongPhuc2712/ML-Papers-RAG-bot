"""The exact BM25 oracle, and the sparse decomposition that must reproduce it."""

from __future__ import annotations

import math

import pytest

from copilot.search.lexical import BM25, K1, B, LexicalError, tokenize


def test_tokenizer_lowercases_and_keeps_alphanumeric_terms():
    assert tokenize("Retrieval-Augmented Generation (RAG)") == [
        "retrieval",
        "augmented",
        "generation",
        "rag",
    ]
    assert tokenize("BERT-base vs. GPT2") == ["bert", "base", "vs", "gpt2"]


def test_tokenizer_normalizes_unicode():
    """Composed and decomposed forms are the same term (spec §7)."""

    assert tokenize("Schrödinger") == tokenize("Schrödinger")


def test_scores_match_a_hand_calculation():
    index = BM25()
    index.fit({"d1": "alpha beta", "d2": "alpha alpha gamma delta"})

    # N=2, df(alpha)=2, avgdl=(2+4)/2=3
    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    d1 = 1 * (K1 + 1) / (1 + K1 * (1 - B + B * 2 / 3))
    d2 = 2 * (K1 + 1) / (2 + K1 * (1 - B + B * 4 / 3))

    scored = dict(index.search("alpha", limit=10))
    assert scored["d1"] == pytest.approx(idf * d1)
    assert scored["d2"] == pytest.approx(idf * d2)


def test_a_rare_term_outweighs_a_common_one():
    index = BM25()
    index.fit({"d1": "common rare", "d2": "common common", "d3": "common x"})
    rare = dict(index.search("rare", limit=10))["d1"]
    common = dict(index.search("common", limit=10))["d1"]
    assert rare > common


def test_length_normalization_favours_the_shorter_document():
    index = BM25()
    index.fit({"short": "alpha beta", "long": "alpha " + " ".join(f"w{i}" for i in range(50))})
    scored = dict(index.search("alpha", limit=10))
    assert scored["short"] > scored["long"]


def test_sparse_dot_product_reproduces_the_oracle():
    """P2.2 serves these weights from Qdrant with no server IDF modifier."""

    index = BM25()
    index.fit({"d1": "alpha beta gamma", "d2": "alpha alpha delta"})
    for query in ("alpha", "alpha beta", "beta gamma delta"):
        oracle = dict(index.search(query, limit=10))
        query_vector = index.encode_query(query)
        for doc_id in ("d1", "d2"):
            doc_vector = index.encode_document(doc_id)
            dot = sum(weight * doc_vector.get(term, 0.0) for term, weight in query_vector.items())
            assert dot == pytest.approx(oracle.get(doc_id, 0.0)), (query, doc_id)


def test_term_ids_come_from_the_sorted_vocabulary():
    index = BM25()
    index.fit({"d1": "gamma alpha", "d2": "beta"})
    assert index.vocabulary == ("alpha", "beta", "gamma")
    assert index.term_id("alpha") == 0
    assert index.term_id("gamma") == 2


def test_unknown_query_terms_are_dropped_not_scored():
    index = BM25()
    index.fit({"d1": "alpha beta"})
    assert index.encode_query("alpha zzz") == index.encode_query("alpha")
    assert index.search("zzz", limit=10) == []


def test_ties_break_on_document_id():
    index = BM25()
    index.fit({"b": "alpha", "a": "alpha", "c": "alpha"})
    assert [doc for doc, _ in index.search("alpha", limit=10)] == ["a", "b", "c"]


def test_empty_corpus_and_empty_query_return_nothing():
    index = BM25()
    index.fit({})
    assert index.search("alpha", limit=10) == []
    index.fit({"d1": "alpha"})
    assert index.search("", limit=10) == []
    assert index.search("   ", limit=10) == []


def test_a_document_with_no_terms_does_not_divide_by_zero():
    index = BM25()
    index.fit({"d1": "alpha beta", "empty": "!!!"})
    scored = dict(index.search("alpha", limit=10))
    assert "empty" not in scored
    assert scored["d1"] > 0


def test_limit_must_be_positive_and_search_needs_a_fitted_index():
    index = BM25()
    with pytest.raises(LexicalError, match="not_fitted"):
        index.search("alpha", limit=10)
    index.fit({"d1": "alpha"})
    with pytest.raises(LexicalError, match="invalid_limit"):
        index.search("alpha", limit=0)


def test_the_same_corpus_scores_identically_twice():
    corpus = {"d1": "alpha beta gamma", "d2": "beta beta delta", "d3": "gamma"}
    first, second = BM25(), BM25()
    first.fit(corpus)
    second.fit(dict(reversed(list(corpus.items()))))
    assert first.search("beta gamma", limit=10) == second.search("beta gamma", limit=10)


def test_statistics_are_recorded_for_the_release():
    index = BM25()
    index.fit({"d1": "alpha beta", "d2": "alpha alpha gamma delta"})
    stats = index.statistics
    assert stats.documents == 2
    assert stats.vocabulary_size == 4
    assert stats.average_length == pytest.approx(3.0)
    assert stats.k1 == K1 and stats.b == B


def test_postings_hold_only_documents_containing_the_term():
    """Search walks postings, not the corpus — 85,729 papers per term is not a plan."""

    index = BM25()
    index.fit({"d1": "alpha beta", "d2": "beta gamma", "d3": "gamma"})
    assert index.postings("alpha") == ("d1",)
    assert index.postings("beta") == ("d1", "d2")
    assert index.postings("zzz") == ()
