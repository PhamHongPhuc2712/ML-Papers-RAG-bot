"""Vectors are checked before any write, and sparse vectors reproduce the BM25 oracle."""

from __future__ import annotations

import pytest

from copilot.models.embeddings import validate_vectors


def test_incompatible_vectors_fail_before_upsert():
    validate_vectors([[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match="dimension"):
        validate_vectors([[1.0]], dimensions=2)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_vectors([[float("nan"), 0.0]], dimensions=2)


def test_a_matrix_is_checked_whole():
    import numpy as np

    validate_vectors(np.zeros((3, 4), dtype=np.float32), dimensions=4)
    with pytest.raises(ValueError, match="dimension"):
        validate_vectors(np.zeros((3, 5), dtype=np.float32), dimensions=4)
    bad = np.zeros((3, 4), dtype=np.float32)
    bad[2, 1] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        validate_vectors(bad, dimensions=4)


def test_fixture_model_is_deterministic_normalized_and_order_preserving():
    import numpy as np

    from copilot.models.embeddings import FixtureEmbedding

    model = FixtureEmbedding()
    texts = ["graph neural networks", "", "contrastive learning", "graph neural networks"]
    first = model.encode_array(texts).vectors
    second = FixtureEmbedding().encode_array(list(reversed(texts))).vectors[::-1]
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(first[0], first[3])
    assert model.encode(["a b c"]) == [list(map(float, model.encode_array(["a b c"]).vectors[0]))]


def test_truncation_is_counted_not_hidden():
    from copilot.models.embeddings import FixtureEmbedding

    batch = FixtureEmbedding().encode_array(["one two three", "one"], max_tokens=2)
    assert batch.truncated == 1


def test_cache_misses_on_any_change_of_model_or_text(tmp_path):
    import numpy as np

    from copilot.models.embeddings import EmbeddingCache, EncodedBatch

    batch = EncodedBatch(np.ones((1, 2), dtype=np.float32), truncated=0)
    cache = EmbeddingCache(tmp_path, "model-a")
    key = EmbeddingCache.key(["h1"], 1024)
    assert cache.load(key) is None
    cache.store(key, batch)
    hit = cache.load(key)
    assert hit is not None and np.array_equal(hit.vectors, batch.vectors)
    assert cache.load(EmbeddingCache.key(["h2"], 1024)) is None
    assert cache.load(EmbeddingCache.key(["h1"], 512)) is None
    assert EmbeddingCache(tmp_path, "model-b").load(key) is None


def _models_yaml(tmp_path, **overrides):
    import yaml

    embedding = {
        "repo": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "dimensions": 1024,
        "preprocessing": "cls-l2-v1",
        "files": [{"file": "tokenizer.json", "sha256": "ab" * 32}],
        "max_tokens": {"papers": 1024, "chunks": 1280},
        "batch_size": 32,
        "precision": {"cpu": "float32", "cuda": "float16"},
        **overrides,
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "embedding": embedding}))
    return path


def test_model_config_requires_a_commit_pin(tmp_path):
    from copilot.models.embeddings import EmbeddingError, load_embedding_spec

    spec = load_embedding_spec(_models_yaml(tmp_path))
    assert spec.identity == "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181#cls-l2-v1"
    with pytest.raises(EmbeddingError, match="revision"):
        load_embedding_spec(_models_yaml(tmp_path, revision="main"))


def test_an_absent_or_altered_model_file_is_refused(tmp_path):
    from copilot.models.embeddings import EmbeddingError, load_embedding_spec, verify_model_files

    spec = load_embedding_spec(_models_yaml(tmp_path))
    with pytest.raises(EmbeddingError, match="model_missing"):
        verify_model_files(spec, tmp_path)
    directory = spec.model_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / "tokenizer.json").write_text("{}")
    with pytest.raises(EmbeddingError, match="model_checksum_mismatch"):
        verify_model_files(spec, tmp_path)


CORPUS = {
    "d1": "graph neural networks for molecules",
    "d2": "contrastive learning of visual representations",
    "d3": "graph contrastive learning with augmentations graph",
    "d4": "",
}


def test_streamed_vocabulary_reproduces_the_oracle_scores():
    """The contract Qdrant's sparse dot product has to meet (spec §7)."""

    from copilot.search.lexical import BM25, BM25Vocabulary

    oracle = BM25()
    oracle.fit(CORPUS)
    vocabulary = BM25Vocabulary.fit(CORPUS.values())
    assert vocabulary.statistics == oracle.statistics

    for query in ("graph learning", "contrastive graph augmentations", "molecules"):
        encoded = vocabulary.encode_query(query)
        query_weights = dict(zip(encoded.indices, encoded.values, strict=True))
        scores = {}
        for doc_id, text in CORPUS.items():
            document = vocabulary.encode_document(text)
            score = sum(
                query_weights.get(index, 0.0) * value
                for index, value in zip(document.indices, document.values, strict=True)
            )
            if score:
                scores[doc_id] = score
        expected = dict(oracle.search(query, limit=10))
        assert scores.keys() == expected.keys()
        for doc_id, score in expected.items():
            assert scores[doc_id] == pytest.approx(score, rel=1e-12)


def test_term_ids_are_positions_in_the_sorted_vocabulary():
    from copilot.search.lexical import BM25Vocabulary

    vocabulary = BM25Vocabulary.fit(["beta alpha", "gamma"])
    assert vocabulary.encode_query("gamma alpha beta").indices == (0, 1, 2)
    assert vocabulary.encode_query("unseen words only").indices == ()


def test_vocabulary_round_trips_and_refuses_an_altered_file(tmp_path):
    from copilot.search.lexical import BM25Vocabulary, LexicalError

    vocabulary = BM25Vocabulary.fit(CORPUS.values())
    path = tmp_path / "papers-bm25.parquet"
    digest = vocabulary.save(path)
    loaded = BM25Vocabulary.load(path, sha256=digest)
    assert loaded.statistics == vocabulary.statistics
    assert loaded.encode_document(CORPUS["d3"]) == vocabulary.encode_document(CORPUS["d3"])
    with pytest.raises(LexicalError, match="vocabulary_checksum_mismatch"):
        BM25Vocabulary.load(path, sha256="0" * 64)


def test_a_document_outside_the_fitted_corpus_is_refused():
    from copilot.search.lexical import BM25Vocabulary, LexicalError

    with pytest.raises(LexicalError, match="unknown_term"):
        BM25Vocabulary.fit(["alpha"]).encode_document("alpha omega")


def test_collection_names_are_namespaced_and_refuse_unsafe_ids():
    from copilot.search.index import IndexBuildError, collection_names

    assert collection_names("dev_", "m2-20260924T095724Z") == {
        "papers": "dev_paper_abstracts_m2-20260924T095724Z",
        "chunks": "dev_paper_chunks_m2-20260924T095724Z",
    }
    for unsafe in ("", "../x", "a/b", "a b", "-leading"):
        with pytest.raises(IndexBuildError, match="invalid_release_id"):
            collection_names("dev_", unsafe)


def test_canary_rankings_tolerate_ties_but_not_reorderings():
    from copilot.search.index import same_ranking

    expected = [["a", 0.9], ["b", 0.5], ["c", 0.5], ["d", 0.1]]
    assert same_ranking(expected, [["a", 0.9], ["c", 0.5], ["b", 0.5], ["d", 0.1]])
    assert not same_ranking(expected, [["b", 0.9], ["a", 0.5], ["c", 0.5], ["d", 0.1]])
    assert not same_ranking(expected, [["a", 0.9], ["b", 0.6], ["c", 0.5], ["d", 0.1]])
    assert not same_ranking(expected, expected[:3])
    # The last item is checked like every other one.
    assert not same_ranking(expected, [["a", 0.9], ["b", 0.5], ["c", 0.5], ["x", 0.1]])


def test_an_all_tied_ranking_must_still_name_the_same_documents():
    """Review finding: a single tie group once skipped the identity check entirely."""

    from copilot.search.index import same_ranking

    assert not same_ranking([["a", 0.5], ["b", 0.5]], [["x", 0.5], ["y", 0.5]])
    assert same_ranking([["a", 0.5], ["b", 0.5]], [["b", 0.5], ["a", 0.5]])


def test_a_tie_cut_by_the_limit_is_found_in_the_slack():
    from copilot.search.index import same_ranking

    recorded = [["a", 0.9], ["b", 0.5]]
    # "c" ties "b" and now sorts first; "b" moved just past the recorded limit.
    assert same_ranking(recorded, [["a", 0.9], ["c", 0.5], ["b", 0.5]])
    assert not same_ranking(recorded, [["a", 0.9], ["c", 0.5], ["d", 0.5]])


def test_both_branches_share_one_filter_and_never_serve_deleted_points():
    from copilot.contracts import PaperFilters
    from copilot.search.index import qdrant_filter

    empty = qdrant_filter(None)
    assert empty.must is None
    assert [condition.key for condition in empty.must_not] == ["deleted"]
    full = qdrant_filter(
        PaperFilters(year_from=2024, year_to=2025, venues=["ICLR", "ACL"], fulltext_only=True)
    )
    keys = {condition.key: condition for condition in full.must}
    assert keys["year"].range.gte == 2024 and keys["year"].range.lte == 2025
    assert keys["venue"].match.any == ["ICLR", "ACL"]
    assert keys["fulltext"].match.value is True
