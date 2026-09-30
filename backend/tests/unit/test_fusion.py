"""Reciprocal rank fusion and reranker score handling, as pure functions (spec §7)."""

from __future__ import annotations

import pytest

from copilot.search.fusion import rrf


def test_fusion_counts_each_list_once():
    result = dict(rrf([["a", "a", "b"], ["b", "c"]]))
    assert result["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert result["a"] == pytest.approx(1 / 61)
    assert rrf([["b", "a"], ["a", "b"]])[0][0] == "a"


def test_ties_sort_by_id_and_a_repeat_counts_at_its_best_rank():
    fused = rrf([["b", "c"], ["c", "b"]])
    assert [item for item, _ in fused] == ["b", "c"]
    assert fused[0][1] == pytest.approx(fused[1][1])
    # "a" appears at ranks 1 and 3 of one list: counted once, at rank 1.
    assert dict(rrf([["a", "x", "a"]]))["a"] == pytest.approx(1 / 61)


def test_empty_inputs_fuse_to_nothing():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_one_list_keeps_its_own_order():
    assert [item for item, _ in rrf([["z", "a", "m"]])] == ["z", "a", "m"]


def test_the_constant_is_ours_and_must_be_positive():
    assert dict(rrf([["a"]], k=1))["a"] == pytest.approx(1 / 2)
    with pytest.raises(ValueError, match="invalid_rrf_k"):
        rrf([["a"]], k=0)


def test_reranking_reorders_but_never_adds_drops_or_swaps_ids():
    from copilot.search.rerank import rerank_order

    ordered = rerank_order(["a", "b", "c"], [0.1, 0.9, 0.5])
    assert ordered == [("b", 0.9), ("c", 0.5), ("a", 0.1)]
    # Equal scores keep the fused order they arrived in.
    assert [item for item, _ in rerank_order(["x", "y", "z"], [1.0, 1.0, 2.0])] == ["z", "x", "y"]


@pytest.mark.parametrize(
    ("scores", "code"),
    [
        ([0.1], "score_count_mismatch"),
        ([0.1, 0.2, 0.3], "score_count_mismatch"),
        ([0.1, float("nan")], "nonfinite_score"),
        ([float("inf"), 0.0], "nonfinite_score"),
    ],
)
def test_a_reranker_that_miscounts_or_returns_nonfinite_scores_is_refused(scores, code):
    from copilot.search.rerank import RerankError, rerank_order

    with pytest.raises(RerankError, match=code):
        rerank_order(["a", "b"], scores)


def test_the_fixture_reranker_is_deterministic_and_ordered_by_term_overlap():
    from copilot.search.rerank import FixtureReranker

    reranker = FixtureReranker()
    texts = ["graph neural networks", "image diffusion", "graph attention networks"]
    scores = reranker.score("graph neural networks", texts)
    assert scores == reranker.score("graph neural networks", texts)
    assert scores[0] == 1.0 and scores[1] == 0.0 and 0.0 < scores[2] < 1.0
    assert reranker.score("", texts) == [0.0, 0.0, 0.0]


def test_the_reranker_pin_requires_a_commit(tmp_path):
    import yaml

    from copilot.models.embeddings import EmbeddingError
    from copilot.search.rerank import load_reranker_spec

    section = {
        "repo": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "files": [{"file": "tokenizer.json", "sha256": "ab" * 32}],
        "pair_max_tokens": 1024,
        "batch_size": 16,
        "precision": {"cpu": "float32", "cuda": "float16"},
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"reranker": section}))
    spec = load_reranker_spec(path)
    assert spec.pair_max_tokens == 1024
    assert spec.model_dir(tmp_path).parts[-4:] == (
        "rerankers",
        "BAAI",
        "bge-reranker-v2-m3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    )
    path.write_text(yaml.safe_dump({"reranker": {**section, "revision": "main"}}))
    with pytest.raises(EmbeddingError, match="revision"):
        load_reranker_spec(path)
