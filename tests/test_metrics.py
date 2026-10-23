import pytest

from rag.metrics import mrr_at_k, ndcg_at_k, recall_at_k, rrf


def test_perfect_ranking_has_ndcg_1():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}) == pytest.approx(1.0)


def test_ndcg_known_value():
    # relevant doc at rank 2 only: (1/log2(3)) / (1/log2(2))
    assert ndcg_at_k(["x", "a"], {"a": 1}) == pytest.approx(0.6309, abs=1e-4)


def test_no_relevant_retrieved():
    assert ndcg_at_k(["x"], {"a": 1}) == 0.0
    assert recall_at_k(["x"], {"a": 1}) == 0.0
    assert mrr_at_k(["x"], {"a": 1}) == 0.0


def test_recall_counts_only_top_k():
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1}, k=2) == 0.5


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("c") > 1 and fused.index("d") > 1
