"""IR metrics on ranked lists of doc ids (same definitions as BEIR / trec_eval)."""
import math


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 100) -> float:
    relevant = {d for d, r in rel.items() if r > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1 / (i + 1)
    return 0.0


def rrf(rankings: list[list[str]], k: int = 60, top: int = 100) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum 1 / (k + rank)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for i, d in enumerate(ranking):
            score[d] = score.get(d, 0.0) + 1 / (k + i + 1)
    return sorted(score, key=score.get, reverse=True)[:top]
