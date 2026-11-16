from collections import defaultdict


def rrf_fuse(
    ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))。返回按分数降序的 (id, score)。"""
    scores: dict[str, float] = defaultdict(float)
    for i, lst in enumerate(ranked_lists):
        w = weights[i] if weights else 1.0
        for rank, item in enumerate(lst, start=1):
            scores[item] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
