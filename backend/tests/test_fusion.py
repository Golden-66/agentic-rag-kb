from app.retrieval.fusion import rrf_fuse


def test_rrf_prefers_items_ranked_high_in_multiple_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    assert ids[:2] == ["a", "b"] or ids[:2] == ["b", "a"]
    assert ids[-1] in {"c", "d"}


def test_rrf_score_formula():
    fused = dict(rrf_fuse([["x"], ["x"]], k=60))
    assert abs(fused["x"] - 2 / 61) < 1e-9


def test_rrf_weights_and_empty():
    assert rrf_fuse([]) == []
    fused = rrf_fuse([["a"], ["b"]], k=60, weights=[1.0, 3.0])
    assert fused[0][0] == "b"
