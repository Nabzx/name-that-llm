import numpy as np
import pandas as pd

from nthl.labels import LABELS
from nthl.pairs import fit_pair, pair_margin, rerank, top_two_is

A, B = "glm-5.3", "glm-5.3-flash"
IA, IB = LABELS.index(A), LABELS.index(B)


def scores(rows):
    """One row per (first, second) pair of labels: first scores 0, second -1, the rest -5."""
    s = np.full((len(rows), len(LABELS)), -5.0)
    for r, (first, second) in enumerate(rows):
        s[r, LABELS.index(first)] = 0.0
        s[r, LABELS.index(second)] = -1.0
    return s


def test_top_two_is_ignores_order_and_needs_both():
    s = scores([(A, B), (B, A), (A, "kimi-k3"), ("kimi-k3", B)])
    assert top_two_is(s, A, B).tolist() == [True, True, False, False]


def test_rerank_swap_follows_the_sign_of_the_margin():
    s = scores([(A, B), (A, B), (B, A), ("kimi-k3", A)])
    margin = np.array([1.0, -1.0, 2.0, -3.0])  # positive favours A
    out = rerank(s, A, B, margin)
    top = np.asarray(LABELS)[out.argmax(axis=1)]
    assert top.tolist() == [A, B, A, "kimi-k3"]
    assert np.array_equal(out[3], s[3])
    assert sorted(out[1]) == sorted(s[1])


def test_rerank_with_a_weight_adds_log_odds_on_pair_rows_only():
    s = scores([(A, B), ("kimi-k3", A)])
    margin = np.array([-0.4, -9.0])
    small = rerank(s, A, B, margin, weight=1.0)
    assert small.argmax(axis=1)[0] == IA
    assert np.isclose(small[0, IA] - small[0, IB], 1.0 - 0.4)
    big = rerank(s, A, B, margin, weight=5.0)
    assert big.argmax(axis=1)[0] == IB
    assert np.array_equal(big[1], s[1])
    assert np.isclose(big[0].sum(), s[0].sum())


def test_fit_pair_learns_an_easy_pair_and_uses_extra_rows():
    rng = np.random.default_rng(0)
    words = lambda: " ".join(rng.choice(["alpha", "beta", "gamma", "delta"], 20))
    texts = [f"left {words()}" for _ in range(10)] + [f"right {words()}" for _ in range(10)]
    labels = [A] * 10 + [B] * 10
    extra = pd.DataFrame({"text": [f"left {words()}" for _ in range(5)], "model": [A] * 5})
    model = fit_pair(texts, labels, A, B, extra=extra, extra_weight=0.25)
    m = pair_margin(model, ["left " + words(), "right " + words()], A)
    assert m[0] > 0 > m[1]
