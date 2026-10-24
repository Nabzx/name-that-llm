"""Blend members in log space, then balance (#20, #23).

Each member adds weight * log(probability). The centred twin of a LoRA
checkpoint reads features with the topic-cluster mean taken away, on test
passages that share a topic with 1 to 17 others. Balancing runs once at the
end, at half strength: each class's total moves halfway towards 1/18.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

K = 18


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn(p, alpha=0.5, iters=300):
    """Scale each column towards (1 - alpha) * its total + alpha * n / K, renormalising rows."""
    target = (1 - alpha) * p.sum(0) + alpha * len(p) / p.shape[1]
    c = np.ones(p.shape[1])
    for _ in range(iters):
        r = p * c
        r /= r.sum(1, keepdims=True)
        c *= target / r.sum(0)
    r = p * c
    return r / r.sum(1, keepdims=True)


def topic_groups(texts, cut=0.85, lo=2, hi=18, return_matrix=False):
    """Cluster by content words. Returns cluster ids and which rows sit in a cluster of lo to hi
    (and the TF-IDF matrix, if asked)."""
    x = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(list(texts))
    d = np.clip(1 - (x @ x.T).toarray(), 0, None)
    np.fill_diagonal(d, 0)
    groups = fcluster(linkage(squareform(d, checks=False), method="average"), cut, criterion="distance")
    size = np.bincount(groups)[groups]
    twin = (size >= lo) & (size <= hi)
    return (groups, twin, x) if return_matrix else (groups, twin)


def centred_probs(z, groups, twin, w, b):
    """Centred-head probabilities for the twin rows: features minus their cluster mean."""
    zc = z.astype(np.float32).copy()
    for g in np.unique(groups[twin]):
        rows = groups == g
        zc[rows] -= zc[rows].mean(0)
    return softmax(zc[twin] @ w.T + b)


def cluster_similarity(x, groups):
    """For every row, the mean pairwise cosine between different members of its cluster (0 for singletons).

    `x` holds L2-normalised TF-IDF rows, as TfidfVectorizer gives by default.
    """
    sim = np.zeros(len(groups))
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        n = len(rows)
        if n < 2:
            continue
        block = (x[rows] @ x[rows].T).toarray()
        sim[rows] = (block.sum() - np.trace(block)) / (n * (n - 1))
    return sim


def peer_correct(z, svc, groups, sim, w_svc, strength=0.5, lo=3, hi=18, prior=8):
    """Peer correction (avazaku, cited): remove a shrunk share of each topic cluster's SVC bias.

    For a cluster of n texts (lo to hi), every row loses strength * r * w_svc times the mean SVC margin
    of the other rows, where r = (n - 1) / (n - 1 + prior) * clip((sim - 0.1) / 0.4, 0, 1). A row never
    removes its own lean, and small or loosely related clusters are corrected less. `z` is the blended
    log score, `svc` the raw SVC scores. Other rows are left alone.
    """
    out = z.astype(float).copy()
    margin = svc - svc.mean(1, keepdims=True)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        n = len(rows)
        if n < lo or n > hi:
            continue
        r = (n - 1) / (n - 1 + prior) * np.clip((sim[rows[0]] - 0.1) / 0.4, 0, 1)
        total = margin[rows].sum(0)
        others = (total[None, :] - margin[rows]) / (n - 1)
        out[rows] -= strength * r * w_svc * others
    return out


def low_margin_rows(p, cut=0.3):
    """Rows whose top two probabilities are less than `cut` apart: the ones a costly member is worth running on."""
    top = np.sort(p, axis=1)
    return np.flatnonzero(top[:, -1] - top[:, -2] < cut)


def mix_rows(p, q, rows, share):
    """Mix `q` into `p` at `share`, on `rows` only. Returns a new array."""
    out = p.copy()
    out[rows] = (1 - share) * p[rows] + share * q[rows]
    return out
