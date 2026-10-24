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


def topic_groups(texts, cut=0.85, lo=2, hi=18):
    """Cluster by content words. Returns cluster ids and which rows sit in a cluster of lo to hi."""
    x = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(list(texts))
    d = np.clip(1 - (x @ x.T).toarray(), 0, None)
    np.fill_diagonal(d, 0)
    groups = fcluster(linkage(squareform(d, checks=False), method="average"), cut, criterion="distance")
    size = np.bincount(groups)[groups]
    return groups, (size >= lo) & (size <= hi)


def centred_probs(z, groups, twin, w, b):
    """Centred-head probabilities for the twin rows: features minus their cluster mean."""
    zc = z.astype(np.float32).copy()
    for g in np.unique(groups[twin]):
        rows = groups == g
        zc[rows] -= zc[rows].mean(0)
    return softmax(zc[twin] @ w.T + b)
