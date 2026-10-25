"""Pairwise specialists for the hardest pairs (#97).

A binary four-view LinearSVC sees only passages of two models. Where the
base prediction's top two are exactly that pair, the specialist re-ranks
them: by its sign alone (a swap), or by adding its margin to the pair's
log-odds at a weight. Every other row is left as it was.
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from nthl.labels import LABELS
from nthl.ngrams import four_views


def top_two_is(scores, a, b):
    """Rows whose two highest scores are a and b, in either order."""
    top2 = np.sort(np.argsort(-np.asarray(scores), axis=1)[:, :2], axis=1)
    return (top2 == sorted((LABELS.index(a), LABELS.index(b)))).all(axis=1)


def rerank(scores, a, b, margin, weight=None):
    """Re-rank a against b on rows where they are the top two. `margin` > 0 favours a.

    weight None swaps the two scores where the margin's sign disagrees with their order.
    A number adds weight * margin to the a-minus-b gap, split evenly so the row sum stays put.
    """
    ia, ib = LABELS.index(a), LABELS.index(b)
    out = np.array(scores, dtype=float)
    rows = top_two_is(out, a, b)
    margin = np.asarray(margin, dtype=float)
    if weight is None:
        flip = rows & ((out[:, ia] > out[:, ib]) != (margin > 0))
        out[flip, ia], out[flip, ib] = out[flip, ib], out[flip, ia].copy()
    else:
        shift = np.where(rows, weight * margin / 2, 0.0)
        out[:, ia] += shift
        out[:, ib] -= shift
    return out


def fit_pair(texts, labels, a, b, extra=None, extra_weight=0.25, C=0.5, min_df=2, seed=42):
    """A balanced binary LinearSVC on the rows labelled a or b, plus `extra` rows (text, model) of those two at a lower weight."""
    labels = np.asarray(labels)
    keep = np.isin(labels, [a, b])
    texts = [t for t, k in zip(texts, keep) if k]
    labels = labels[keep].tolist()
    if extra is not None:
        extra = extra[extra["model"].isin([a, b])]
        weights = np.r_[np.ones(len(texts)), np.full(len(extra), extra_weight)]
        texts, labels = texts + extra["text"].tolist(), labels + extra["model"].tolist()
    else:
        weights = np.ones(len(texts))
    model = make_pipeline(four_views(min_df=min_df), LinearSVC(C=C, class_weight="balanced", random_state=seed))
    model.fit(texts, labels, linearsvc__sample_weight=weights)
    return model


def pair_margin(model, texts, a):
    """The specialist's decision on `texts`, signed so that > 0 favours a."""
    d = model.decision_function(list(texts))
    return d if model.classes_[1] == a else -d
