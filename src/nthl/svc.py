"""The four-view n-gram LinearSVC, with or without the public pool.

One model per frozen fold gives out-of-fold predictions on host train, one
model on all of train gives test predictions. Pool rows join every training
fold at a lower weight. They never join validation, and they were filtered
against every train passage, so no fold sees its own prompts.

LinearSVC gives scores, not probabilities, so scores go through a softmax
with one temperature picked on out-of-fold log loss. It never changes the
top pick.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.ngrams import four_views
from nthl.preds import save_preds, to_labels


def _scores(model, texts):
    order = [list(model.classes_).index(label) for label in LABELS]
    return model.decision_function(texts)[:, order]


def _frame(ids, logp):
    df = pd.DataFrame(np.exp(logp), columns=list(LABELS))
    df.insert(0, "id", np.asarray(ids))
    return df


def run(train, test, pool=None, pool_weight=1.0, routes=None, drop_pool_labels=None, C=0.5, min_df=2, out=".", name="svc", seed=42):
    """Fit, write oof.parquet, test.parquet and summary.json to `out`, return the summary.

    `train` needs id, text, model and fold. `pool` needs text, model and route.
    `drop_pool_labels` leaves those models' pool rows out, so they learn from host train only.
    """
    start = time.time()
    out = Path(out)
    if pool is not None and routes is not None:
        pool = pool[pool["route"].isin(routes)]
    if pool is not None and drop_pool_labels:
        pool = pool[~pool["model"].isin(drop_pool_labels)]
    extra_text = [] if pool is None else pool["text"].tolist()
    extra_label = [] if pool is None else pool["model"].tolist()

    def fit(texts, labels):
        model = make_pipeline(four_views(min_df=min_df), LinearSVC(C=C, class_weight="balanced", random_state=seed))
        weights = np.r_[np.ones(len(texts)), np.full(len(extra_text), pool_weight)]
        model.fit(list(texts) + extra_text, list(labels) + extra_label, linearsvc__sample_weight=weights)
        return model

    oof = np.zeros((len(train), len(LABELS)))
    per_fold = {}
    for k in sorted(train["fold"].unique()):
        tr = (train["fold"] != k).to_numpy()
        va = ~tr
        model = fit(train.loc[tr, "text"], train.loc[tr, "model"])
        oof[va] = _scores(model, train.loc[va, "text"])
        pred = np.asarray(LABELS)[oof[va].argmax(axis=1)]
        per_fold[int(k)] = balanced_accuracy(train.loc[va, "model"], pred)
        print(f"{name} fold {k}: {per_fold[int(k)]:.4f} ({time.time() - start:.0f} s)", flush=True)

    truth = train["model"].to_numpy()
    true_idx = np.array([LABELS.index(m) for m in truth])
    temps = np.logspace(-1, 2, 61)
    losses = [-log_softmax(oof * t, axis=1)[np.arange(len(train)), true_idx].mean() for t in temps]
    temp = float(temps[int(np.argmin(losses))])

    oof_df = _frame(train["id"], log_softmax(oof * temp, axis=1))
    save_preds(oof_df, out / "oof.parquet")
    final = fit(train["text"], train["model"])
    save_preds(_frame(test["id"], log_softmax(_scores(final, test["text"]) * temp, axis=1)), out / "test.parquet")

    pred = to_labels(oof_df)
    summary = {
        "experiment": name,
        "cv": balanced_accuracy(truth, pred),
        "per_fold": per_fold,
        "pool_rows": len(extra_text),
        "pool_weight": pool_weight,
        "routes": routes,
        "drop_pool_labels": drop_pool_labels,
        "C": C,
        "min_df": min_df,
        "temperature": temp,
        "oof_log_loss": float(min(losses)),
        "seconds": round(time.time() - start),
        "recall": {m: float(np.mean(pred[truth == m] == m)) for m in LABELS},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def fit_scores(train, texts, pool=None, pool_weight=0.5, C=0.5, min_df=2, seed=42):
    """One fit on all of train (plus the pool), raw LinearSVC scores for `texts` in LABELS order.

    For the submission notebook, which blends a plain softmax of these scores as Nishant's notebook does.
    """
    extra_text = [] if pool is None else pool["text"].tolist()
    extra_label = [] if pool is None else pool["model"].tolist()
    model = make_pipeline(four_views(min_df=min_df), LinearSVC(C=C, class_weight="balanced", random_state=seed))
    weights = np.r_[np.ones(len(train)), np.full(len(extra_text), pool_weight)]
    model.fit(train["text"].tolist() + extra_text, train["model"].tolist() + extra_label, linearsvc__sample_weight=weights)
    return _scores(model, list(texts))
