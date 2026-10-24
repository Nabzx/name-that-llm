"""Experiment 1 (#25): four-view n-gram LinearSVC, host train only.

Fits one model per frozen fold for out-of-fold predictions, then one model
on all of train for the test predictions. LinearSVC gives scores, not
probabilities, so scores go through a softmax with one temperature picked
on the out-of-fold log loss. The temperature never changes the top pick.
"""

import glob
import json
import shutil
import sys
import time

# Kaggle unpacks the code dataset flat, so copy it into a folder named nthl.
code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from nthl.folds import load_folds
from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.ngrams import four_views
from nthl.preds import save_preds, to_labels

SEED = 42
C = 0.5
COMP = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"

train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
assert len(train) == 2699 and train["fold"].notna().all()


def fit(texts, labels):
    model = make_pipeline(four_views(), LinearSVC(C=C, class_weight="balanced", random_state=SEED))
    model.fit(texts, labels)
    return model


def scores(model, texts):
    order = [list(model.classes_).index(label) for label in LABELS]
    return model.decision_function(texts)[:, order]


def as_frame(ids, logp):
    df = pd.DataFrame(np.exp(logp), columns=list(LABELS))
    df.insert(0, "id", ids.to_numpy())
    return df


start = time.time()
oof = np.zeros((len(train), len(LABELS)))
per_fold = {}
for k in range(5):
    tr, va = train["fold"] != k, train["fold"] == k
    model = fit(train.loc[tr, "text"], train.loc[tr, "model"])
    oof[va.to_numpy()] = scores(model, train.loc[va, "text"])
    pred = np.asarray(LABELS)[oof[va.to_numpy()].argmax(axis=1)]
    per_fold[k] = balanced_accuracy(train.loc[va, "model"], pred)
    print(f"fold {k}: {per_fold[k]:.4f} ({time.time() - start:.0f} s)", flush=True)

truth = train["model"].to_numpy()
true_idx = np.array([LABELS.index(m) for m in truth])
temps = np.logspace(-1, 2, 61)
losses = [-log_softmax(oof * t, axis=1)[np.arange(len(train)), true_idx].mean() for t in temps]
temp = float(temps[int(np.argmin(losses))])

oof_df = as_frame(train["id"], log_softmax(oof * temp, axis=1))
save_preds(oof_df, f"{OUT}/oof.parquet")
cv = balanced_accuracy(truth, to_labels(oof_df))

final = fit(train["text"], train["model"])
test_df = as_frame(test["id"], log_softmax(scores(final, test["text"]) * temp, axis=1))
save_preds(test_df, f"{OUT}/test.parquet")

recall = {m: float(np.mean(to_labels(oof_df)[truth == m] == m)) for m in LABELS}
summary = {
    "experiment": "svc-train",
    "cv": cv,
    "per_fold": per_fold,
    "temperature": temp,
    "oof_log_loss": float(min(losses)),
    "n_features": int(final[0].transform(train["text"].iloc[:1]).shape[1]),
    "C": C,
    "seconds": round(time.time() - start),
    "recall": recall,
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
