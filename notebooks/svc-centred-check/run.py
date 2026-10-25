"""Does a prompt-centred SVC twin help our SVC on test-like groups? (#46) No submission.

Plain scores are the out-of-fold SVC of LOG row 11 (CV 0.4838). The twin is fitted per fold on host
train only, with rows centred within their prompt. Each draw keeps 1 to 5 passages per prompt, as in
the hidden test, and clusters them together with the real test texts so clusters merge by mistake as
often as they would in a scoring run. Twin scores are only used on rows in a cluster of 2 to 18.
"""

import glob
import json
import shutil
import sys
import time

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, code
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from nthl.blend import topic_groups
from nthl.folds import load_folds
from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.preds import load_preds
from nthl.svc import centred_scores, fit_centred

SEED, DRAWS, WEIGHTS = 42, 10, [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
train["text"] = train["text"].fillna("").astype(str)
test["text"] = test["text"].fillna("").astype(str)

run_dir = glob.glob("/kaggle/input/**/oof.parquet", recursive=True)
assert len(run_dir) == 1, run_dir
oof = load_preds(run_dir[0]).set_index("id").loc[train["id"], LABELS].to_numpy()
temp = json.load(open(run_dir[0].replace("oof.parquet", "summary.json")))["temperature"]
plain = log_softmax(np.log(np.clip(oof, 1e-12, None)) / temp, axis=1)   # = log softmax of the raw SVC scores, as in the blend
truth = train["model"].to_numpy()
print("plain OOF", balanced_accuracy(truth, np.asarray(LABELS)[plain.argmax(1)]), flush=True)

models = {}
for k in sorted(train["fold"].unique()):
    tr = train["fold"] != k
    models[k] = fit_centred(train.loc[tr, "text"], train.loc[tr, "model"], train.loc[tr, "prompt_id"].to_numpy())
    print(f"twin fold {k} fitted ({time.time() - T0:.0f} s)", flush=True)

rng = np.random.default_rng(SEED)
results = {str(w): [] for w in WEIGHTS}
results["replace"], coverage, twin_alone, plain_on_twin = [], [], [], []
for d in range(DRAWS):
    picks = []
    for _, g in train.groupby("prompt_id"):
        picks += list(rng.choice(g.index.to_numpy(), size=min(len(g), rng.integers(1, 6)), replace=False))
    picks = np.array(sorted(picks))
    texts = train.loc[picks, "text"].tolist() + test["text"].tolist()
    all_groups, all_twin = topic_groups(texts)          # one clustering of drawn rows plus every test text
    twin = all_twin[: len(picks)]
    scores = np.zeros((len(picks), len(LABELS)))
    for k, (vec, svc) in models.items():
        mine = (train.loc[picks, "fold"] == k).to_numpy()
        z = centred_scores(vec, svc, texts, all_groups, all_twin)   # cluster means over all members, test texts included
        scores[mine] = z[: len(picks)][mine]
    p, y = plain[picks], truth[picks]
    tz = log_softmax(scores, axis=1)
    for w in WEIGHTS:
        s = p.copy()
        s[twin] += w * tz[twin]
        results[str(w)].append(balanced_accuracy(y, np.asarray(LABELS)[s.argmax(1)]))
    s = p.copy()
    s[twin] = tz[twin]
    results["replace"].append(balanced_accuracy(y, np.asarray(LABELS)[s.argmax(1)]))
    coverage.append(float(twin.mean()))
    twin_alone.append(float(np.mean(np.asarray(LABELS)[tz[twin].argmax(1)] == y[twin])))
    plain_on_twin.append(float(np.mean(np.asarray(LABELS)[p[twin].argmax(1)] == y[twin])))
    print(f"draw {d}: rows {len(picks)}, twin rows {twin.sum()}, " + ", ".join(f"{k} {v[-1]:.4f}" for k, v in results.items()), flush=True)

summary = {
    "draws": DRAWS,
    "balanced_accuracy_mean": {k: round(float(np.mean(v)), 4) for k, v in results.items()},
    "gain_over_plain": {k: round(float(np.mean(np.array(v) - np.array(results["0.0"]))), 4) for k, v in results.items()},
    "gain_sd": {k: round(float(np.std(np.array(v) - np.array(results["0.0"]))), 4) for k, v in results.items()},
    "twin_coverage": round(float(np.mean(coverage)), 3),
    "accuracy_on_twin_rows": {"plain": round(float(np.mean(plain_on_twin)), 4), "twin_alone": round(float(np.mean(twin_alone)), 4)},
    "seconds": round(time.time() - T0),
}
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
