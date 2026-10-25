"""Does a shared-prompt Gaussian group head help a LoRA24 view on test-like groups? (#46) No submission.

Features come from our own LoRA's stage-1 checkpoint, which never saw host train, so they are honest
on train. The head is fitted per frozen fold on the other folds' rows. Each draw keeps 1 to 5 passages
per prompt, as in the hidden test, and clusters them with every real test text, so groups merge by
mistake as often as in a scoring run. Group members that are test texts join the joint decoding.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
from nthl.gauss import fit, log_posteriors, transform
from nthl.labels import LABELS
from nthl.lora24 import find_base, read_ours, run_spec
from nthl.metric import balanced_accuracy

SEED, DRAWS = 42, 10
TEMPS = [1, 2, 4, 8, 16, 32, 64, 128]
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0]
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
train["text"] = train["text"].fillna("").astype(str)
test["text"] = test["text"].fillna("").astype(str)
y = np.array([LABELS.index(m) for m in train["model"]])

meta = glob.glob("/kaggle/input/**/ckpt_stage1/meta.json", recursive=True)
assert len(meta) == 1, meta
shutil.copytree(os.path.dirname(meta[0]), "/tmp/stage1", dirs_exist_ok=True)
base8 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
probs, _, info, z = run_spec(read_ours("/tmp/stage1"), base8, train["text"].tolist() + test["text"].tolist(), return_features=True)
plain = np.log(np.clip(probs, 1e-12, None))
z_tr, z_te = z[: len(train)], z[len(train):]
print("features", z.shape, "plain on train", round(balanced_accuracy(train["model"], np.asarray(LABELS)[plain[: len(train)].argmax(1)]), 4),
      f"({time.time() - T0:.0f} s)", flush=True)

heads = {}
for k in sorted(train["fold"].unique()):
    tr = (train["fold"] != k).to_numpy()
    prm = fit(z_tr[tr], y[tr], train.loc[tr, "prompt_id"].to_numpy())
    heads[k] = (prm, transform(np.r_[z_tr, z_te], prm))
    print(f"head fold {k}: d {prm['d']} ({time.time() - T0:.0f} s)", flush=True)

rng = np.random.default_rng(SEED)
nll = {T: [] for T in TEMPS}
store = []
for draw in range(DRAWS):
    picks = []
    for _, g in train.groupby("prompt_id"):
        picks += list(rng.choice(g.index.to_numpy(), size=min(len(g), rng.integers(1, 6)), replace=False))
    picks = np.array(sorted(picks))
    texts = train.loc[picks, "text"].tolist() + test["text"].tolist()
    groups, twin = topic_groups(texts)
    has_pick = np.isin(groups, np.unique(groups[: len(picks)][twin[: len(picks)]]))
    rows = twin & has_pick                                      # decode only groups that hold a drawn row
    feat_rows = np.r_[picks, len(train) + np.arange(len(test))]  # rows of the [train; test] transform
    by_t = {T: np.zeros((len(picks), 18)) for T in TEMPS}
    for k, (prm, u_all) in heads.items():
        mine = (train.loc[picks, "fold"] == k).to_numpy()
        u = u_all[feat_rows]
        for T in TEMPS:
            lp = log_posteriors(u, prm, groups, rows, T=T)
            by_t[T][mine] = lp[: len(picks)][mine]
    tw = twin[: len(picks)]
    for T in TEMPS:
        nll[T].append(float(-by_t[T][tw, y[picks][tw]].mean()))
    store.append((picks, tw, by_t))
    print(f"draw {draw}: rows {len(picks)}, grouped {tw.sum()} ({time.time() - T0:.0f} s)", flush=True)

T_best = min(TEMPS, key=lambda T: np.mean(nll[T]))
scores = {str(w): [] for w in WEIGHTS}
scores["gauss_alone_on_grouped"], scores["plain_on_grouped"] = [], []
for picks, tw, by_t in store:
    p, g, yy = plain[picks], by_t[T_best], train["model"].to_numpy()[picks]
    for w in WEIGHTS:
        s = p.copy()
        s[tw] += w * g[tw]
        scores[str(w)].append(balanced_accuracy(yy, np.asarray(LABELS)[s.argmax(1)]))
    scores["gauss_alone_on_grouped"].append(float(np.mean(np.asarray(LABELS)[g[tw].argmax(1)] == yy[tw])))
    scores["plain_on_grouped"].append(float(np.mean(np.asarray(LABELS)[p[tw].argmax(1)] == yy[tw])))

base = np.array(scores["0.0"])
summary = {
    "features_info": info, "temperature": T_best, "nll_by_T": {str(T): round(float(np.mean(v)), 4) for T, v in nll.items()},
    "balanced_accuracy_mean": {k: round(float(np.mean(v)), 4) for k, v in scores.items()},
    "gain_over_plain": {w: round(float(np.mean(np.array(scores[w]) - base)), 4) for w in map(str, WEIGHTS)},
    "gain_sd": {w: round(float(np.std(np.array(scores[w]) - base)), 4) for w in map(str, WEIGHTS)},
    "seconds": round(time.time() - T0),
}
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
