"""Pairs check (#97): do binary specialists fix glm-5.3 vs glm-5.3-flash and gemini-3.7-flash vs 3.8-flash?

Base is row 11's out-of-fold SVC (pool weight 0.25). Per frozen fold, a binary four-view LinearSVC per pair
is fitted on the other folds' host rows of the two models plus pool rows of them at weight 0.25, with three
pools: none, ours, nuridudak's 122k (CC BY-SA 4.0). Only rows whose base top two are exactly the pair are
re-ranked. CV only, no submission."""

import glob
import json
import os
import shutil
import sys
import time

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, code
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import pandas as pd

from nthl.folds import load_folds
from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.pairs import fit_pair, pair_margin, rerank, top_two_is

PAIRS = [("glm-5.3", "glm-5.3-flash"), ("gemini-3.7-flash", "gemini-3.8-flash")]
WEIGHTS = [None, 0.25, 0.5, 1.0, 2.0, 4.0]
COMP = "/kaggle/input/competitions/name-that-llm"
start = time.time()

train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
assert len(train) == 2699
oof_path = glob.glob("/kaggle/input/**/oof.parquet", recursive=True)
assert len(oof_path) == 1, oof_path
oof = pd.read_parquet(oof_path[0]).set_index("id").loc[train["id"], list(LABELS)]
temperature = json.load(open(os.path.join(os.path.dirname(oof_path[0]), "summary.json")))["temperature"]
base = np.log(np.clip(oof.to_numpy(), 1e-300, None))
truth = train["model"].to_numpy()
folds = train["fold"].to_numpy()


def ba(scores, rows=slice(None)):
    return balanced_accuracy(truth[rows], np.asarray(LABELS)[scores[rows].argmax(axis=1)])


base_ba = ba(base)
print(f"base oof balanced accuracy {base_ba:.4f}, temperature {temperature:.3f}", flush=True)

ours = pd.read_parquet(glob.glob("/kaggle/input/**/pool.parquet", recursive=True)[0])[["text", "model"]]
p122 = glob.glob("/kaggle/input/**/ntl_encoder_pool_122k.parquet", recursive=True)
assert len(p122) == 1, p122
big = pd.read_parquet(p122[0])
src = big["source"].astype(str) if "source" in big else pd.Series("", index=big.index)
big = big[~src.str.contains("olegzholobov|ankiteshtiwari", case=False) & big["model"].isin(LABELS)]
big = big[~big["text"].isin(set(train["text"]))][["text", "model"]].reset_index(drop=True)
POOLS = {"host": None, "ours": ours, "122k": big}
pair_models = [m for p in PAIRS for m in p]
for name, pool in POOLS.items():
    if pool is not None:
        print(name, "pair rows", pool[pool["model"].isin(pair_models)]["model"].value_counts().to_dict(), flush=True)

margins = {}
for name, pool in POOLS.items():
    for a, b in PAIRS:
        m = np.zeros(len(train))
        for k in sorted(np.unique(folds)):
            tr, va = folds != k, folds == k
            model = fit_pair(train.loc[tr, "text"].tolist(), truth[tr], a, b, extra=pool, extra_weight=0.25)
            m[va] = pair_margin(model, train.loc[va, "text"], a)
        margins[(name, a)] = m
        print(f"{name} {a} vs {b} done ({time.time() - start:.0f} s)", flush=True)

report = {"base_ba": base_ba, "temperature": temperature, "pairwise": [], "overall": []}
ia_of = {m: LABELS.index(m) for m in pair_models}
for a, b in PAIRS:
    on_pair = np.isin(truth, [a, b])
    top = top_two_is(base, a, b)
    base_says_a = base[:, ia_of[a]] > base[:, ia_of[b]]
    for subset, rows in [("all pair rows", on_pair), ("pair rows with base top two = pair", on_pair & top)]:
        def pair_ba(says_a):
            return float(np.mean([np.mean(says_a[rows & (truth == a)]), np.mean(~says_a[rows & (truth == b)])]))
        row = {"pair": f"{a} vs {b}", "subset": subset, "n": int(rows.sum()), "base": pair_ba(base_says_a)}
        for name in POOLS:
            row[name] = pair_ba(margins[(name, a)] > 0)
        report["pairwise"].append(row)
        print(json.dumps(row), flush=True)
    print(f"{a} vs {b}: base top two = pair on {int(top.sum())} rows, of which truly in the pair {int((top & on_pair).sum())}", flush=True)

base_pred = base.argmax(axis=1)
for name in POOLS:
    for which in ["both", *[a for a, _ in PAIRS]]:
        for w in WEIGHTS:
            s = base
            for a, b in PAIRS:
                if which in ("both", a):
                    s = rerank(s, a, b, margins[(name, a)], weight=w)
            pred = s.argmax(axis=1)
            per_fold = [ba(s, folds == k) - ba(base, folds == k) for k in sorted(np.unique(folds))]
            row = {"pool": name, "pairs": which, "weight": "swap" if w is None else w, "ba": ba(s),
                   "gain": ba(s) - base_ba, "changed": int((pred != base_pred).sum()),
                   "fixed": int(((pred != base_pred) & (np.asarray(LABELS)[pred] == truth)).sum()),
                   "broken": int(((pred != base_pred) & (np.asarray(LABELS)[base_pred] == truth)).sum()),
                   "per_fold_gain": [round(g, 4) for g in per_fold]}
            report["overall"].append(row)
            print(json.dumps(row), flush=True)

pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in margins.items()}).assign(id=train["id"].to_numpy()).to_parquet("/kaggle/working/margins.parquet")
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
best = max(report["overall"], key=lambda r: r["gain"])
print("best", json.dumps(best), f"({time.time() - start:.0f} s)", flush=True)
