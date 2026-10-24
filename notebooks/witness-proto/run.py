"""Witness prototype (#34): is there signal in how likely each lab's open model finds a passage?

One likelihood witness, Qwen3-4B-Base (Qwen, fp16), plus Gemma 4's tokenizer for Google (Gemma's weights do not fit on the T4s next to its per-layer embedding table), score host
train and test. A logistic-regression head on their features is fitted per
frozen fold, here on Kaggle, giving honest out-of-fold predictions. Then the
question that matters: does it add to our SVC's out-of-fold predictions?
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
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nthl.folds import load_folds
from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.witness import FEATURES, score_witness

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"
train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
test["text"] = test["text"].fillna("").astype(str)
texts = train["text"].tolist() + test["text"].tolist()


def model_dir(fragment):
    hits = sorted({os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{fragment}/**/config.json", recursive=True)})
    assert hits, f"no model under {fragment}"
    return hits[0]


WITNESSES = {"qwen3_4b": ("4b-base", torch.float16)}
cols, blocks, timing = [], [], {}
for name, (fragment, dtype) in WITNESSES.items():
    feats, tpw = score_witness(model_dir(fragment), texts, dtype=dtype)
    blocks += [feats, tpw[:, None]]
    cols += [f"{name}_{f}" for f in FEATURES] + [f"{name}_tokens_per_word"]
    timing[name] = round(time.time() - T0)
    print(name, "done at", timing[name], "s", flush=True)

# Gemma 4's weights do not fit beside its 8.75 GiB per-layer embedding table, so Google joins
# through its tokenizer only: tokens per word, no model loaded.
from transformers import AutoTokenizer

from nthl.witness import tokens_per_word

blocks.append(np.asarray(tokens_per_word(AutoTokenizer.from_pretrained(model_dir("gemma-4-e2b")), texts), np.float32)[:, None])
cols.append("gemma4_tokens_per_word")
x = pd.DataFrame(np.hstack(blocks), columns=cols)
x.insert(0, "id", train["id"].tolist() + test["id"].tolist())
x.to_parquet(f"{OUT}/witness_features.parquet", index=False)
xt = x.iloc[: len(train), 1:].to_numpy()
y = train["model"].to_numpy()

oof = np.zeros((len(train), len(LABELS)))
for k in range(5):
    tr, va = (train["fold"] != k).to_numpy(), (train["fold"] == k).to_numpy()
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(xt[tr], y[tr])
    oof[va] = head.predict_proba(xt[va])[:, [list(head.classes_).index(m) for m in LABELS]]
wit_pred = np.asarray(LABELS)[oof.argmax(1)]
wit_cv = balanced_accuracy(y, wit_pred)

svc = pd.read_parquet(glob.glob("/kaggle/input/**/nthl-svc-pool-all-w05/oof.parquet", recursive=True)[0]).set_index("id").loc[train["id"], list(LABELS)].to_numpy()
svc_cv = balanced_accuracy(y, np.asarray(LABELS)[svc.argmax(1)])
curve = {}
for w in np.round(np.arange(0, 1.55, 0.05), 2):
    blend = np.log(np.clip(svc, 1e-9, 1)) + w * np.log(np.clip(oof, 1e-9, 1))
    curve[float(w)] = balanced_accuracy(y, np.asarray(LABELS)[blend.argmax(1)])
best_w = max(curve, key=curve.get)

lab = lambda m: m.split("-")[0].rstrip("0123456789.")
labs = np.array([lab(m) for m in y])
summary = {
    "witness_cv": wit_cv,
    "witness_lab_accuracy": float(np.mean([lab(p) == l for p, l in zip(wit_pred, labs)])),
    "svc_cv": svc_cv,
    "blend_best_weight": best_w,
    "blend_best_cv": curve[best_w],
    "blend_gain": curve[best_w] - svc_cv,
    "blend_curve": curve,
    "witness_recall": {m: float(np.mean(wit_pred[y == m] == m)) for m in LABELS},
    "timing": timing,
    "seconds": round(time.time() - T0),
}
pd.DataFrame(oof, columns=list(LABELS)).assign(id=train["id"].to_numpy()).to_parquet(f"{OUT}/witness_oof.parquet", index=False)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
