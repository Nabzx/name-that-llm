"""Check (#28, #46): lightningv08's full-text Qwen3.8-27B classifier through our loader. No submission.

JOBS picks the work, in this order (one GPU kernel at a time, OOF first):
- "oof": the outer0 adapter on the 539 train rows it held out (prompt-disjoint,
  the bundle's own outer0 split), an honest score; then the final adapter on
  the same rows, in sample, as a loader check.
- "test": the final adapter on the 4,000 test passages, saved as members/qwen27b.npy.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import json
import shutil
import sys
import time

JOBS = ["test"]

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

from nthl.qwen27b import find_assets, heldout_rows, install_wheels

T0 = time.time()
assets = find_assets("/kaggle/input")
install_wheels(assets["wheels"])

from importlib import metadata

import numpy as np
import pandas as pd
import torch

from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.qwen27b import Qwen27B

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
COMP = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/members", exist_ok=True)
summary = {"jobs": JOBS, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
summary["wheel_licences"] = {}
for dist in ["accelerate", "bitsandbytes", "huggingface_hub", "safetensors", "tokenizers", "transformers"]:
    m = metadata.metadata(dist)
    summary["wheel_licences"][dist] = [m["Version"], m.get("License-Expression") or (m.get("License") or "")[:60],
                                       [c for c in m.get_all("Classifier") or [] if c.startswith("License")]]
print(json.dumps(summary), flush=True)


def score(name, probs, truth):
    pred = np.asarray(LABELS)[probs.argmax(1)]
    summary[name] = {"balanced_accuracy": round(balanced_accuracy(truth, pred), 4),
                     "recall": {m: round(float(np.mean(pred[truth == m] == m)), 3) for m in LABELS}, **model.last}
    print(name, json.dumps(summary[name]), flush=True)


if "oof" in JOBS:
    train = pd.read_csv(f"{COMP}/train.csv")
    rows = heldout_rows(assets["fold_plan"], f"{COMP}/train.csv", train["prompt_id"].to_numpy())
    texts, truth = train["text"].fillna("").astype(str).iloc[rows].tolist(), train["model"].to_numpy()[rows]
    model = Qwen27B(assets["base"], assets["outer0"])
    summary["load"] = model.info
    p = model.predict(texts)
    np.save(f"{OUT}/outer0_heldout_probs.npy", p)
    np.save(f"{OUT}/outer0_heldout_rows.npy", rows)
    score("outer0_heldout", p, truth)
    model.load_adapter(assets["final"])
    p = model.predict(texts)
    np.save(f"{OUT}/final_heldout_probs.npy", p)
    score("final_insample", p, truth)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

if "test" in JOBS:
    test = pd.read_csv(f"{COMP}/test.csv")
    if "model" not in globals():
        model = Qwen27B(assets["base"], assets["final"])
        summary["load"] = model.info
    else:
        model.load_adapter(assets["final"])
    p = model.predict(test["text"].fillna("").astype(str).tolist())
    assert p.shape == (len(test), 18) and np.allclose(p.sum(1), 1, atol=1e-4)
    np.save(f"{OUT}/members/qwen27b.npy", p)
    pd.Series(test["id"].astype(str)).to_csv(f"{OUT}/members/qwen27b_ids.csv", index=False)
    summary["test"] = {**model.last, "predicted": pd.Series(np.asarray(LABELS)[p.argmax(1)]).value_counts().to_dict()}

summary["seconds"] = round(time.time() - T0)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
