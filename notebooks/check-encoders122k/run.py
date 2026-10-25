"""Loader check (#28): nuridudak's two-stage ModernBERT-large and desklib DeBERTa-v3-large.

Runs host train and test through our own loader. Stage 2 trained on all of
host train, so the train score is in-sample, a check of the loader and the
label order, not OOF. The author's prompt-grouped OOF is 0.569 (ModernBERT)
and 0.523 (desklib). Test probabilities go to members/<name>.npy, rows in
test.csv order, columns in LABELS order, like the submit notebook.
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
import transformers

from nthl.encoders122k import run_seqcls
from nthl.labels import LABELS
from nthl.metric import balanced_accuracy

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/members", exist_ok=True)
train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
train["text"] = train["text"].fillna("").astype(str)
test["text"] = test["text"].fillna("").astype(str)
root = glob.glob("/kaggle/input/**/ntl-two-stage-encoders", recursive=True)
assert len(root) == 1, root
truth = train["model"].to_numpy()
texts = train["text"].tolist() + test["text"].tolist()
test[["id"]].to_csv(f"{OUT}/members/ids.csv", index=False)

summary = {"transformers": transformers.__version__, "torch": torch.__version__,
           "train_rows": len(train), "test_rows": len(test)}
for name in ["modernbert", "desklib"]:
    t = time.time()
    probs = run_seqcls(os.path.join(root[0], name), texts)
    seconds = time.time() - t
    tr, te = probs[: len(train)], probs[len(train):]
    assert np.allclose(probs.sum(1), 1, atol=1e-4)
    np.save(f"{OUT}/members/{name}.npy", te)
    np.save(f"{OUT}/members/{name}_train_insample.npy", tr)
    pred = np.asarray(LABELS)[tr.argmax(1)]
    summary[name] = {
        "insample_balanced_accuracy": round(balanced_accuracy(truth, pred), 4),
        "recall": {m: round(float(np.mean(pred[truth == m] == m)), 3) for m in LABELS},
        "test_pred_counts": {k: int(v) for k, v in pd.Series(np.asarray(LABELS)[te.argmax(1)]).value_counts().items()},
        "seconds_incl_copy": round(seconds),
        "seconds_per_1000": round(1000 * seconds / len(texts), 1),
    }
    print(name, json.dumps(summary[name]), flush=True)

summary["seconds"] = round(time.time() - T0)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
