"""Loader check (#28): pjmathematician's five UltraBERT-380M folds through our own loader.

Runs host train and test. Each fold saw 4/5 of train, so the train score is
in-sample for most passages, a check of the loader, not OOF. The authors'
OOF is 0.408. The dataset ships no OOF predictions to compare against.
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

from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.preds import save_preds
from nthl.ultrabert import run_ultrabert

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
train["text"] = train["text"].fillna("").astype(str)
test["text"] = test["text"].fillna("").astype(str)
root = glob.glob("/kaggle/input/**/ultrabert-380m-ntl", recursive=True)
assert len(root) == 1, root

t = time.time()
probs = run_ultrabert(root[0], train["text"].tolist() + test["text"].tolist())
seconds = round(time.time() - t)
tr, te = probs[: len(train)], probs[len(train):]
pred = np.asarray(LABELS)[tr.argmax(1)]
truth = train["model"].to_numpy()
summary = {
    "transformers": transformers.__version__,
    "insample_balanced_accuracy": balanced_accuracy(truth, pred),
    "claude_recall": {m: float(np.mean(pred[truth == m] == m)) for m in LABELS if m.startswith("claude")},
    "passages": len(probs),
    "inference_seconds": seconds,
    "seconds_per_1000": round(1000 * seconds / len(probs), 1),
}


def frame(ids, p):
    df = pd.DataFrame(p, columns=list(LABELS))
    df.insert(0, "id", ids.to_numpy())
    return df


save_preds(frame(train["id"], tr), "/kaggle/working/train_insample.parquet")
save_preds(frame(test["id"], te), "/kaggle/working/test.parquet")
summary["seconds"] = round(time.time() - T0)
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
