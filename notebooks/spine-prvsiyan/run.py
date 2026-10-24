"""Loader check (#28): prvsiyan's two Qwen3-8B seeds through our own loader.

Runs host train and test. The seeds saw every train passage, so train is a
check of the loader (their own OOF is 0.568), not a score.
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

from nthl.labels import LABELS
from nthl.lora24 import find_base, read_prvsiyan, run_spec
from nthl.metric import balanced_accuracy

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
test["text"] = test["text"].fillna("").astype(str)
texts = train["text"].tolist() + test["text"].tolist()
base = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})

summary = {}
for seed in ["full_s0", "full_s1"]:
    meta = glob.glob(f"/kaggle/input/**/q8b_L24_ctr/{seed}/meta.json", recursive=True)
    assert len(meta) == 1, meta
    local = f"/tmp/{seed}"
    shutil.copytree(os.path.dirname(meta[0]), local, dirs_exist_ok=True)
    probs, _, info = run_spec(read_prvsiyan(local), base, texts)
    pred = np.asarray(LABELS)[probs[: len(train)].argmax(1)]
    truth = train["model"].to_numpy()
    summary[seed] = {"insample_balanced_accuracy": balanced_accuracy(truth, pred),
                     "claude_recall": {m: float(np.mean(pred[truth == m] == m)) for m in LABELS if m.startswith("claude")}, **info}
    np.save(f"/kaggle/working/{seed}_probs.npy", probs)
    print(seed, json.dumps(summary[seed]), flush=True)

summary["seconds"] = round(time.time() - T0)
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
