"""Loader check (#28): Zbigniew Kirchner's in-style Qwen3-8B LoRA through our own loader.

The checkpoint saw every train passage, so train is a check of the loader
(his fold-0 score is 0.516), not a score. It also settles which end token
he trained with: a wrong one would show as a big drop.
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
from nthl.lora24 import find_base, read_zbigniew, run_spec
from nthl.metric import balanced_accuracy

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
test["text"] = test["text"].fillna("").astype(str)
texts = train["text"].tolist() + test["text"].tolist()
base = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})

hits = glob.glob("/kaggle/input/**/final/readouts.npz", recursive=True)
assert len(hits) == 1, hits
local = "/tmp/zbigniew_final"
shutil.copytree(os.path.dirname(hits[0]), local, dirs_exist_ok=True)
probs, _, info = run_spec(read_zbigniew(local), base, texts)
pred = np.asarray(LABELS)[probs[: len(train)].argmax(1)]
truth = train["model"].to_numpy()
summary = {"insample_balanced_accuracy": balanced_accuracy(truth, pred),
           "claude_recall": {m: float(np.mean(pred[truth == m] == m)) for m in LABELS if m.startswith("claude")},
           "seconds": round(time.time() - T0), **info}
np.save("/kaggle/working/zbigniew_probs.npy", probs)
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
