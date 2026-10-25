"""Loader check (#43): our own Qwen3-8B LoRA24 from nabilshah/nthl-train-lora on host train.

ours_final saw every train passage in stage 2, so its train score checks the
loader, not the model. ckpt_stage1 only saw the public pool, so its train
score is an honest out-of-sample number.
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
from nthl.lora24 import find_base, read_ours, run_spec
from nthl.metric import balanced_accuracy

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv")
truth = train["model"].to_numpy()
base = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})

train_log = glob.glob("/kaggle/input/**/train_log.json", recursive=True)
summary = {"train_log_tail": json.load(open(train_log[0]))[-3:] if len(train_log) == 1 else None}
for name, held_out in [("ckpt_stage1", True), ("ours_final", False)]:
    meta = glob.glob(f"/kaggle/input/**/{name}/meta.json", recursive=True)
    assert len(meta) == 1, meta
    local = f"/tmp/{name}"
    shutil.copytree(os.path.dirname(meta[0]), local, dirs_exist_ok=True)
    probs, _, info = run_spec(read_ours(local), base, train["text"].tolist())
    pred = np.asarray(LABELS)[probs.argmax(1)]
    key = "balanced_accuracy" if held_out else "insample_balanced_accuracy"
    summary[name] = {key: balanced_accuracy(truth, pred),
                     "recall": {m: round(float(np.mean(pred[truth == m] == m)), 3) for m in LABELS}, **info}
    np.save(f"/kaggle/working/{name}_train_probs.npy", probs)
    print(name, json.dumps(summary[name]), flush=True)

summary["seconds"] = round(time.time() - T0)
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
