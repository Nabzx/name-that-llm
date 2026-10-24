"""Spine step 1 (#27): Nishant Dahal's Qwen3-14B LoRA checkpoint through our own loader.

Runs train and test in one pass so the 14B trunk loads once. The checkpoint
saw every train passage, so its train predictions are a smoke test of the
loader, not a score: a wrong label order or end token would show as a big
drop. Saves insample.parquet (train, not out of fold) and test.parquet.
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
from nthl.lora24 import find_base, run_checkpoint
from nthl.metric import balanced_accuracy
from nthl.preds import save_preds, to_labels

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
start = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
test["text"] = test["text"].fillna("").astype(str)
print("gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)

meta = glob.glob("/kaggle/input/**/lora_qwen3_14b/meta.json", recursive=True)
assert len(meta) == 1, meta
ckpt = "/tmp/lora_qwen3_14b"
shutil.copytree(os.path.dirname(meta[0]), ckpt, dirs_exist_ok=True)
base = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40})
print("checkpoint", meta[0], "base", base, f"{time.time() - start:.0f} s", flush=True)

texts = train["text"].tolist() + test["text"].tolist()
probs, info = run_checkpoint(ckpt, base, texts)
print(json.dumps(info), flush=True)


def frame(ids, p):
    df = pd.DataFrame(p, columns=list(LABELS))
    df.insert(0, "id", ids.to_numpy())
    return df


insample = frame(train["id"], probs[: len(train)])
save_preds(insample, "/kaggle/working/insample.parquet")
save_preds(frame(test["id"], probs[len(train):]), "/kaggle/working/test.parquet")

pred = to_labels(insample)
truth = train["model"].to_numpy()
summary = {
    "member": "nishant lora_qwen3_14b",
    "insample_balanced_accuracy": balanced_accuracy(truth, pred),
    "insample_recall": {m: float(np.mean(pred[truth == m] == m)) for m in LABELS},
    "passages": len(texts),
    "seconds": round(time.time() - start),
    **info,
}
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
