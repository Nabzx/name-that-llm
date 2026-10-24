"""Smoke test: does our code run on Kaggle? No model is fitted.

Prints where Kaggle mounts each input, imports nthl from the private code
dataset, checks the frozen folds against the competition train file, and
records the library versions and GPUs we get.
"""

import glob
import importlib
import platform
import shutil
import subprocess
import sys

for depth in ("*", "*/*", "*/*/*"):
    for path in sorted(glob.glob(f"/kaggle/input/{depth}")):
        print("input:", path)

# Kaggle unpacks the code dataset flat, so copy it into a folder named nthl.
code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.folds import load_folds
from nthl.labels import LABELS

train_path = glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True)
assert len(train_path) == 1, train_path
train = pd.read_csv(train_path[0])
folds = load_folds()
assert set(folds["id"]) == set(train["id"]), "frozen folds do not match the competition train file"
assert set(LABELS) == set(train["model"])
print("train:", train_path[0], train.shape, "folds:", folds.groupby("fold").size().to_dict())

print("python", platform.python_version())
for name in ["numpy", "pandas", "sklearn", "pyarrow", "torch", "transformers", "peft", "accelerate", "bitsandbytes", "lightgbm"]:
    try:
        print(name, importlib.import_module(name).__version__)
    except ImportError:
        print(name, "missing")

import torch

print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)

folds.groupby("fold").size().rename("passages").to_csv("/kaggle/working/smoke.csv")
print("done")
