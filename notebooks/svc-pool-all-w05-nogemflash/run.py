"""SVC run pool-all-w05-nogemflash (#26). Config: {'pool': True, 'pool_weight': 0.5, 'routes': None, 'C': 0.5, 'drop_pool_labels': ['gemini-3.7-flash', 'gemini-3.8-flash']}"""

import glob
import json
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.folds import load_folds
from nthl.svc import run

CONFIG = {'pool': True, 'pool_weight': 0.5, 'routes': None, 'C': 0.5, 'drop_pool_labels': ['gemini-3.7-flash', 'gemini-3.8-flash']}
COMP = "/kaggle/input/competitions/name-that-llm"

train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
assert len(train) == 2699

pool = None
if CONFIG.pop("pool"):
    path = glob.glob("/kaggle/input/**/pool.parquet", recursive=True)
    assert len(path) == 1, f"expected one pool.parquet, found {path}"
    pool = pd.read_parquet(path[0])

summary = run(train, test, pool=pool, out="/kaggle/working", name="pool-all-w05-nogemflash", **CONFIG)
print(json.dumps(summary, indent=1))
