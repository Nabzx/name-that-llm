"""SVC run pool-w025-c025 (#26). Config: {'pool': True, 'pool_weight': 0.25, 'routes': None, 'C': 0.25, 'min_df': 2}"""

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

CONFIG = {'pool': True, 'pool_weight': 0.25, 'routes': None, 'C': 0.25, 'min_df': 2}
COMP = "/kaggle/input/competitions/name-that-llm"

train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
assert len(train) == 2699

pool = None
if CONFIG.pop("pool"):
    path = glob.glob("/kaggle/input/**/pool.parquet", recursive=True)
    assert len(path) == 1, f"expected one pool.parquet, found {path}"
    pool = pd.read_parquet(path[0])

summary = run(train, test, pool=pool, out="/kaggle/working", name="pool-w025-c025", **CONFIG)
print(json.dumps(summary, indent=1))
