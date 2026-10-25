"""SVC run pool122k: row 11 (pool weight 0.25, C 0.5) with nuridudak's 122k public pool (CC BY-SA 4.0)
in place of our 84k pool. The two sources we leave out of our own pool for licence reasons are dropped
here too. CV only, no submission."""

import glob
import json
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, code
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.folds import load_folds
from nthl.labels import LABELS
from nthl.svc import run

COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
path = glob.glob("/kaggle/input/**/ntl_encoder_pool_122k.parquet", recursive=True)
assert len(path) == 1, path
pool = pd.read_parquet(path[0])
print("pool", pool.shape, list(pool.columns), flush=True)
src = pool["source"].astype(str) if "source" in pool else pd.Series("", index=pool.index)
excluded = src.str.contains("olegzholobov|ankiteshtiwari", case=False)
print("dropped for licence:", int(excluded.sum()), flush=True)
pool = pool[~excluded & pool["model"].isin(LABELS)]
pool = pool[~pool["text"].isin(set(train["text"]))][["text", "model"]].reset_index(drop=True)
print("pool used", len(pool), pool["model"].value_counts().to_dict(), flush=True)

summary = run(train, test, pool=pool, pool_weight=0.25, C=0.5, out="/kaggle/working", name="pool122k")
print(json.dumps(summary, indent=1))
