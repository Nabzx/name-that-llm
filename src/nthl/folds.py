"""Prompt-grouped cross-validation folds.

Train has one passage per model per prompt, and test prompts never appear
in train. So a fold must hold whole prompts: GroupKFold on prompt_id. The
prompt order is shuffled with a fixed seed, then prompts are dealt into
folds in turn, which gives 30 prompts per fold.

Every experiment reads the frozen folds.csv. Changing it needs an ADR.

    PYTHONPATH=src python -m nthl.folds data/raw/train.csv   # rebuild folds.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_FOLDS = 5
SEED = 42
FOLDS_CSV = Path(__file__).resolve().parent / "folds.csv"


def build_folds(train, n_folds=N_FOLDS, seed=SEED):
    prompts = np.array(sorted(train["prompt_id"].unique()))
    order = np.random.default_rng(seed).permutation(len(prompts))
    fold_of = {prompts[j]: i % n_folds for i, j in enumerate(order)}
    folds = train[["id", "prompt_id"]].copy()
    folds["fold"] = folds["prompt_id"].map(fold_of).astype("int64")
    return folds.sort_values("id").reset_index(drop=True)


def load_folds():
    return pd.read_csv(FOLDS_CSV)


if __name__ == "__main__":
    build_folds(pd.read_csv(sys.argv[1])).to_csv(FOLDS_CSV, index=False)
    print(f"wrote {FOLDS_CSV}")
