"""One file format for every member's predictions.

A prediction file has an `id` column and one probability column per model,
in the order of `LABELS`. Rows sum to 1. Each experiment writes two files:

    outputs/<experiment>/oof.parquet    out-of-fold, one row per train passage
    outputs/<experiment>/test.parquet   one row per test passage

so any member can be blended with any other later.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nthl.labels import LABELS

COLUMNS = ["id", *LABELS]


def check_preds(df, ids=None):
    assert list(df.columns) == COLUMNS, "columns must be id then the 18 labels in order"
    assert df["id"].is_unique, "repeated id"
    p = df[list(LABELS)].to_numpy(dtype=float)
    assert np.isfinite(p).all(), "missing or infinite probability"
    assert (p >= 0).all() and (p <= 1).all(), "probability outside 0 to 1"
    assert np.allclose(p.sum(axis=1), 1, atol=1e-4), "a row does not sum to 1"
    if ids is not None:
        assert len(df) == len(ids) and set(df["id"]) == set(ids), "ids differ from the expected set"
    return df


def to_labels(df):
    return np.asarray(LABELS)[df[list(LABELS)].to_numpy().argmax(axis=1)]


def save_preds(df, path):
    check_preds(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_preds(path):
    return check_preds(pd.read_parquet(path))
