import numpy as np
import pandas as pd
import pytest

from nthl.labels import LABELS
from nthl.preds import check_preds, load_preds, save_preds, to_labels


def make_preds(n=4, seed=0):
    p = np.random.default_rng(seed).random((n, len(LABELS)))
    df = pd.DataFrame(p / p.sum(1, keepdims=True), columns=list(LABELS))
    df.insert(0, "id", [f"id{i}" for i in range(n)])
    return df


def test_a_valid_frame_passes():
    check_preds(make_preds())


def test_a_missing_label_column_fails():
    with pytest.raises(AssertionError):
        check_preds(make_preds().drop(columns=[LABELS[0]]))


def test_columns_in_the_wrong_order_fail():
    df = make_preds()
    with pytest.raises(AssertionError):
        check_preds(df[["id"] + list(reversed(LABELS))])


def test_a_nan_fails():
    df = make_preds()
    df.loc[0, LABELS[3]] = np.nan
    with pytest.raises(AssertionError):
        check_preds(df)


def test_a_repeated_id_fails():
    df = make_preds()
    df.loc[1, "id"] = df.loc[0, "id"]
    with pytest.raises(AssertionError):
        check_preds(df)


def test_rows_that_do_not_sum_to_one_fail():
    df = make_preds()
    df.loc[0, LABELS[0]] += 0.5
    with pytest.raises(AssertionError):
        check_preds(df)


def test_expected_ids_must_match_exactly():
    df = make_preds()
    check_preds(df, ids=df["id"])
    with pytest.raises(AssertionError):
        check_preds(df, ids=list(df["id"]) + ["extra"])


def test_to_labels_takes_the_most_likely_model():
    df = make_preds()
    df.loc[2, LABELS] = 0.0
    df.loc[2, "kimi-k3"] = 1.0
    assert to_labels(df)[2] == "kimi-k3"


def test_save_and_load_round_trip(tmp_path):
    df = make_preds()
    path = tmp_path / "exp" / "oof.parquet"
    save_preds(df, path)
    pd.testing.assert_frame_equal(load_preds(path), df)
