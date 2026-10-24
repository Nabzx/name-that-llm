import numpy as np
import pandas as pd

from nthl.labels import LABELS
from nthl.preds import load_preds
from nthl.svc import run


def toy(n_per=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, label in enumerate(LABELS):
        for j in range(n_per):
            words = rng.choice(["alpha", "beta", "gamma", "delta", "omega"], 30)
            rows.append({"id": f"{i}-{j}", "text": f"marker{i} " + " ".join(words), "model": label, "fold": j % 5})
    return pd.DataFrame(rows)


def test_run_writes_valid_files_and_learns_an_easy_task(tmp_path):
    train = toy()
    test = toy(n_per=1, seed=1).drop(columns=["model", "fold"]).assign(id=lambda d: "t" + d["id"])
    summary = run(train, test, out=tmp_path, name="toy")
    assert summary["cv"] > 0.9
    oof = load_preds(tmp_path / "oof.parquet")
    assert set(oof["id"]) == set(train["id"])
    assert len(load_preds(tmp_path / "test.parquet")) == len(test)


def test_run_keeps_only_the_asked_routes(tmp_path):
    train = toy()
    test = toy(n_per=1, seed=1).drop(columns=["model", "fold"]).assign(id=lambda d: "t" + d["id"])
    pool = toy(n_per=4, seed=2).drop(columns=["id", "fold"])
    pool["source"] = "x/y"
    pool["route"] = np.where(np.arange(len(pool)) % 2 == 0, "api", "proxy")
    summary = run(train, test, pool=pool, routes=["api"], pool_weight=0.5, out=tmp_path, name="toy")
    assert summary["pool_rows"] == (pool["route"] == "api").sum()
    assert summary["pool_weight"] == 0.5


def test_fit_scores_gives_raw_scores_in_our_label_order():
    from nthl.svc import fit_scores

    train = toy()
    test = toy(n_per=1, seed=1)
    scores = fit_scores(train, test["text"])
    assert scores.shape == (18, 18)
    assert (np.asarray(LABELS)[scores.argmax(1)] == test["model"].to_numpy()).mean() > 0.9
