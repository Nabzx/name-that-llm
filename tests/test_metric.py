from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from nthl.labels import LABELS
from nthl.metric import balanced_accuracy

TRAIN = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv"


def test_there_are_18_unique_labels():
    assert len(LABELS) == 18
    assert len(set(LABELS)) == 18


@pytest.mark.skipif(not TRAIN.exists(), reason="needs data/raw/train.csv")
def test_labels_match_the_train_set():
    assert set(LABELS) == set(pd.read_csv(TRAIN)["model"])


def test_balanced_accuracy_on_a_small_case():
    # A: 1 of 2 right, B: 1 of 1 right. Plain accuracy would be 2/3.
    y_true = ["A", "A", "B"]
    y_pred = ["A", "B", "B"]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_balanced_accuracy_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.choice(LABELS, 500, p=np.linspace(1, 3, 18) / np.linspace(1, 3, 18).sum())
    y_pred = np.where(rng.random(500) < 0.6, y_true, rng.choice(LABELS, 500))
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_a_class_only_predicted_does_not_count():
    # C is never true, so it adds no recall term. Kaggle averages over true models.
    assert balanced_accuracy(["A", "B"], ["A", "C"]) == pytest.approx(0.5)
