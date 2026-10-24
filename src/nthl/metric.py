"""Balanced accuracy, as Kaggle scores this competition.

The mean, over the models that appear in the truth, of the share of that
model's passages labelled correctly. A model that is only ever predicted,
never true, adds no term.
"""

import numpy as np


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "truth and predictions differ in length"
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))
