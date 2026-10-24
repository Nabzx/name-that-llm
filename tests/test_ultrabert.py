import json

import numpy as np
import pytest

from nthl.labels import LABELS
from nthl.ultrabert import ckpt_labels, find_folds, to_our_order


def _fold(root, name, model_type="ultrabert"):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return d


def test_folds_are_found_in_numeric_order(tmp_path):
    for k in [10, 2, 0, 1, 3]:
        _fold(tmp_path, f"fold{k}")
    (tmp_path / "wheels").mkdir()
    assert [p.name for p in find_folds(tmp_path, n=5)] == ["fold0", "fold1", "fold2", "fold3", "fold10"]


def test_folds_are_found_below_a_nested_mount(tmp_path):
    for k in range(5):
        _fold(tmp_path / "ultrabert-380m-ntl", f"fold{k}")
    assert len(find_folds(tmp_path)) == 5


def test_a_missing_fold_or_other_model_fails(tmp_path):
    for k in range(4):
        _fold(tmp_path, f"fold{k}")
    _fold(tmp_path, "fold4", model_type="bert")
    with pytest.raises(AssertionError):
        find_folds(tmp_path)


def test_labels_follow_the_integer_keys_not_the_json_order():
    # Keys stored as strings and out of order; "10" sorts before "2" as text.
    names = sorted(LABELS)
    cfg = {"id2label": {str(i): names[i] for i in reversed(range(18))}}
    assert ckpt_labels(cfg) == names


def test_labels_must_be_ours():
    cfg = {"id2label": {str(i): f"model-{i}" for i in range(18)}}
    with pytest.raises(AssertionError):
        ckpt_labels(cfg)


def test_columns_come_back_in_our_order():
    names = sorted(LABELS)  # the checkpoint's alphabetical order
    q = to_our_order(np.eye(18), names)  # row i puts all mass on the checkpoint's column i
    for i in range(18):
        assert LABELS[q[i].argmax()] == names[i]
