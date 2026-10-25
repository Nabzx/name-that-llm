import numpy as np
import pytest

from nthl.encoders122k import NURIDUDAK_ORDER, seqcls_labels
from nthl.labels import LABELS
from nthl.lora24 import to_our_order


def _cfg(names):
    return {"id2label": {str(i): n for i, n in enumerate(names)}}


def test_nuridudak_configs_are_alphabetical_not_ours():
    # Both config.json files list the labels sorted by name; ours follow the competition page.
    assert list(NURIDUDAK_ORDER) == sorted(LABELS)
    assert list(NURIDUDAK_ORDER) != list(LABELS)


def test_labels_are_read_from_the_config():
    assert seqcls_labels(_cfg(NURIDUDAK_ORDER)) == list(NURIDUDAK_ORDER)


def test_reordered_columns_land_on_the_right_model():
    labels = seqcls_labels(_cfg(NURIDUDAK_ORDER))
    p = np.full((18, 18), 0.01 / 17)
    p[np.arange(18), np.arange(18)] = 0.99  # row i is confident in the checkpoint's column i
    q = to_our_order(p, labels)
    for i in range(18):
        assert LABELS[q[i].argmax()] == NURIDUDAK_ORDER[i]
    assert np.allclose(q.sum(1), 1)
    # gpt-5.5 is column 11 in the checkpoint and column 0 for us.
    assert q[11, 0] == 0.99


def test_a_config_with_other_labels_fails():
    with pytest.raises(AssertionError):
        seqcls_labels(_cfg([f"model-{i}" for i in range(18)]))
