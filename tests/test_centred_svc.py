import numpy as np
from scipy.sparse import csr_matrix

from nthl.labels import LABELS
from nthl.svc import centred_scores, fit_centred, group_centre


def test_group_centre_removes_the_group_mean_and_leaves_other_rows():
    x = csr_matrix(np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 5.0]]))
    out = group_centre(x, np.array([1, 1, 2]), rows=np.array([True, True, False])).toarray()
    assert np.allclose(out[:2], [[-1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(out[2], [5.0, 5.0])


def test_centred_svc_scores_come_back_in_our_label_order():
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "notably", "indeed", "thus", "moreover"]
    texts, labels, prompts = [], [], []
    for p in range(4):
        for i, m in enumerate(LABELS):
            texts.append(" ".join(rng.choice(words, 12)) + f" style{i} topic{p}")
            labels.append(m)
            prompts.append(p)
    vec, svc = fit_centred(texts, labels, np.array(prompts), C=0.5, min_df=1)
    groups = np.array([0, 0, 1])
    out = centred_scores(vec, svc, texts[:3], groups, np.array([True, True, False]))
    assert out.shape == (3, len(LABELS))
    assert np.all(out[2] == 0)            # rows outside a group get no twin score
    assert out[0].argmax() == 0 and out[1].argmax() == 1
