"""Sinkhorn balancing on our out-of-fold predictions, balanced and unbalanced (#23).

Scores balanced accuracy with no balancing, full Sinkhorn (every model gets
1/18 of the probability mass) and half Sinkhorn (target halfway between the
current mass and 1/18). Test mixes are simulated by letting each model keep a
random share of its OOF rows. A lean multiplies each model's column by a
random factor first, the kind of prior a pool-trained member carries onto the
test. The split check balances the whole set once, then scores two parts with
different model mixes, as Kaggle does with the public and private LB.
Analysis of saved predictions only: nothing is fitted.

    PYTHONPATH=src python scripts/sinkhorn_sim.py OOF_PARQUET TRAIN_CSV
"""

import sys

import numpy as np
import pandas as pd

from nthl.labels import LABELS
from nthl.metric import balanced_accuracy

SEED, DRAWS, K = 23, 200, len(LABELS)
FLOORS = (1.0, 0.7, 0.5, 0.3, 0.1)  # each model keeps a share of its rows drawn from [floor, 1]
LEANS = (0.0, 0.3)  # standard deviation of the log lean per model


def sinkhorn(P, alpha, iters=300):
    """Labels after scaling each column towards the target mass."""
    target = (1 - alpha) * P.sum(0) + alpha * len(P) / K
    c = np.ones(K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    return (P * c).argmax(1)


def leaned(P, lean, rng):
    P = P * np.exp(rng.normal(0, lean, K))
    return P / P.sum(1, keepdims=True)


def main(oof_path, train_path):
    oof = pd.read_parquet(oof_path)
    truth = pd.read_csv(train_path).set_index("id").loc[oof["id"], "model"]
    y = truth.map({m: i for i, m in enumerate(LABELS)}).to_numpy()
    P0 = oof[list(LABELS)].to_numpy()
    rng = np.random.default_rng(SEED)
    out = []
    for lean in LEANS:
        for floor in FLOORS:
            res = []
            for _ in range(DRAWS):
                keep = rng.uniform(floor, 1.0, K)
                idx = np.concatenate([rng.permutation(np.flatnonzero(y == k))[: round(keep[k] * (y == k).sum())]
                                      for k in range(K)])
                P, t = leaned(P0[idx], lean, rng), y[idx]
                b = balanced_accuracy(t, P.argmax(1))
                res.append([len(idx), b] + [balanced_accuracy(t, sinkhorn(P, a)) - b for a in (1.0, 0.5)])
            r = np.array(res)
            out.append(dict(lean=lean, floor=floor, rows=int(r[:, 0].mean()), none=r[:, 1].mean(),
                            full=r[:, 2].mean(), half=r[:, 3].mean(),
                            full_hurts=(r[:, 2] < 0).mean(), half_hurts=(r[:, 3] < 0).mean()))
    print(pd.DataFrame(out).round(4).to_string(index=False))

    for lean in LEANS:  # split check: public takes 20% to 60% of each model's rows
        res = []
        for _ in range(DRAWS):
            share = rng.uniform(0.2, 0.6, K)
            pub = np.zeros(len(y), bool)
            for k in range(K):
                rows = rng.permutation(np.flatnonzero(y == k))
                pub[rows[: round(share[k] * len(rows))]] = True
            P = leaned(P0, lean, rng)
            plain, bal = P.argmax(1), sinkhorn(P, 1.0)
            res.append([balanced_accuracy(y[m], bal[m]) - balanced_accuracy(y[m], plain[m]) for m in (pub, ~pub)])
        r = np.array(res)
        print(f"split lean {lean}: public {r[:, 0].mean():+.4f} (hurts {(r[:, 0] < 0).mean():.2f}), "
              f"private {r[:, 1].mean():+.4f} (hurts {(r[:, 1] < 0).mean():.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
