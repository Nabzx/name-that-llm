"""A shared-prompt Gaussian group head on any feature view (#46).

Our own implementation of the method in prvsiyan's public notebook *The Prompt Is Not the Author*
(Apache 2.0). A passage's features are modelled as

    x = mean of its model + effect of its prompt + noise

with the prompt effect shared by every passage written for that prompt. In whitened coordinates the
noise has identity covariance and the prompt effect a diagonal one (`lam`). For a group of test
passages that probably share a prompt, the prompt effect is integrated out and the labels are decoded
jointly, using the fact that one model answers a prompt at most once. Pairs and triples are exact;
larger groups use a mean-field approximation.
"""

import itertools

import numpy as np
from scipy.special import log_softmax, logsumexp

K = 18


def _lw(x):
    from sklearn.covariance import ledoit_wolf_shrinkage

    return float(ledoit_wolf_shrinkage(x, assume_centered=True))


def _shrink(s, a):
    return (1 - a) * s + a * np.trace(s) / len(s) * np.eye(len(s))


def fit(x, y, prompts, d=256, iters=4):
    """Fit on training rows: features `x` [n, D], label indices `y` in 0..17, prompt ids `prompts`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    _, g = np.unique(np.asarray(prompts), return_inverse=True)
    n_g = np.bincount(g).astype(float)
    mx, sx = x.mean(0), x.std(0)
    sx = np.where(sx > 1e-8, sx, 1.0)
    xs = (x - mx) / sx

    # directions of within-prompt variation (prompt means removed), via the Gram matrix when n < D
    pm = np.zeros((len(n_g), xs.shape[1]))
    np.add.at(pm, g, xs)
    r = xs - (pm / n_g[:, None])[g]
    d = int(min(d, len(x) - len(n_g) - 1, xs.shape[1]))
    if len(x) < xs.shape[1]:
        w, u = np.linalg.eigh(r @ r.T)
        top = np.argsort(w)[::-1][:d]
        v = r.T @ u[:, top] / np.sqrt(np.clip(w[top], 1e-12, None))
    else:
        v = np.linalg.svd(r, full_matrices=False)[2][:d].T
    z = xs @ v

    # additive two-way fit z = mu[y] + p[prompt] + e
    p = np.zeros((len(n_g), d))
    mu = np.zeros((K, d))
    for _ in range(iters):
        res = z - p[g]
        for c in np.unique(y):
            mu[c] = res[y == c].mean(0)
        p = np.zeros_like(p)
        np.add.at(p, g, z - mu[y])
        p /= n_g[:, None]
    e = z - mu[y] - p[g]
    se_raw = e.T @ e / max(len(x) - len(n_g) - K + 1, 1)
    se = _shrink(se_raw, _lw(e))
    pc = p - p.mean(0)
    sp = pc.T @ pc / max(len(n_g) - 1, 1) - np.mean(1 / n_g) * se_raw
    a_p = _lw(pc)
    sp = (1 - a_p) * sp + a_p * max(np.trace(sp), 1e-8 * np.trace(se)) / d * np.eye(d)

    # joint diagonalisation: A se A' = I and A sp A' = diag(lam)
    li = np.linalg.inv(np.linalg.cholesky(se))
    m = li @ sp @ li.T
    lam, q = np.linalg.eigh(0.5 * (m + m.T))
    a = q.T @ li
    return {"mx": mx, "sx": sx, "W": v @ a.T, "m": mu @ a.T, "lam": np.clip(lam, 1e-4, None), "d": d}


def transform(x, prm):
    return ((np.asarray(x, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]


def _exact(r, gam, T):
    """Exact per-row log marginals for a small group. r is [n, K, d] (row minus each class mean)."""
    n = len(r)
    a = -0.5 * ((r ** 2) * (1 - gam)).sum(-1)                  # [n, K]
    table = np.zeros([K] * n)
    for i in range(n):
        shape = [1] * n
        shape[i] = K
        table = table + a[i].reshape(shape)
    for i, j in itertools.combinations(range(n), 2):
        shape = [1] * n
        shape[i] = shape[j] = K
        table = table + np.einsum("cd,ed->ce", r[i] * gam, r[j]).reshape(shape)
    table = table / T
    grids = np.meshgrid(*[np.arange(K)] * n, indexing="ij")
    for i, j in itertools.combinations(range(n), 2):
        table = np.where(grids[i] == grids[j], -np.inf, table)
    table -= logsumexp(table)
    return np.stack([logsumexp(table, axis=tuple(k for k in range(n) if k != i)) for i in range(n)])


def _mean_field(r, gam, T, iters):
    """Approximate log marginals for a larger group, with a soft one-label-per-model penalty."""
    a = -0.5 * ((r ** 2) * (1 - gam)).sum(-1)
    q = np.exp(log_softmax(a / T, axis=1))
    for _ in range(iters):
        rbar = np.einsum("ic,icd->id", q, r)                    # expected residual of each row
        logq = np.empty_like(a)
        for i in range(len(r)):
            others = np.delete(np.arange(len(r)), i)
            pull = r[i] @ (gam * rbar[others].sum(0))
            repel = np.log(np.clip(1 - q[others], 1e-6, None)).sum(0)
            logq[i] = log_softmax((a[i] + pull) / T + repel)
        q = np.exp(logq)
    return logq


def log_posteriors(u, prm, groups, rows, T=1.0, n_exact=3, iters=5):
    """Log posteriors [n, 18] for transformed rows `u`. Rows in `rows` are decoded with the other
    `rows` members of their group; every other row (and any group of one) is decoded alone."""
    m, lam = prm["m"], prm["lam"]
    r = u[:, None, :] - m[None]
    out = log_softmax(-0.5 * (r ** 2 / (1 + lam)).sum(-1) / T, axis=1)
    rows = np.asarray(rows, dtype=bool)
    groups = np.asarray(groups)
    for grp in np.unique(groups[rows]):
        idx = np.flatnonzero(rows & (groups == grp))
        if len(idx) < 2:
            continue
        gam = lam / (1 + len(idx) * lam)
        if len(idx) <= n_exact:
            out[idx] = _exact(r[idx], gam, T)
        else:
            out[idx] = _mean_field(r[idx], gam, T, iters)
    return out
