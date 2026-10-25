import itertools

import numpy as np
from scipy.stats import multivariate_normal

from nthl.gauss import fit, log_posteriors, transform

K = 18


def make(rng, prompts, d=12, prompt_scale=3.0, noise=1.0, mu=None):
    mu = rng.normal(size=(K, d)) if mu is None else mu
    x, y, g = [], [], []
    for p in range(prompts):
        shift = rng.normal(scale=prompt_scale, size=d)
        for c in range(K):
            x.append(mu[c] + shift + rng.normal(scale=noise, size=d))
            y.append(c)
            g.append(p)
    return np.array(x), np.array(y), np.array(g), mu


def test_exact_pair_matches_a_brute_force_gaussian():
    rng = np.random.default_rng(0)
    x, y, g, mu = make(rng, 40)
    prm = fit(x, y, g, d=8)
    u = transform(x[:2], prm)
    got = log_posteriors(u, prm, groups=np.array([0, 0]), rows=np.array([True, True]))
    lam, m = prm["lam"], prm["m"]
    cov = np.block([[np.eye(len(lam)) + np.diag(lam), np.diag(lam)], [np.diag(lam), np.eye(len(lam)) + np.diag(lam)]])
    table = np.full((K, K), -np.inf)
    for a, b in itertools.product(range(K), range(K)):
        if a != b:   # one model answers a prompt at most once
            table[a, b] = multivariate_normal.logpdf(np.r_[u[0], u[1]], np.r_[m[a], m[b]], cov)
    table -= np.logaddexp.reduce(table.ravel())
    want = np.stack([np.logaddexp.reduce(table, axis=1), np.logaddexp.reduce(table, axis=0)])
    assert np.allclose(got, want, atol=1e-6)


def test_groups_beat_single_rows_when_the_prompt_effect_is_large():
    rng = np.random.default_rng(1)
    x, y, g, mu = make(rng, 60)
    prm = fit(x, y, g, d=12)
    xt, yt, gt, _ = make(rng, 30, mu=mu)
    keep = np.concatenate([np.flatnonzero(gt == p)[rng.choice(K, 3, replace=False)] for p in range(30)])
    u, yk, gk = transform(xt[keep], prm), yt[keep], gt[keep]
    single = log_posteriors(u, prm, groups=gk, rows=np.zeros(len(keep), bool)).argmax(1)
    joint = log_posteriors(u, prm, groups=gk, rows=np.ones(len(keep), bool)).argmax(1)
    assert (joint == yk).mean() > (single == yk).mean() + 0.05


def test_large_groups_use_the_approximation_and_stay_normalised():
    rng = np.random.default_rng(2)
    x, y, g, mu = make(rng, 40)
    prm = fit(x, y, g, d=8)
    u = transform(x[:6], prm)
    out = log_posteriors(u, prm, groups=np.zeros(6, int), rows=np.ones(6, bool))
    assert out.shape == (6, K) and np.allclose(np.exp(out).sum(1), 1)
