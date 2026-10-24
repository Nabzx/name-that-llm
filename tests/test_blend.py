import numpy as np

from nthl.blend import centred_probs, sinkhorn, topic_groups


def probs(n=60, k=18, seed=0):
    p = np.random.default_rng(seed).random((n, k)) ** 3
    p[:, 0] *= 5  # a lean towards class 0
    return p / p.sum(1, keepdims=True)


def test_full_sinkhorn_gives_every_class_the_same_mass():
    out = sinkhorn(probs(), alpha=1.0)
    assert np.allclose(out.sum(1), 1)
    assert np.allclose(out.sum(0), len(out) / 18, rtol=1e-3)


def test_zero_strength_changes_nothing():
    p = probs()
    assert np.allclose(sinkhorn(p, alpha=0.0), p)


def test_half_strength_lands_halfway():
    p = probs()
    out = sinkhorn(p, alpha=0.5)
    target = 0.5 * p.sum(0) + 0.5 * len(p) / 18
    assert np.allclose(out.sum(0), target, rtol=1e-3)


def test_topic_groups_put_same_topic_passages_together():
    texts = [
        "Volcanoes erupt when magma pressure in the crust builds beneath the volcano",
        "Magma pressure builds beneath a volcano until the crust gives way and it erupts",
        "Photosynthesis lets plants turn sunlight water and carbon dioxide into sugar",
        "Plants use photosynthesis to turn sunlight and carbon dioxide into sugar",
        "The stock market rose sharply after the central bank cut interest rates",
    ]
    groups, twin = topic_groups(texts, cut=0.85, lo=2, hi=18)
    assert groups[0] == groups[1] and groups[2] == groups[3]
    assert groups[0] != groups[2]
    assert twin.tolist() == [True, True, True, True, False]


def test_centred_probs_subtract_the_group_mean():
    z = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 5.0]])
    groups = np.array([1, 1, 2])
    twin = np.array([True, True, False])
    w = np.array([[1.0, 0.0], [-1.0, 0.0]])  # two classes, reads the first feature only
    b = np.zeros(2)
    out = centred_probs(z, groups, twin, w, b)
    assert out.shape == (2, 2)
    assert out[0].argmax() == 1 and out[1].argmax() == 0  # below and above the group mean


def test_peer_correction_leaves_small_clusters_and_singletons_alone():
    from nthl.blend import peer_correct

    z = np.zeros((4, 18))
    svc = np.random.default_rng(0).normal(size=(4, 18))
    groups = np.array([1, 1, 2, 3])          # a pair and two singletons
    sim = np.ones(4)
    assert np.allclose(peer_correct(z, svc, groups, sim, w_svc=1.0), z)


def test_peer_correction_subtracts_the_others_shrunk_mean_not_its_own():
    from nthl.blend import peer_correct

    svc = np.zeros((3, 18))
    svc[1, 5] = 3.0                          # only row 1 leans to class 5
    groups = np.array([7, 7, 7])
    sim = np.full(3, 0.5)                    # (0.5 - 0.1) / 0.4 = 1, full topic strength
    out = peer_correct(np.zeros((3, 18)), svc, groups, sim, w_svc=1.0, strength=0.5, prior=8)
    margin = svc - svc.mean(1, keepdims=True)
    r = (3 - 1) / (3 - 1 + 8)
    expected_row0 = -0.5 * r * (margin[1] + margin[2]) / 2
    assert np.allclose(out[0], expected_row0)
    assert np.allclose(out[1], -0.5 * r * (margin[0] + margin[2]) / 2)   # row 1 never removes its own lean


def test_cluster_similarity_is_the_mean_off_diagonal_cosine():
    from scipy.sparse import csr_matrix

    from nthl.blend import cluster_similarity

    x = csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    sim = cluster_similarity(x, np.array([1, 1, 1]))
    assert np.allclose(sim, (1 + 0 + 0) / 3)
