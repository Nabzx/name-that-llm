import math

import torch

from nthl.witness import FEATURES, token_stats, tokens_per_word


def test_a_witness_that_predicts_every_token_scores_near_zero_surprise():
    # 1 passage, 4 tokens, vocab of 5. The logits at position t point at token t+1.
    ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, 5), -20.0)
    for t in range(3):
        logits[0, t, ids[0, t + 1]] = 20.0
    mask = torch.ones_like(ids)
    f = token_stats(logits, ids, mask)
    assert f.shape == (1, len(FEATURES))
    row = dict(zip(FEATURES, f[0].tolist()))
    assert row["logp_mean"] > -1e-3
    assert row["top1_share"] == 1.0
    assert row["log_rank_mean"] == 0.0
    assert row["entropy_mean"] < 1e-3


def test_padding_is_ignored():
    ids = torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]])
    logits = torch.zeros((2, 4, 5))  # uniform: every logp is -log 5
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    f = token_stats(logits, ids, mask)
    row = dict(zip(FEATURES, f[1].tolist()))
    assert abs(row["logp_mean"] + math.log(5)) < 1e-5
    assert abs(row["entropy_mean"] - math.log(5)) < 1e-5


def test_tokens_per_word():
    class Tok:
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [list(range(len(t))) for t in texts]}  # one token per character

    assert tokens_per_word(Tok(), ["ab cd", "abc"]) == [2.5, 3.0]
