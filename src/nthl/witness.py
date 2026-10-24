"""Lab witnesses (#34): how likely an open model from each lab finds a passage.

For every passage and witness, per-token statistics of the witness's own
predictions: how surprised it is by each actual token (log probability),
how unsure it was (entropy), where the actual token ranked, and the
Fast-DetectGPT discrepancy (how far the actual token's log probability sits
from what the witness expected, in standard deviations). Padding and the
first token, which has nothing to predict it, are left out.
"""

import numpy as np
import torch

FEATURES = [
    "logp_mean", "logp_std", "logp_p10", "logp_p50", "logp_p90",
    "entropy_mean", "log_rank_mean", "top1_share", "discrepancy", "n_tokens",
]


@torch.no_grad()
def token_stats(logits, ids, mask):
    """Features [batch, len(FEATURES)] from causal LM logits [batch, tokens, vocab]."""
    logp_all = torch.log_softmax(logits[:, :-1].float(), -1)
    target = ids[:, 1:]
    valid = mask[:, 1:].bool()
    logp = logp_all.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p_all = logp_all.exp()
    entropy = -(p_all * logp_all).sum(-1)
    mu = -entropy
    var = (p_all * logp_all ** 2).sum(-1) - mu ** 2
    rank = (logp_all > logp.unsqueeze(-1)).sum(-1).float()
    out = []
    for b in range(ids.shape[0]):
        v = valid[b]
        lp = logp[b][v].cpu().numpy()
        disc = ((logp[b][v] - mu[b][v]).sum() / var[b][v].sum().clamp(min=1e-6).sqrt()).item()
        out.append([
            lp.mean(), lp.std(), *np.percentile(lp, [10, 50, 90]),
            entropy[b][v].mean().item(), torch.log1p(rank[b][v]).mean().item(),
            (rank[b][v] == 0).float().mean().item(), disc, float(v.sum().item()),
        ])
    return np.asarray(out, dtype=np.float32)


def tokens_per_word(tokenizer, texts):
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [len(i) / max(len(t.split()), 1) for i, t in zip(ids, texts)]
