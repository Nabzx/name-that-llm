import numpy as np
import pytest
import torch
from torch import nn

from nthl.labels import LABELS
from nthl.lora24 import check_meta, merge_lora, pool_layers, to_our_order

CKPT_ORDER = [l for l in LABELS if l != "claude-fable-5"]
CKPT_ORDER.insert(CKPT_ORDER.index("claude-sonnet-5") + 1, "claude-fable-5")


def test_columns_come_back_in_our_order():
    probs = np.eye(18)[[CKPT_ORDER.index("claude-fable-5")]]
    out = to_our_order(probs, CKPT_ORDER)
    assert out.shape == (1, 18)
    assert LABELS[out[0].argmax()] == "claude-fable-5"


def test_meta_must_hold_our_18_labels_and_an_end_token():
    meta = {"labels": CKPT_ORDER, "cfg": {"eos_id": 151643, "keep": 27, "layers": [2, 27], "max_len": 447}}
    assert check_meta(meta) == 151643
    with pytest.raises(AssertionError):
        check_meta({"labels": CKPT_ORDER[:-1], "cfg": meta["cfg"]})
    with pytest.raises(AssertionError):
        check_meta({"labels": CKPT_ORDER, "cfg": {"keep": 27, "layers": [2], "max_len": 447}})


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 3, bias=False)


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])


def test_merge_adds_b_times_a_times_the_scale():
    torch.manual_seed(0)
    model = Trunk()
    before = model.layers[1].self_attn.q_proj.weight.detach().clone()
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    state = {
        "base_model.model.layers.1.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.layers.1.self_attn.q_proj.lora_B.weight": b,
    }
    assert merge_lora(model, state, scale=2.0) == 1
    after = model.layers[1].self_attn.q_proj.weight
    assert torch.allclose(after, before + 2.0 * b @ a, atol=1e-6)


def test_pool_skips_the_first_token_and_the_padding():
    # one passage, three real tokens then one pad; hidden size 2
    h = torch.tensor([[[100.0, 100.0], [1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    hidden_states = [h * 0, h]
    mask = torch.tensor([[1, 1, 1, 0]])
    pooled = pool_layers(hidden_states, mask, layers=[1])
    assert len(pooled) == 1
    assert torch.allclose(pooled[0], torch.tensor([[2.0, 3.0]]))
