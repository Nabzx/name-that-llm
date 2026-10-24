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


def test_merge_accepts_prvsiyan_style_keys():
    torch.manual_seed(0)
    model = Trunk()
    before = model.layers[0].self_attn.q_proj.weight.detach().clone()
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    state = {"backbone.layers.0.self_attn.q_proj.lora_A": a, "backbone.layers.0.self_attn.q_proj.lora_B": b,
             "fc.weight": torch.zeros(18, 4)}
    assert merge_lora(model, state, scale=2.0) == 1
    assert torch.allclose(model.layers[0].self_attn.q_proj.weight, before + 2.0 * b @ a, atol=1e-6)


def test_prvsiyan_reader_maps_heads_norms_and_framing(tmp_path):
    import json

    from safetensors.torch import save_file

    from nthl.lora24 import read_prvsiyan

    labels = sorted(LABELS)
    meta = {"labels": labels, "pool_layers": [2, 4], "n_layers": 4,
            "hp": {"r": 16, "alpha": 32, "max_len": 384, "bos_id": 151645}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    state = {"head_norm.norms.0.weight": torch.ones(3), "head_norm.norms.0.bias": torch.zeros(3),
             "head_norm.norms.1.weight": torch.ones(3), "head_norm.norms.1.bias": torch.zeros(3),
             "fc.weight": torch.zeros(18, 6), "fc.bias": torch.zeros(18),
             "fc_c.weight": torch.zeros(18, 6), "fc_c.bias": torch.zeros(18),
             "backbone.layers.0.self_attn.q_proj.lora_A": torch.zeros(16, 3)}
    save_file(state, str(tmp_path / "adapter.safetensors"))
    spec = read_prvsiyan(str(tmp_path))
    assert spec["labels"] == labels and spec["keep"] == 4 and spec["layers"] == [2, 4]
    assert spec["prefix_id"] == 151645 and spec["suffix_id"] is None and spec["max_len"] == 383
    assert spec["scale"] == 2.0 and len(spec["norms"]) == 2
    assert spec["plain"][0].shape == (18, 6) and spec["centred"] is not None
