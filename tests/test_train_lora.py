import json

import torch
from torch import nn

from nthl.labels import LABELS
from nthl.train_lora import LoRALinear, add_lora, save_checkpoint


def test_lora_linear_is_base_plus_scaled_b_a():
    torch.manual_seed(0)
    base = nn.Linear(4, 3, bias=False)
    lora = LoRALinear(base, r=2, alpha=4, dropout=0.0)
    nn.init.normal_(lora.lora_B)
    x = torch.randn(5, 4)
    expected = base(x) + 2.0 * (x @ lora.lora_A.T) @ lora.lora_B.T
    assert torch.allclose(lora(x), expected, atol=1e-5)


def test_add_lora_wraps_only_the_targets_and_freezes_the_rest():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = nn.Module()
    model.layers[0].self_attn.q_proj = nn.Linear(4, 4)
    model.layers[0].self_attn.other = nn.Linear(4, 4)
    n = add_lora(model, ["q_proj"], r=2, alpha=4, dropout=0.0)
    assert n == 1
    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    trainable = [k for k, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["layers.0.self_attn.q_proj.lora_A", "layers.0.self_attn.q_proj.lora_B"]


def test_saved_checkpoint_loads_through_our_reader(tmp_path):
    from nthl.lora24 import read_ours

    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.up_proj = nn.Linear(4, 4)
    add_lora(model, ["up_proj"], r=2, alpha=4, dropout=0.0)
    norms = nn.ModuleList([nn.LayerNorm(4), nn.LayerNorm(4)])
    head = nn.Linear(8, 18)
    save_checkpoint(str(tmp_path), model, norms, head, keep=24, layers=[2, 24], max_len=320, eos_id=151643, r=2, alpha=4)
    spec = read_ours(str(tmp_path))
    assert spec["labels"] == list(LABELS) and spec["keep"] == 24 and spec["layers"] == [2, 24]
    assert spec["suffix_id"] == 151643 and spec["prefix_id"] is None and spec["scale"] == 2.0
    assert "layers.0.mlp.up_proj.lora_A" in spec["adapter"]
    assert spec["plain"][0].shape == (18, 8) and len(spec["norms"]) == 2 and spec["centred"] is None
    assert json.load(open(tmp_path / "meta.json"))["labels"] == list(LABELS)
