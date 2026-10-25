import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from nthl.labels import LABELS
from nthl.qwen27b import (
    END_ID,
    LoraLinear,
    add_end,
    find_assets,
    heldout_rows,
    pooled_logits,
    to_our_order,
)


def test_to_our_order_moves_columns_by_name():
    theirs = sorted(LABELS)
    p = np.tile(np.arange(18, dtype=float), (2, 1))
    out = to_our_order(p, theirs)
    for j, label in enumerate(LABELS):
        assert out[0, j] == theirs.index(label)


def test_to_our_order_rejects_other_labels():
    with pytest.raises(AssertionError):
        to_our_order(np.zeros((1, 18)), [*sorted(LABELS)[:-1], "not-a-model"])


def test_add_end_appends_the_class_token_and_keeps_every_text_token():
    assert add_end([[5, 6, 7], []]) == [[5, 6, 7, END_ID], [END_ID]]


def test_pooled_logits_is_masked_mean_then_norm_then_concat_then_head():
    torch.manual_seed(0)
    hidden = [torch.randn(2, 4, 3), torch.randn(2, 4, 3)]
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    norms = nn.ModuleList([nn.LayerNorm(3), nn.LayerNorm(3)])
    head = nn.Linear(6, 18)
    got = pooled_logits(hidden, mask, norms, head)
    want = []
    for i in range(2):
        n = int(mask[i].sum())
        z = torch.cat([norms[k](hidden[k][i, :n].mean(0)) for k in range(2)])
        want.append(head(z))
    assert torch.allclose(got, torch.stack(want), atol=1e-5)


def test_pooled_logits_ignores_padding_values():
    torch.manual_seed(1)
    hidden = torch.randn(1, 4, 3)
    noisy = hidden.clone()
    noisy[:, 2:] = 1e4
    mask = torch.tensor([[1, 1, 0, 0]])
    norms, head = nn.ModuleList([nn.LayerNorm(3)]), nn.Linear(3, 18)
    assert torch.allclose(pooled_logits([hidden], mask, norms, head), pooled_logits([noisy], mask, norms, head))


def test_lora_linear_adds_scaled_low_rank_update():
    torch.manual_seed(2)
    w = torch.randn(5, 4)
    lin = LoraLinear(lambda: w, 4, 5, rank=2, scale=2.0, device="cpu")
    with torch.no_grad():
        lin.lora_A.copy_(torch.randn(2, 4))
        lin.lora_B.copy_(torch.randn(5, 2))
    x = torch.randn(3, 4)
    want = x @ w.T + 2.0 * (x @ lin.lora_A.T) @ lin.lora_B.T
    assert torch.allclose(lin(x), want, atol=1e-5)


def test_lora_linear_without_rank_is_plain():
    w = torch.randn(5, 4)
    lin = LoraLinear(lambda: w, 4, 5, rank=0, scale=2.0, device="cpu")
    assert lin.lora_A is None
    x = torch.randn(3, 4)
    assert torch.allclose(lin(x), x @ w.T)


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_find_assets_locates_base_adapters_and_wheels(tmp_path):
    root = tmp_path / "bundle" / "name-that-llm-artifacts"
    write(root / "qwen38-27b-nf4" / "config.json", "{}")
    write(root / "qwen38-27b-lora-tail8" / "final.safetensors")
    own = tmp_path / "bundle" / "ntl627-own-fulltext" / "ntl627-own-fulltext"
    for name in ["final", "outer0"]:
        write(own / f"{name}.safetensors")
        write(own / f"{name}.json", "{}")
    for wheel in ["accelerate-1.whl", "bitsandbytes-0.whl"]:
        write(root / "qwen38-offline-linux" / wheel)
    write(root / "ntl-oof-q35-heads" / "fold_plan.json", "{}")
    found = find_assets(tmp_path)
    assert found["base"] == str(root / "qwen38-27b-nf4")
    assert found["final"] == str(own / "final.safetensors")
    assert found["outer0"] == str(own / "outer0.safetensors")
    assert sorted(found["wheels"]) == sorted(str(p) for p in (root / "qwen38-offline-linux").glob("*.whl"))
    assert found["fold_plan"] == str(root / "ntl-oof-q35-heads" / "fold_plan.json")


def test_find_assets_fails_loudly_when_the_base_is_missing(tmp_path):
    with pytest.raises(AssertionError):
        find_assets(tmp_path)


def test_heldout_rows_checks_the_train_file_and_prompt_disjointness(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("id,prompt_id\n")
    sha = hashlib.sha256(train.read_bytes()).hexdigest()
    plan = tmp_path / "fold_plan.json"
    plan.write_text(json.dumps({"train_sha256": sha, "jobs": {"outer0": {"fit": [0, 1], "valid": [2, 3]}}}))
    prompts = np.array(["a", "a", "b", "b"])
    assert heldout_rows(plan, train, prompts).tolist() == [2, 3]
    with pytest.raises(AssertionError):
        heldout_rows(plan, train, np.array(["a", "b", "b", "c"]))
    train.write_text("id,prompt_id,changed\n")
    with pytest.raises(AssertionError):
        heldout_rows(plan, train, prompts)
