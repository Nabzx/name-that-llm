"""Train our own LoRA24 classifier on Kaggle (#43): the published two-stage recipe, re-implemented.

A truncated decoder (first `keep` layers, final norm off) with LoRA on the
chosen projections, mean pools at several depths without the first token,
one LayerNorm per depth and a linear head. Stage one reads the public pool,
stage two the host train. Only the LoRA, norms and head are trained.
"""

import json
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from nthl.labels import LABELS


class LoRALinear(nn.Module):
    """base(x) + (alpha / r) * B A x, with A and B kept in fp32."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(self.drop(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return y + (z * self.scale).to(y.dtype)


def add_lora(model, targets, r, alpha, dropout):
    """Freeze the model and wrap every Linear whose name ends in a target. Returns how many."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = 0
    for name, module in list(model.named_modules()):
        for child, sub in list(module.named_children()):
            if child in targets and isinstance(sub, nn.Linear):
                setattr(module, child, LoRALinear(sub, r, alpha, dropout))
                wrapped += 1
    return wrapped


def lora_state(model):
    """LoRA weights keyed `layers.N....lora_A`, the names our merge expects."""
    out = {}
    for k, v in model.named_parameters():
        if v.requires_grad and (k.endswith("lora_A") or k.endswith("lora_B")):
            out[k[k.index("layers."):].replace(".base", "")] = v.detach().float().cpu().contiguous()
    return out


def save_checkpoint(out_dir, model, norms, head, keep, layers, max_len, eos_id, r, alpha):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    state = lora_state(model)
    for i, nm in enumerate(norms):
        state[f"norms.{i}.weight"] = nm.weight.detach().float().cpu().contiguous()
        state[f"norms.{i}.bias"] = nm.bias.detach().float().cpu().contiguous()
    state["fc.weight"] = head.weight.detach().float().cpu().contiguous()
    state["fc.bias"] = head.bias.detach().float().cpu().contiguous()
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    meta = {"labels": list(LABELS), "keep": keep, "layers": list(layers), "max_len": max_len,
            "eos_id": eos_id, "r": r, "alpha": alpha}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
