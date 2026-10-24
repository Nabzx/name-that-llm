"""Run a published LoRA24 checkpoint: passages in, 18 probabilities out (#21, #27).

The recipe, from the checkpoint's own meta.json: keep the first `keep`
decoder layers, merge the LoRA adapter into the weights, take a masked mean
of the hidden states at 8 depths without the first token, put each through
its own LayerNorm, concatenate, and apply a linear head. Columns come back in
our LABELS order, whatever order the checkpoint used.

The torch and transformers parts only run on Kaggle. The small parts are
tested locally on tiny tensors.
"""

import glob
import json
import os
import time

import numpy as np
import torch
from torch import nn

from nthl.labels import LABELS


def to_our_order(probs, ckpt_labels):
    """Columns from the checkpoint's label order to ours."""
    return np.asarray(probs)[:, [list(ckpt_labels).index(label) for label in LABELS]]


def check_meta(meta):
    """Assert the checkpoint knows our 18 labels and its training end token. Returns the end token."""
    assert sorted(meta["labels"]) == sorted(LABELS), "checkpoint labels differ from ours"
    cfg = meta["cfg"]
    assert "eos_id" in cfg, "checkpoint does not record its training end token"
    assert cfg["layers"][-1] <= cfg["keep"], "pooled depth beyond the kept layers"
    return int(cfg["eos_id"])


@torch.no_grad()
def merge_lora(model, state, scale):
    """W becomes W + B A scale for every adapted linear layer. Returns how many were merged."""
    modules = dict(model.named_modules())
    merged = 0
    for key, a in state.items():
        if ".lora_A." not in key:
            continue
        name = key.split(".lora_A.")[0]
        name = name[name.index("layers."):]
        assert name in modules, f"adapter targets a layer the trunk does not have: {name}"
        lin = modules[name]
        b = state[key.replace(".lora_A.", ".lora_B.")]
        delta = (b.float().to(lin.weight.device) @ a.float().to(lin.weight.device)) * scale
        lin.weight.add_(delta.to(lin.weight.dtype))
        merged += 1
    return merged


def pool_layers(hidden_states, mask, layers):
    """Masked mean over tokens at each depth, leaving out the first token, which soaks up attention."""
    m = mask.unsqueeze(-1).float().clone()
    m[:, 0] = 0
    den = m.sum(1).clamp(min=1)
    out = []
    for layer in layers:
        h = hidden_states[layer].float()
        out.append((h * m.to(h.device)).sum(1) / den.to(h.device))
    return out


def find_base(root, spec):
    """The attached base model whose config.json matches every key in spec."""
    for path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
        try:
            config = json.load(open(path))
        except (OSError, ValueError):
            continue
        if all(config.get(k) == v for k, v in spec.items()):
            return os.path.dirname(path)
    raise FileNotFoundError(f"no attached base model matches {spec}")


def load_trunk(base_dir, keep, dtype=torch.float16):
    """The first `keep` decoder layers, split over the GPUs when one is not enough, final norm off."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(base_dir)
    config.num_hidden_layers = keep
    config.use_cache = False
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:keep]
    memory = {i: "13GiB" for i in range(torch.cuda.device_count())}
    model = AutoModel.from_pretrained(base_dir, config=config, dtype=dtype, device_map="auto", max_memory=memory).eval()
    model.norm = nn.Identity()
    return model


@torch.no_grad()
def run_checkpoint(ckpt_dir, base_dir, texts, batch_size=8):
    """Probabilities [n, 18] in our LABELS order, and a dict of what happened."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    start = time.time()
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    eos_id = check_meta(meta)
    cfg = meta["cfg"]

    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]
    ids = [x + [eos_id] for x in ids]

    model = load_trunk(base_dir, cfg["keep"])
    loaded = time.time() - start
    adapter = load_file(os.path.join(ckpt_dir, "adapter", "adapter_model.safetensors"))
    merged = merge_lora(model, adapter, scale=cfg["alpha"] / cfg["r"])
    heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
    hidden = model.config.hidden_size
    first = model.embed_tokens.weight.device
    norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in cfg["layers"]]).float().to(first)
    norms.load_state_dict({k[len("norms."):]: v for k, v in heads.items() if k.startswith("norms.")})

    z = np.zeros((len(ids), hidden * len(cfg["layers"])), np.float32)
    redone_bf16 = redone_single = 0

    def embed(idx):
        nonlocal redone_bf16
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(first)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in cfg["layers"]):
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
            model.to(torch.float16)
            redone_bf16 += 1
        pooled = pool_layers(hs, enc["attention_mask"], cfg["layers"])
        z[idx] = torch.cat([nm(p.to(first)) for nm, p in zip(norms, pooled)], -1).cpu().numpy()

    order = np.argsort([len(x) for x in ids], kind="stable")
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        try:
            embed(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for i in idx:
                embed([i])
            redone_single += 1

    w, b = heads["plain.weight"].float().numpy(), heads["plain.bias"].float().numpy()
    logits = z @ w.T + b
    logits -= logits.max(1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    info = {
        "eos_used": eos_id,
        "eos_tokenizer": tok.eos_token_id,
        "keep": cfg["keep"],
        "layers": cfg["layers"],
        "merged_linear_layers": merged,
        "redone_bf16_batches": redone_bf16,
        "redone_single_batches": redone_single,
        "load_seconds": round(loaded),
        "total_seconds": round(time.time() - start),
    }
    del model
    torch.cuda.empty_cache()
    return to_our_order(probs, meta["labels"]), info
