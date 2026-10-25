"""Run a published LoRA24 checkpoint: passages in, 18 probabilities out (#21, #27).

The recipe, from the checkpoint's own meta.json: keep the first `keep`
decoder layers, merge the LoRA adapter into the weights, take a masked mean
of the hidden states at 8 depths without the first token, put each through
its own LayerNorm, concatenate, and apply a linear head. Columns come back in
our LABELS order, whatever order the checkpoint used.

The torch and transformers parts only run on Kaggle. The small parts are
tested locally on tiny tensors.
"""

import gc
import glob
import json
import os
import re
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
    """W becomes W + B A scale for every adapted linear layer. Returns how many were merged.

    Accepts PEFT keys (`...q_proj.lora_A.weight`) and plain ones (`...q_proj.lora_A`); other keys are ignored.
    """
    modules = dict(model.named_modules())
    merged = 0
    for key, a in state.items():
        found = re.match(r"^(.*)\.lora_A(\.weight)?$", key)
        if not found:
            continue
        name = found.group(1)
        name = name[name.index("layers."):]
        assert name in modules, f"adapter targets a layer the trunk does not have: {name}"
        lin = modules[name]
        b = state[key.replace(".lora_A", ".lora_B")]
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


def read_nishant(ckpt_dir):
    """Spec for Nishant Dahal's layout: meta.json, adapter/adapter_model.safetensors, heads.safetensors."""
    from safetensors.torch import load_file

    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    eos_id = check_meta(meta)
    cfg = meta["cfg"]
    heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
    return {
        "labels": meta["labels"], "keep": cfg["keep"], "layers": cfg["layers"], "max_len": cfg["max_len"],
        "prefix_id": None, "suffix_id": eos_id, "scale": cfg["alpha"] / cfg["r"],
        "adapter": load_file(os.path.join(ckpt_dir, "adapter", "adapter_model.safetensors")),
        "norms": [(heads[f"norms.{i}.weight"], heads[f"norms.{i}.bias"]) for i in range(len(cfg["layers"]))],
        "plain": (heads["plain.weight"].float().numpy(), heads["plain.bias"].float().numpy()),
        "centred": (heads["centered.weight"].float().numpy(), heads["centered.bias"].float().numpy()) if "centered.weight" in heads else None,
    }


def read_prvsiyan(adapter_dir):
    """Spec for prvsiyan's layout: meta.json and one adapter.safetensors holding LoRA, norms and heads.

    A dummy start token goes in front and is never pooled. The cut includes it.
    """
    from safetensors.torch import load_file

    meta = json.load(open(os.path.join(adapter_dir, "meta.json")))
    assert sorted(meta["labels"]) == sorted(LABELS), "checkpoint labels differ from ours"
    hp = meta["hp"]
    state = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
    layers = meta["pool_layers"]
    return {
        "labels": meta["labels"], "keep": meta["n_layers"], "layers": layers, "max_len": hp["max_len"] - 1,
        "prefix_id": int(hp["bos_id"]), "suffix_id": None, "scale": hp["alpha"] / hp["r"], "adapter": state,
        "norms": [(state[f"head_norm.norms.{i}.weight"], state[f"head_norm.norms.{i}.bias"]) for i in range(len(layers))],
        "plain": (state["fc.weight"].float().numpy(), state["fc.bias"].float().numpy()),
        "centred": (state["fc_c.weight"].float().numpy(), state["fc_c.bias"].float().numpy()) if "fc_c.weight" in state else None,
    }


def read_zbigniew(ckpt_dir):
    """Spec for Zbigniew Kirchner's layout, the `final/` folder: adapter/ (PEFT), heads.safetensors
    (LayerNorms only), readouts.npz (labels, pooled depths, plain and centred heads) and tokenizer/.

    There is no meta.json. The end token is the shipped tokenizer's own (`<|im_end|>`), the trunk keeps
    as many layers as the deepest pooled one, and the cut of 447 tokens is fixed in his notebook.
    """
    from safetensors.torch import load_file

    ro = np.load(os.path.join(ckpt_dir, "readouts.npz"))
    labels = [str(label) for label in ro["classes"]]
    assert sorted(labels) == sorted(LABELS), "checkpoint labels differ from ours"
    layers = [int(layer) for layer in ro["pool_layers"]]
    lora = json.load(open(os.path.join(ckpt_dir, "adapter", "adapter_config.json")))
    assert not lora.get("use_rslora"), "rank-stabilised LoRA scales by alpha / sqrt(r)"
    eos = json.load(open(os.path.join(ckpt_dir, "tokenizer", "tokenizer_config.json")))["eos_token"]
    added = json.load(open(os.path.join(ckpt_dir, "tokenizer", "tokenizer.json")))["added_tokens"]
    heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
    return {
        "labels": labels, "keep": max(layers), "layers": layers, "max_len": 447,
        "prefix_id": None, "suffix_id": next(t["id"] for t in added if t["content"] == eos),
        "scale": lora["lora_alpha"] / lora["r"],
        "adapter": load_file(os.path.join(ckpt_dir, "adapter", "adapter_model.safetensors")),
        "norms": [(heads[f"norms.{i}.weight"], heads[f"norms.{i}.bias"]) for i in range(len(layers))],
        "plain": (ro["plain_coef"], ro["plain_intercept"]),
        "centred": (ro["center_coef"], ro["center_intercept"]) if "center_coef" in ro.files else None,
    }


def read_ours(ckpt_dir):
    """Spec for checkpoints trained by our own nthl.train_lora: meta.json and model.safetensors."""
    from safetensors.torch import load_file

    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    assert meta["labels"] == list(LABELS)
    state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    n = len(meta["layers"])
    return {
        "labels": meta["labels"], "keep": meta["keep"], "layers": meta["layers"], "max_len": meta["max_len"],
        "prefix_id": None, "suffix_id": meta["eos_id"], "scale": meta["alpha"] / meta["r"], "adapter": state,
        "norms": [(state[f"norms.{i}.weight"], state[f"norms.{i}.bias"]) for i in range(n)],
        "plain": (state["fc.weight"].float().numpy(), state["fc.bias"].float().numpy()),
        "centred": None,
    }


def run_checkpoint(ckpt_dir, base_dir, texts, batch_size=8, groups=None, twin=None):
    """Nishant's layout. See run_spec."""
    return run_spec(read_nishant(ckpt_dir), base_dir, texts, batch_size, groups, twin)


@torch.no_grad()
def run_spec(spec, base_dir, texts, batch_size=8, groups=None, twin=None):
    """Probabilities [n, 18] in our LABELS order, centred-twin probabilities for the `twin` rows
    (or None), and a dict of what happened."""
    from transformers import AutoTokenizer

    start = time.time()
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=spec["max_len"])["input_ids"]
    ids = [([spec["prefix_id"]] if spec["prefix_id"] is not None else []) + x
           + ([spec["suffix_id"]] if spec["suffix_id"] is not None else []) for x in ids]

    model = load_trunk(base_dir, spec["keep"])
    loaded = time.time() - start
    merged = merge_lora(model, spec["adapter"], scale=spec["scale"])
    hidden = model.config.hidden_size
    first = model.embed_tokens.weight.device
    norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in spec["layers"]]).float().to(first)
    for nm, (w, b) in zip(norms, spec["norms"]):
        nm.weight.copy_(w.float())
        nm.bias.copy_(b.float())

    z = np.zeros((len(ids), hidden * len(spec["layers"])), np.float32)
    redone_bf16 = redone_single = 0

    def embed(idx):
        nonlocal redone_bf16
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(first)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in spec["layers"]):
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
            model.to(torch.float16)
            redone_bf16 += 1
        pooled = pool_layers(hs, enc["attention_mask"], spec["layers"])
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

    centred = None
    if twin is not None and twin.any() and spec["centred"] is not None:
        from nthl.blend import centred_probs

        centred = to_our_order(centred_probs(z, groups, twin, *spec["centred"]), spec["labels"])
    w, b = spec["plain"]
    logits = z @ w.T + b
    logits -= logits.max(1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    info = {
        "prefix_id": spec["prefix_id"], "suffix_id": spec["suffix_id"], "eos_tokenizer": tok.eos_token_id,
        "keep": spec["keep"], "layers": spec["layers"], "merged_linear_layers": merged,
        "redone_bf16_batches": redone_bf16, "redone_single_batches": redone_single,
        "load_seconds": round(loaded), "total_seconds": round(time.time() - start),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return to_our_order(probs, spec["labels"]), centred, info
