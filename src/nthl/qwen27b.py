"""Run lightningv08's full-text Qwen3.8-27B NF4 classifier: passages in, 18 probabilities out (#28, #46).

The assets are public, in lightningv08/ntl-olmo2-native24-final-inference-assets:
the NF4 base under qwen38-27b-nf4 (Apache 2.0, greghavens/Qwen3.8-27B-bnb-4bit),
the task adapters under ntl627-own-fulltext (`final` saw all of train, `outer0`
held out 539 prompt-disjoint official rows) and offline wheels for
transformers 5.14.1 and bitsandbytes 0.50.2. Their 0.642 notebook was read,
not copied. This is our own loader for the same weight format.

The recipe: no truncation, no special tokens, the end token 248044 appended
to every passage. All 64 decoder layers run. Every linear layer keeps its
NF4 weight and is dequantised on the fly. Layers 56 to 63 add a rank-16 LoRA
update at scale 2. The masked mean of the hidden states after layers 8, 16,
..., 64 (no final norm) goes through its own LayerNorm; the eight are
concatenated and the `plain` head gives 18 logits in the adapter's label order.

The torch, bitsandbytes and transformers parts run only on Kaggle, after
`install_wheels` puts the bundled wheels first on sys.path. The small parts
are tested locally on tiny tensors.
"""

import gc
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nthl.labels import LABELS

END_ID = 248044
DEPTHS = [8, 16, 24, 32, 40, 48, 56, 64]
LORA_LAYERS = range(56, 64)
RANK, SCALE = 16, 2.0
PREFIX = "model.language_model."


def to_our_order(probs, ckpt_labels):
    """Columns from the adapter's label order to ours."""
    assert sorted(ckpt_labels) == sorted(LABELS), "adapter labels differ from ours"
    return np.asarray(probs)[:, [list(ckpt_labels).index(label) for label in LABELS]]


def add_end(token_lists):
    """Every passage keeps all its tokens and ends with the class token, as in training."""
    return [list(ids) + [END_ID] for ids in token_lists]


def pooled_logits(hidden, mask, norms, head):
    """Masked mean of each depth's hidden states, its LayerNorm, concatenated, then the head. In float32."""
    m = mask.unsqueeze(-1).float()
    den = m.sum(1)
    z = [norm((h.float() * m.to(h.device)).sum(1).to(den.device) / den) for h, norm in zip(hidden, norms)]
    return head(torch.cat(z, dim=-1))


class LoraLinear(nn.Module):
    """y = x W^T + scale (x A^T) B^T, with W from `weight()` (an NF4 dequantise on Kaggle)."""

    def __init__(self, weight, in_features, out_features, rank, scale, device):
        super().__init__()
        self.weight_fn, self.device, self.scale = weight, torch.device(device), scale
        self.in_features, self.out_features = in_features, out_features
        if rank:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device), requires_grad=False)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device), requires_grad=False)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x):
        home = x.device
        x = x.to(self.device)
        y = F.linear(x, self.weight_fn().to(x.dtype))
        if self.lora_A is not None:
            update = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
            y = y + self.scale * update.to(y.dtype)
        return y.to(home)


def one(root, pattern):
    hits = sorted(glob.glob(os.path.join(str(root), "**", pattern), recursive=True))
    assert len(hits) == 1, f"{pattern}: {hits}"
    return hits[0]


def find_assets(root):
    """Paths of the NF4 base, the two full-text adapters, the offline wheels and the fold plan."""
    wheels = sorted(glob.glob(os.path.join(str(root), "**", "qwen38-offline-linux", "*.whl"), recursive=True))
    assert wheels, "no offline wheels"
    return {
        "base": os.path.dirname(one(root, "qwen38-27b-nf4/config.json")),
        "final": one(root, "ntl627-own-fulltext/final.safetensors"),
        "outer0": one(root, "ntl627-own-fulltext/outer0.safetensors"),
        "wheels": wheels,
        "fold_plan": one(root, "ntl-oof-q35-heads/fold_plan.json"),
    }


def heldout_rows(fold_plan, train_csv, prompt_ids):
    """Train rows the outer0 adapter never saw: the bundle's outer0 valid split, checked against our train file."""
    plan = json.loads(Path(fold_plan).read_text())
    assert plan["train_sha256"] == hashlib.sha256(Path(train_csv).read_bytes()).hexdigest(), "fold plan is for another train.csv"
    fit, valid = plan["jobs"]["outer0"]["fit"], plan["jobs"]["outer0"]["valid"]
    prompt_ids = np.asarray(prompt_ids)
    assert not set(prompt_ids[fit]) & set(prompt_ids[valid]), "outer0 split shares prompts"
    return np.asarray(valid)


def install_wheels(wheels, target="/tmp/qwen27b-python"):
    """Install the bundled wheels without the internet and put them first on sys.path. Call before importing transformers."""
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet",
                    "--disable-pip-version-check", "--target", target, *wheels], check=True)
    sys.path.insert(0, target)
    return target


class HostEmbedding(nn.Module):
    """The 2.5 GB embedding table stays in CPU memory; only the looked-up rows go to the GPU."""

    def __init__(self, weight):
        super().__init__()
        self.table = weight

    def forward(self, ids):
        return F.embedding(ids.cpu(), self.table).to(ids.device)


class Qwen27B:
    """The NF4 backbone on every visible GPU, with one adapter loaded at a time."""

    def __init__(self, base, adapter):
        from accelerate.utils import set_module_tensor_to_device
        from bitsandbytes.functional import QuantState, dequantize_4bit
        from safetensors import safe_open
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        t0 = time.time()
        self.base = Path(base)
        self.device = torch.device("cuda:0")
        ngpu = torch.cuda.device_count()
        cfg = Qwen3_5TextConfig(**json.loads((self.base / "config.json").read_text())["text_config"])
        assert cfg.num_hidden_layers == 64 and cfg.hidden_size == 5120
        cfg._attn_implementation = "sdpa"
        cfg.use_cache = False
        self.max_len = cfg.max_position_embeddings
        with torch.device("meta"):
            model = qm.Qwen3_5TextModel(cfg)
        model.norm = nn.Identity()

        where = json.loads((self.base / "model.safetensors.index.json").read_text())["weight_map"]
        files = {name: safe_open(str(self.base / name), framework="pt") for name in set(where.values())}

        def tensor(key):
            return files[where[key]].get_tensor(key)

        linears = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            parts = name.split(".")
            assert parts[0] == "layers" and module.bias is None, f"unexpected linear {name}"
            layer = int(parts[1])
            device = torch.device(f"cuda:{layer * ngpu // 64}")
            key = PREFIX + name + ".weight"
            if key + ".quant_state.bitsandbytes__nf4" in where:
                state = QuantState.from_dict({k: tensor(k) for k in where if k.startswith(key + ".")}, device=device)
                assert tuple(state.shape) == (module.out_features, module.in_features), name
                packed = tensor(key).to(device)
                weight = lambda packed=packed, state=state: dequantize_4bit(packed, state)
            else:
                plain = tensor(key).to(device)
                weight = lambda plain=plain: plain
            rank = RANK if layer in LORA_LAYERS else 0
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr,
                    LoraLinear(weight, module.in_features, module.out_features, rank, SCALE, device))
            linears += 1

        model.embed_tokens = HostEmbedding(tensor(PREFIX + "embed_tokens.weight"))
        model.rotary_emb = qm.Qwen3_5TextRotaryEmbedding(cfg, device=self.device)
        for name, param in list(model.named_parameters()):
            if ".lora_" in name:
                continue
            value = tensor(PREFIX + name)
            set_module_tensor_to_device(model, name, self.device, value=value, dtype=value.dtype)
        assert not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers())
        # T4 has no Ampere Triton kernels: use the reference torch gated-delta and conv paths.
        for layer in model.layers:
            attn = getattr(layer, "linear_attn", None)
            if attn is not None:
                attn.chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule
                attn.recurrent_gated_delta_rule = qm.torch_recurrent_gated_delta_rule
                attn.causal_conv1d_fn = None
                attn.causal_conv1d_update = None
        del files
        gc.collect()

        self.model = model.eval()
        self.norms = nn.ModuleList([nn.LayerNorm(5120) for _ in DEPTHS]).to(self.device).eval()
        self.plain = nn.Linear(8 * 5120, 18).to(self.device).eval()
        self.centered = nn.Linear(8 * 5120, 18).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base), local_files_only=True)
        self.load_adapter(adapter)
        self.info = {"load_seconds": round(time.time() - t0), "linears": linears, "gpus": ngpu}
        print("qwen27b loaded", json.dumps(self.info), flush=True)

    def load_adapter(self, adapter):
        """Swap in the LoRA weights, LayerNorms and heads of one adapter. Checks every key matches."""
        from safetensors.torch import load_file

        meta = json.loads(Path(adapter).with_suffix(".json").read_text())
        assert meta["executed_decoder_layers"] == 64 and meta["lora_target_layer_indices"] == list(LORA_LAYERS)
        assert meta["text_truncation"] is False and meta["inference_max_length"] is None
        assert meta["classification_token_id"] == END_ID and meta["rank"] == RANK and meta["alpha"] / RANK == SCALE
        assert meta["depths"] == DEPTHS
        self.labels, self.stage = meta["labels"], meta["stage"]
        saved = load_file(str(adapter), device="cpu")
        mine = {f"backbone.{n}": p for n, p in self.model.named_parameters() if ".lora_" in n}
        for prefix, module in [("norms", self.norms), ("plain", self.plain), ("centered", self.centered)]:
            mine.update({f"{prefix}.{n}": p for n, p in module.named_parameters()})
        assert set(saved) == set(mine), (sorted(set(saved) - set(mine))[:5], sorted(set(mine) - set(saved))[:5])
        with torch.no_grad():
            for key, param in mine.items():
                assert saved[key].shape == param.shape and torch.isfinite(saved[key]).all(), key
                param.copy_(saved[key])

    @torch.inference_mode()
    def logits(self, ids):
        batch = self.tokenizer.pad({"input_ids": ids}, padding=True, pad_to_multiple_of=64, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        assert mask.sum(1).tolist() == [len(x) for x in ids], "padding changed a passage"
        hidden = {}
        hooks = [self.model.layers[d - 1].register_forward_hook(
            lambda _m, _i, out, d=d: hidden.__setitem__(d, out[0] if isinstance(out, tuple) else out))
            for d in DEPTHS]
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                self.model(inputs_embeds=self.model.embed_tokens(input_ids), attention_mask=mask, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        out = pooled_logits([hidden[d] for d in DEPTHS], mask, self.norms, self.plain)
        assert torch.isfinite(out).all(), "non-finite logits"
        return out.float().cpu().numpy()

    def predict(self, texts, batch_size=3):
        """Probabilities in our LABELS order, one row per passage, in input order."""
        t0 = time.time()
        self.tokenizer.pad_token_id = END_ID
        self.tokenizer.padding_side = "right"
        raw = self.tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
        tokens = add_end(raw)
        lengths = np.array([len(x) for x in tokens])
        assert lengths.max() <= self.max_len, "a passage is longer than the native context"
        order = np.argsort(lengths, kind="stable")
        out = np.zeros((len(tokens), 18), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            try:
                out[rows] = self.logits([tokens[i] for i in rows])
                failed = False
            except torch.cuda.OutOfMemoryError:
                failed = True
            if failed:
                gc.collect()
                torch.cuda.empty_cache()
                for i in rows:
                    out[i] = self.logits([tokens[i]])[0]
            done = start + len(rows)
            if done % 300 < batch_size or done == len(order):
                print(f"qwen27b {self.stage}: {done}/{len(order)} at {time.time() - t0:.0f} s, "
                      f"max tokens {int(lengths[rows].max())}", flush=True)
        seconds = time.time() - t0
        e = np.exp(out - out.max(1, keepdims=True))
        probs = to_our_order(e / e.sum(1, keepdims=True), self.labels)
        self.last = {"stage": self.stage, "rows": len(tokens), "seconds": round(seconds),
                     "seconds_per_1000": round(1000 * seconds / len(tokens)),
                     "tokens_mean": round(float(lengths.mean())), "tokens_max": int(lengths.max()),
                     "peak_gb": [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]}
        print("qwen27b", json.dumps(self.last), flush=True)
        return probs


def run_qwen27b(asset_dir, texts, adapter="final"):
    """Probabilities [n, 18] in LABELS order from one adapter. The wheels must already be installed."""
    assets = find_assets(asset_dir)
    return Qwen27B(assets["base"], assets[adapter]).predict(texts)
