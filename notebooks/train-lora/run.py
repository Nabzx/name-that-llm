"""Our own two-stage LoRA24 classifier on Qwen3-8B-Base, trained on Kaggle (#43).

Stage 1: a class-balanced sample of our public pool, stopped by a time budget.
Stage 2: two epochs on the 2,699 host passages. Checkpoints every hour so a
crash keeps the night's work. The final model goes to /kaggle/working/ours_final
in our own layout, read by nthl.lora24.read_ours.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import glob
import json
import random
import shutil
import sys
import time

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nthl.labels import LABELS
from nthl.lora24 import find_base, pool_layers
from nthl.train_lora import add_lora, save_checkpoint

SEED = 42
KEEP, LAYERS, MAX_LEN, EOS = 24, [2, 4, 6, 8, 12, 16, 20, 24], 320, 151643
R, ALPHA, DROP = 16, 32, 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MICRO, ACCUM = 4, 4
PER_CLASS = 2500                       # stage 1 sample, at most this many pool passages per model
STAGE1_HOURS = 6.5
STAGE1_LR, STAGE2_LR = (1e-4, 1e-3), (5e-5, 5e-4)   # (LoRA, norms and head)
STAGE2_EPOCHS = 2
OUT = "/kaggle/working"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
T0 = time.time()
log = []


def note(**kw):
    kw["minutes"] = round((time.time() - T0) / 60, 1)
    log.append(kw)
    print(json.dumps(kw), flush=True)
    json.dump(log, open(f"{OUT}/train_log.json", "w"))


COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv")
pool = pd.read_parquet(glob.glob("/kaggle/input/**/pool.parquet", recursive=True)[0])
pool = pool.groupby("model", group_keys=False).apply(lambda g: g.sample(min(len(g), PER_CLASS), random_state=SEED))
note(stage="data", pool_rows=len(pool), train_rows=len(train))

base = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
tok = AutoTokenizer.from_pretrained(base)
config = AutoConfig.from_pretrained(base)
config.num_hidden_layers = KEEP
config.use_cache = False
if getattr(config, "layer_types", None) is not None:
    config.layer_types = config.layer_types[:KEEP]
model = AutoModel.from_pretrained(base, config=config, dtype=torch.float16, device_map="auto",
                                  max_memory={i: "13GiB" for i in range(torch.cuda.device_count())})
model.norm = nn.Identity()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
wrapped = add_lora(model, TARGETS, R, ALPHA, DROP)
dev = model.embed_tokens.weight.device
hidden = model.config.hidden_size
norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in LAYERS]).to(dev).float()
head = nn.Linear(hidden * len(LAYERS), len(LABELS)).to(dev).float()
note(stage="model", wrapped=wrapped, gpus=torch.cuda.device_count())

lora_params = [p for p in model.parameters() if p.requires_grad]
label_id = {m: i for i, m in enumerate(LABELS)}


def encode(texts):
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=MAX_LEN - 1)["input_ids"]
    return [x + [EOS] for x in ids]


def batches(ids, ys, shuffle):
    order = np.random.permutation(len(ids)) if shuffle else np.arange(len(ids))
    chunk = MICRO * 64   # sort by length inside chunks to cut padding, keep the order random overall
    for s in range(0, len(order), chunk):
        part = sorted(order[s:s + chunk], key=lambda i: len(ids[i]))
        groups = [part[j:j + MICRO] for j in range(0, len(part), MICRO)]
        random.shuffle(groups)
        yield from groups


def forward(idx, ids, ys):
    enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        hs = model(**enc, output_hidden_states=True).hidden_states
    pooled = pool_layers(hs, enc["attention_mask"], LAYERS)
    z = torch.cat([nm(p.to(dev)) for nm, p in zip(norms, pooled)], -1)
    return head(z), torch.tensor([ys[i] for i in idx], device=dev)


def run_stage(name, texts, labels, epochs, lrs, hours, class_weight):
    ids, ys = encode(texts), [label_id[m] for m in labels]
    steps_total = max(1, int(epochs * len(ids) / (MICRO * ACCUM)))
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lrs[0]},
                             {"params": list(norms.parameters()) + list(head.parameters()), "lr": lrs[1]}], weight_decay=0.01)
    warm = max(1, int(0.1 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps_total))))
    scaler = torch.amp.GradScaler("cuda")
    weight = None
    if class_weight:
        counts = np.bincount(ys, minlength=len(LABELS)).astype(float)
        weight = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    start, last_save, step, micro, seen, run_loss, skipped = time.time(), time.time(), 0, 0, 0, 0.0, 0
    model.train()
    norms.train()
    head.train()
    for epoch in range(int(np.ceil(epochs))):
        for idx in batches(ids, ys, shuffle=True):
            logits, y = forward(idx, ids, ys)
            loss = F.cross_entropy(logits.float(), y, weight=weight) / ACCUM
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                micro = 0
                continue
            scaler.scale(loss).backward()
            run_loss += loss.item() * ACCUM
            micro += 1
            seen += len(idx)
            if micro == ACCUM:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lora_params + list(norms.parameters()) + list(head.parameters()), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step, micro = step + 1, 0
                if step % 50 == 0:
                    note(stage=name, step=step, of=steps_total, seen=seen, loss=round(run_loss / (50 * ACCUM), 4), skipped=skipped)
                    run_loss = 0.0
                if time.time() - last_save > 3600:
                    save_checkpoint(f"{OUT}/ckpt_{name}", model, norms, head, KEEP, LAYERS, MAX_LEN, EOS, R, ALPHA)
                    last_save = time.time()
                    note(stage=name, saved=True, step=step)
                if step >= steps_total or time.time() - start > hours * 3600:
                    note(stage=name, done=True, step=step, seen=seen, stopped_by_time=time.time() - start > hours * 3600)
                    return
    note(stage=name, done=True, step=step, seen=seen)


run_stage("stage1", pool["text"].tolist(), pool["model"].tolist(), 1, STAGE1_LR, STAGE1_HOURS, class_weight=True)
save_checkpoint(f"{OUT}/ckpt_stage1", model, norms, head, KEEP, LAYERS, MAX_LEN, EOS, R, ALPHA)
run_stage("stage2", train["text"].tolist(), train["model"].tolist(), STAGE2_EPOCHS, STAGE2_LR, 3.0, class_weight=False)
save_checkpoint(f"{OUT}/ours_final", model, norms, head, KEEP, LAYERS, MAX_LEN, EOS, R, ALPHA)
note(stage="end", total_hours=round((time.time() - T0) / 3600, 2))
