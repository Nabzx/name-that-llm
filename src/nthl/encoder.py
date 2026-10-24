"""Run a published fine-tuned encoder checkpoint, such as Nishant Dahal's DeBERTa-v3-large (#28).

The whole encoder, a masked mean of the last layer (first token kept), one
linear head, fp32. Columns come back in our LABELS order.
"""

import json
import os
import shutil

import numpy as np
import torch

from nthl.lora24 import to_our_order


@torch.no_grad()
def run_encoder(ckpt_dir, texts, batch_size=32):
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    if ckpt_dir.startswith("/kaggle/input"):
        # A Dataset mount can crawl on large reads (#22), so copy once to local disk.
        local = os.path.join("/tmp", os.path.basename(ckpt_dir.rstrip("/")))
        shutil.copytree(ckpt_dir, local, dirs_exist_ok=True)
        ckpt_dir = local
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "encoder"))
    enc = AutoModel.from_pretrained(os.path.join(ckpt_dir, "encoder"), dtype=torch.float32).to("cuda:0").eval()
    head = load_file(os.path.join(ckpt_dir, "head.safetensors"))
    w, b = head["weight"].float().numpy(), head["bias"].float().numpy()
    ids = tok(list(texts), truncation=True, max_length=meta["max_len"])["input_ids"]
    z = np.zeros((len(ids), enc.config.hidden_size), np.float32)
    order = np.argsort([len(x) for x in ids], kind="stable")
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to("cuda:0")
        m = batch["attention_mask"].unsqueeze(-1).float()
        z[idx] = ((enc(**batch).last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
    logits = z @ w.T + b
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    del enc
    torch.cuda.empty_cache()
    return to_our_order(p, meta["labels"])
