"""Run pjmathematician's five UltraBERT-380M fold fine-tunes as one member (#28).

Each fold is dmis-lab/UltraBERT-380M with a linear head on the first token,
shipped as fp16 safetensors with its own modelling code. We run every fold,
average the softmax probabilities, and return columns in our LABELS order,
read from each fold's own id2label.
"""

import gc
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from nthl.labels import LABELS
from nthl.lora24 import to_our_order


def find_folds(root_dir, n=5):
    """The n fold folders below root_dir, in numeric order. Each holds an UltraBERT config.json."""
    found = []
    for cfg in Path(root_dir).glob("**/fold*/config.json"):
        m = re.fullmatch(r"fold(\d+)", cfg.parent.name)
        if m and json.loads(cfg.read_text()).get("model_type") == "ultrabert":
            found.append((int(m.group(1)), cfg.parent))
    assert len(found) == n, f"expected {n} UltraBERT folds below {root_dir}, found {[str(p) for _, p in found]}"
    return [p for _, p in sorted(found)]


def ckpt_labels(cfg):
    """The checkpoint's column order, from id2label keyed by the integer index."""
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    labels = [id2label[i] for i in range(len(id2label))]
    assert sorted(labels) == sorted(LABELS), "checkpoint labels differ from ours"
    return labels


def run_ultrabert(root_dir, texts, batch_size=32, max_len=512):
    """Mean softmax of the five folds, [n, 18] in LABELS order."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = np.zeros((len(texts), len(LABELS)))
    folds = find_folds(root_dir)
    for fold in folds:
        # A Dataset mount can crawl on large reads (#22), so copy each fold to local disk first.
        local = os.path.join("/tmp/ultrabert", fold.name)
        shutil.copytree(fold, local, dirs_exist_ok=True)
        labels = ckpt_labels(json.load(open(os.path.join(local, "config.json"))))
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            local, trust_remote_code=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda:0").eval()
        ids = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        p = np.zeros((len(texts), len(labels)))
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to("cuda:0")
                p[idx] = model(**batch).logits.float().softmax(-1).cpu().numpy()
        out += to_our_order(p, labels) / len(folds)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(local)
    return out
