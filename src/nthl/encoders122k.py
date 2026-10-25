"""Run nuridudak's two-stage encoders, ModernBERT-large and desklib DeBERTa-v3-large (#28).

Each is a full fine-tune saved for AutoModelForSequenceClassification: stage 1
on the open 122k pool (nuridudak/ntl-130k-clean-passages), stage 2 on official
train. Both stored config.json files list the labels alphabetically, so
columns are reordered to our LABELS from each config's own id2label.
"""

import gc
import os
import shutil

import numpy as np

from nthl.labels import LABELS
from nthl.lora24 import to_our_order
from nthl.ultrabert import ckpt_labels

# id2label as both config.json files store it, kept for the tests.
NURIDUDAK_ORDER = tuple(sorted(LABELS))


def seqcls_labels(cfg):
    """The checkpoint's column order, from id2label keyed by the integer index."""
    return ckpt_labels(cfg)


def run_seqcls(model_dir, texts, max_len=512, batch_size=32):
    """Softmax probabilities [n, 18] in LABELS order, fp16 autocast on cuda."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    # A Dataset mount can crawl on large reads (#22), so copy to local disk first.
    local = os.path.join("/tmp/encoders122k", os.path.basename(model_dir.rstrip("/")))
    shutil.copytree(model_dir, local, dirs_exist_ok=True)
    cfg = AutoConfig.from_pretrained(local)
    labels = seqcls_labels(cfg.to_dict())
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False  # ModernBERT: no torch.compile on the T4
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSequenceClassification.from_pretrained(
        local, config=cfg, dtype=torch.float32).to("cuda:0").eval()
    ids = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
    order = np.argsort([len(x) for x in ids], kind="stable")
    p = np.zeros((len(texts), len(labels)))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to("cuda:0")
            p[idx] = model(**batch).logits.float().softmax(-1).cpu().numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(local)
    return to_our_order(p, labels)
