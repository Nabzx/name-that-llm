"""Joseph Adamski's published ATLAS adapter (josephadamski91/ntl-atlas-qwen3-8b-v1), run through its own
bundled inference package (Apache 2.0 code, CC BY-SA 4.0 weights). We add the row choice and a time budget:
rows go shortest first and any row not reached by the deadline is left out, never guessed."""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from nthl.labels import LABELS


def run_atlas(bundle_dir, base_path, texts, deadline, batch_size=4, source_chunk=3):
    """Probabilities [n, 18] in LABELS order and a mask of the rows that were scored before `deadline` (epoch secs)."""
    import torch
    from safetensors.torch import load_file

    root = Path(bundle_dir)
    sys.path.insert(0, str(root))
    from ntl_atlas.model import load_model

    meta = json.loads((root / "atlas_bundle.json").read_text())
    labels, cfg = meta["labels"], meta["training_config"]
    assert sorted(labels) == sorted(LABELS) and len(labels) == 18
    model, tok = load_model(base_path, labels, cfg, device="auto")
    model.load_trainable(load_file(str(root / "adapters.safetensors")))
    model.eval()

    ids = tok(list(texts), truncation=True, max_length=cfg["max_length"])["input_ids"]
    order = np.argsort([len(t) for t in ids], kind="stable")
    out = np.zeros((len(texts), 18), dtype=np.float32)
    done = np.zeros(len(texts), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            if time.time() > deadline:
                break
            idx = order[start:start + batch_size]
            pack = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt").to(model.device)
            p, _ = model.score(pack["input_ids"], pack["attention_mask"], source_chunk=source_chunk)
            out[idx] = p.float().cpu().numpy()
            done[idx] = True
            if start % (50 * batch_size) == 0:
                print(f"atlas {done.sum()}/{len(texts)}", flush=True)
    assert np.isfinite(out[done]).all() and np.allclose(out[done].sum(1), 1, atol=1e-4)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reorder = [labels.index(m) for m in LABELS]
    return out[:, reorder], done
