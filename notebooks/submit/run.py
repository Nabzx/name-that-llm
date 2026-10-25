"""Submission notebook (#28, #29): Nishant Dahal's published spine plus our own SVC.

Members, blended in log space with Nishant's published weights (#20):
- our four-view n-gram SVC, fitted here on host train plus our public pool at weight 0.5,
  read as a plain softmax of its raw scores like his SVC (weight 1.0);
- his Qwen3-14B LoRA (0.4) and Qwen2.5-14B LoRA (0.2), each with its centred twin (0.1)
  on test topic clusters of 2 to 18, run through our own loader (#27);
- his fine-tuned DeBERTa-v3-large (0.25).
Then Sinkhorn balancing at SINKHORN_ALPHA, skipped on the 3-row commit test.
No try/except around members: a member that breaks stops the run (#22).
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc
import glob
import json
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

from nthl.gauss import fit as gauss_fit, log_posteriors, transform as gauss_transform
from nthl.blend import cluster_similarity, low_margin_rows, mix_rows, peer_correct, sinkhorn, softmax, topic_groups
from nthl.encoder import run_encoder
from nthl.labels import LABELS
from nthl.lora24 import find_base, read_ours, read_prvsiyan, read_zbigniew, run_checkpoint, run_spec
from nthl.svc import fit_scores

POST = "sinkhorn"   # "sinkhorn" (half strength) or "peer" (peer correction, no class quotas, #46)
SINKHORN_ALPHA = 0.5   # half strength, as #23 recommends. Probe 1 (v2) used 1.0.
POOL_WEIGHT = 0.25   # best SVC on CV, 0.4838 (#63). Probes 1 and 2 used 0.5.
WEIGHTS = {"svc": 1.0, "qwen3_14b": 0.4, "qwen25_14b": 0.2, "deberta": 0.25}
TWIN_WEIGHT = 0.1
WITNESS_WEIGHT = 0.0   # lab witness head (#34 to #36), off until probed. +0.003 to +0.005 on the SVC OOF.
ZBIGNIEW_SHARE = 0.0   # probe 4 put Zbigniew Kirchner's in-style Qwen3-8B at 10%: LB 0.627 against 0.631, so off
ULTRABERT_SHARE = 0.0   # probe 5: pjmathematician's UltraBERT-380M, 5 folds averaged, a fixed 10% (#36)
ATLAS_SHARE = 0.0   # probe 5: Joseph Adamski's ATLAS at 10%, only on rows whose top two are under ATLAS_MARGIN apart
ATLAS_MARGIN = 0.3
ATLAS_STOP_HOURS = 7.5   # ATLAS stops here whatever is left, so the run stays inside 9 hours
PRVSIYAN_SHARE = 0.1   # probe 3: prvsiyan's two Qwen3-8B seeds, mixed in at a fixed 10% before balancing (#20)
OURS_SHARE = 0.1   # probe 7: our own Qwen3-8B LoRA24 from nabilshah/nthl-train-lora (#43) at a fixed 10%
GAUSS_WEIGHT = 0.4   # probe 8: shared-prompt Gaussian group head on our honest stage-1 features, grouped rows only (#46, LOG row 25)
GAUSS_T = 2.0   # temperature picked by held-out log loss in nthl-gauss-check

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
COMP = "/kaggle/input/competitions/name-that-llm"
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/members", exist_ok=True)
train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
test["text"] = test["text"].fillna("").astype(str)
big = len(test) >= 50
print("test rows", len(test), "gpus", torch.cuda.device_count(), flush=True)

groups, twin, tfidf = topic_groups(test["text"], return_matrix=True) if big else (None, None, None)
if big:
    print(f"topic clusters: {int(twin.sum())} passages in clusters of 2 to 18", flush=True)

logit = np.zeros((len(test), len(LABELS)))
timings = {}


def vote(name, p, w, rows=None):
    rows = slice(None) if rows is None else rows
    logit[rows] += w * np.log(np.clip(p, 1e-9, 1))
    np.save(f"{OUT}/members/{name}.npy", p)
    timings[name] = round(time.time() - T0)
    print(f"{name}: weight {w}, done at {timings[name]} s", flush=True)


def one(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert len(hits) == 1, f"{pattern}: {hits}"
    return hits[0]


# 1. Our SVC on host train plus our public pool.
pool = pd.read_parquet(one("pool.parquet"))
svc_raw = fit_scores(train, test["text"], pool=pool, pool_weight=POOL_WEIGHT)
vote("svc", softmax(svc_raw), WEIGHTS["svc"])
del pool

# 2 and 3. Nishant's two LoRA checkpoints through our loader.
LORAS = {
    "qwen3_14b": ("lora_qwen3_14b", {"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40}),
    "qwen25_14b": ("lora_qwen25_14b", {"model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 48}),
}
for name, (folder, spec) in LORAS.items():
    ckpt = f"/tmp/{folder}"
    shutil.copytree(os.path.dirname(one(f"{folder}/meta.json")), ckpt, dirs_exist_ok=True)
    p, centred, info = run_checkpoint(ckpt, find_base("/kaggle/input", spec), test["text"].tolist(), groups=groups, twin=twin)
    print(name, json.dumps(info), flush=True)
    vote(name, p, WEIGHTS[name])
    if centred is not None:
        vote(f"{name}_twin", centred, TWIN_WEIGHT, rows=twin)
    gc.collect()
    torch.cuda.empty_cache()
    print(name, "GPU memory left allocated:", [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())], flush=True)

# 4. Nishant's DeBERTa-v3-large.
vote("deberta", run_encoder(os.path.dirname(one("deberta_v3_large/meta.json")), test["text"].tolist()), WEIGHTS["deberta"])

# 6. Our lab witness: Qwen3-4B-Base likelihood features plus Gemma 4 tokens per word, a
# logistic-regression head fitted here on all of host train (#34).
if WITNESS_WEIGHT > 0:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    from nthl.witness import score_witness, tokens_per_word

    both = train["text"].tolist() + test["text"].tolist()
    qwen4 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36})
    feats, tpw = score_witness(qwen4, both)
    gemma_dir = os.path.dirname(one("gemma-4-e2b/**/tokenizer_config.json"))
    gtpw = np.asarray(tokens_per_word(AutoTokenizer.from_pretrained(gemma_dir), both), np.float32)
    x = np.hstack([feats, tpw[:, None], gtpw[:, None]])
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(x[: len(train)], train["model"])
    wp = head.predict_proba(x[len(train):])[:, [list(head.classes_).index(m) for m in LABELS]]
    vote("witness", wp, WITNESS_WEIGHT)

# 7. Shared-prompt Gaussian group head on our own LoRA's stage-1 features (#46). Stage 1 never saw host
# train, so the head is fitted on honest features. Grouped test rows are decoded jointly per topic cluster.
gauss_info = None
if GAUSS_WEIGHT > 0:
    local = "/tmp/stage1"
    shutil.copytree(os.path.dirname(one("ckpt_stage1/meta.json")), local, dirs_exist_ok=True)
    base8 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
    _, _, info, feats = run_spec(read_ours(local), base8, train["text"].fillna("").astype(str).tolist() + test["text"].tolist(), return_features=True)
    prm = gauss_fit(feats[: len(train)], np.array([LABELS.index(m) for m in train["model"]]), train["prompt_id"].to_numpy())
    rows = twin if big else np.zeros(len(test), bool)
    lp = log_posteriors(gauss_transform(feats[len(train):], prm), prm, groups if big else np.arange(len(test)), rows, T=GAUSS_T)
    np.save(f"{OUT}/members/gauss_stage1.npy", lp)
    if big:
        vote("gauss_stage1", np.exp(lp[twin]), GAUSS_WEIGHT, rows=twin)
    gauss_info = {"d": prm["d"], "grouped_rows": int(rows.sum()), "features_seconds": info["total_seconds"]}
    print("gauss", json.dumps(gauss_info), flush=True)
    del feats
    gc.collect()
    torch.cuda.empty_cache()
    timings["gauss"] = round(time.time() - T0)

if big and POST == "peer":
    logit = peer_correct(logit, svc_raw, groups, cluster_similarity(tfidf, groups), w_svc=WEIGHTS["svc"])
p = softmax(logit)

# 5. prvsiyan's two Qwen3-8B seeds, averaged, as a fixed share of the probabilities.
if PRVSIYAN_SHARE > 0:
    base8 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
    seeds = []
    for seed in ["full_s0", "full_s1"]:
        local = f"/tmp/{seed}"
        shutil.copytree(os.path.dirname(one(f"q8b_L24_ctr/{seed}/meta.json")), local, dirs_exist_ok=True)
        q, _, info = run_spec(read_prvsiyan(local), base8, test["text"].tolist())
        print(seed, json.dumps(info), flush=True)
        seeds.append(q)
        gc.collect()
        torch.cuda.empty_cache()
    prv = np.mean(seeds, axis=0)
    np.save(f"{OUT}/members/prvsiyan.npy", prv)
    timings["prvsiyan"] = round(time.time() - T0)

# 5b. Zbigniew's in-style Qwen3-8B.
zb = None
if ZBIGNIEW_SHARE > 0:
    local = "/tmp/zbigniew_final"
    shutil.copytree(os.path.dirname(one("final/readouts.npz")), local, dirs_exist_ok=True)
    base8 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
    zb, _, info = run_spec(read_zbigniew(local), base8, test["text"].tolist())
    print("zbigniew", json.dumps(info), flush=True)
    np.save(f"{OUT}/members/zbigniew.npy", zb)
    gc.collect()
    torch.cuda.empty_cache()
    timings["zbigniew"] = round(time.time() - T0)

# 5c. UltraBERT, the five fold models averaged.
ub = None
if ULTRABERT_SHARE > 0:
    from nthl.ultrabert import run_ultrabert

    root = glob.glob("/kaggle/input/**/ultrabert-380m-ntl", recursive=True)
    assert len(root) == 1, root
    ub = run_ultrabert(root[0], test["text"].tolist())   # copies each fold to local disk itself
    np.save(f"{OUT}/members/ultrabert.npy", ub)
    gc.collect()
    torch.cuda.empty_cache()
    timings["ultrabert"] = round(time.time() - T0)

# 5e. Our own Qwen3-8B LoRA24, the final checkpoint of nabilshah/nthl-train-lora.
ours = None
if OURS_SHARE > 0:
    local = "/tmp/ours_final"
    shutil.copytree(os.path.dirname(one("ours_final/meta.json")), local, dirs_exist_ok=True)
    base8 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
    ours, _, info = run_spec(read_ours(local), base8, test["text"].tolist())
    print("ours", json.dumps(info), flush=True)
    np.save(f"{OUT}/members/ours.npy", ours)
    gc.collect()
    torch.cuda.empty_cache()
    timings["ours"] = round(time.time() - T0)

# Published additions as fixed shares of the probabilities, before balancing (#20).
shares = {"prvsiyan": PRVSIYAN_SHARE if PRVSIYAN_SHARE > 0 else 0, "zbigniew": ZBIGNIEW_SHARE if zb is not None else 0,
          "ultrabert": ULTRABERT_SHARE if ub is not None else 0, "ours": OURS_SHARE if ours is not None else 0}
mixed = (1 - sum(shares.values())) * p
if shares["prvsiyan"]:
    mixed += PRVSIYAN_SHARE * prv
if shares["zbigniew"]:
    mixed += ZBIGNIEW_SHARE * zb
if shares["ultrabert"]:
    mixed += ULTRABERT_SHARE * ub
if shares["ours"]:
    mixed += OURS_SHARE * ours
p = mixed

# 5d. ATLAS on the rows the blend is unsure of, at a fixed 10% there, as in the public 0.636 recipe.
atlas_info = None
if ATLAS_SHARE > 0:
    from nthl.atlas import run_atlas

    bundle = glob.glob("/kaggle/input/**/atlas_bundle.json", recursive=True)
    assert len(bundle) == 1, bundle
    shutil.copytree(os.path.dirname(bundle[0]), "/tmp/atlas", dirs_exist_ok=True)
    base8 = find_base("/kaggle/input", {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
    rows = low_margin_rows(p, ATLAS_MARGIN) if big else np.arange(len(test))
    q, done = run_atlas("/tmp/atlas", base8, test["text"].iloc[rows].tolist(), deadline=T0 + ATLAS_STOP_HOURS * 3600)
    full = np.zeros_like(p)
    full[rows] = q
    p = mix_rows(p, full, rows[done], ATLAS_SHARE)
    np.save(f"{OUT}/members/atlas.npy", full)
    atlas_info = {"rows": int(len(rows)), "scored": int(done.sum())}
    print("atlas", json.dumps(atlas_info), flush=True)
    timings["atlas"] = round(time.time() - T0)

if big and POST == "sinkhorn":
    p = sinkhorn(p, alpha=SINKHORN_ALPHA)
sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.asarray(LABELS)[p.argmax(1)]})
assert len(sub) == len(test) and sub["id"].is_unique and sub["model"].isin(LABELS).all()
sub.to_parquet(f"{OUT}/submission.parquet", index=False)
np.save(f"{OUT}/members/blend.npy", p)
summary = {"rows": len(test), "post": POST, "sinkhorn_alpha": SINKHORN_ALPHA if big else None, "weights": WEIGHTS,
           "twin_weight": TWIN_WEIGHT, "prvsiyan_share": PRVSIYAN_SHARE, "zbigniew_share": ZBIGNIEW_SHARE, "ultrabert_share": ULTRABERT_SHARE, "ours_share": OURS_SHARE, "gauss_weight": GAUSS_WEIGHT, "gauss": gauss_info, "atlas_share": ATLAS_SHARE, "atlas": atlas_info, "witness_weight": WITNESS_WEIGHT, "pool_weight": POOL_WEIGHT, "timings": timings, "seconds": round(time.time() - T0),
           "predicted": sub["model"].value_counts().to_dict()}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
