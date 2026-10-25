## Nabil (2026-10-25 10:04:10 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): report-draft = local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900 (done, PR #94); video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (done, PR #95); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (working); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

Background (from the session that started this one):
Repo ~/Random/research-arm-kaggle (GitHub Nabzx/name-that-llm, private) is Nabil's entry to the Imperial "Name That LLM" Kaggle competition (18-way LLM attribution, balanced accuracy). Final deadline 18:00 GMT today; results needed by about 12:45. Read AGENTS.md and CLAUDE.md: British English, no em dashes, Conventional Commits, small commits, refs #n, no co-author trailers, tests first. Hard rules: never submit to Kaggle; never handle Kaggle tokens; models are fitted only on Kaggle in our own private notebooks (CPU is fine), never locally beyond tiny unit tests; never fork or copy public notebooks; data must be public, free and cited; no hand-labelling of test data. Kaggle CLI is ~/Random/research-arm-kaggle/.venv/bin/kaggle. Code dataset nabilshah/nthl-src is uploaded with scripts/push_src.sh; other sessions upload it too, so message the main session (SendMessage to the session that started you) and wait for an OK before any upload. Never push nabilshah/nthl-submit. Work only in your own git worktree. Research notes on the hard pairs exist in docs/research/issue-38*, -39*, -40* (they found no lever then).

Task:
Make your own worktree first: `git -C ~/Random/research-arm-kaggle fetch -q && git -C ~/Random/research-arm-kaggle worktree add ~/Random/rak-pairs -b exp/hard-pairs origin/exp/gauss-heads`, work only there.

Goal: test, by cross-validation only (no submission), whether pairwise specialists fix our two hardest pairs: glm-5.3 vs glm-5.3-flash, and gemini-3.7-flash vs gemini-3.8-flash.

Design: for each pair, a binary LinearSVC on our four n-gram views (src/nthl/ngrams.py four_views) trained on host train rows of those two models plus our public pool's rows of those two models (pool weight 0.25, see nthl.svc), and optionally a version on nuridudak/ntl-130k-clean-passages (CC BY-SA 4.0) rows. Re-ranking rule: only when the base prediction's top two classes are exactly that pair, replace the order of those two by the specialist's decision (or add its log-odds with a weight). Base = the out-of-fold SVC predictions of LOG row 11 (kernel output nabilshah/nthl-svc-pool-all-w025, oof.parquet; scores there are log_softmax(raw * temperature), temperature in its summary.json). Use our frozen folds (nthl.folds.load_folds); fit the specialists per fold on the other folds only.

Notebook notebooks/pairs-check (private CPU kernel nabilshah/nthl-pairs-check, internet off; kernel_sources nabilshah/nthl-svc-pool-all-w025 and nabilshah/nthl-pool-build for pool.parquet). Report: pairwise accuracy of base vs specialist on rows of each pair, overall balanced accuracy of base vs base+specialists at a few weights, and how many rows change. Put any new functions in src/nthl/pairs.py with unit tests. Ask the main session before uploading nthl-src.

Done = PR in house format plus a short report of the numbers and whether it is worth adding to the submission (it is only worth it if overall OOF gain is clearly above noise, about +0.003).

Shared notes folder for this group of sessions (outside the repo; read siblings' notes there, leave hand-over notes or large findings there instead of pasting them into messages): /Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459

**Tool call: Bash**

```json
{
 "command": "git -C ~/Random/research-arm-kaggle fetch -q && git -C ~/Random/research-arm-kaggle worktree add ~/Random/rak-pairs -b exp/hard-pairs origin/exp/gauss-heads 2>&1 | tail -2; cd ~/Random/rak-pairs && cat AGENTS.md CLAUDE.md | head -150; ls src/nthl notebooks tests; ls \"/Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459\""
}
```

**Tool result**

```
branch 'exp/hard-pairs' set up to track 'origin/exp/gauss-heads'.
HEAD is now at c66c595 exp(svc): try nuridudak's 122k public pool in place of ours
# AGENTS.md

Structure and conventions for anyone, human or agent, working in this repository.

This repository is a solo entry to the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm), run as the Imperial Research Arm qualifier (Sat 24 Oct to Sun 25 Oct 2026, 18:00 UK).

## Competition invariants

These hold for every change. Breaking one can disqualify the entry.

- **Train on Kaggle only.** Every model fit runs in a Kaggle notebook under the account `nabilshah`. The local machine is for writing code, orchestrating runs and descriptive EDA.
- **No forking public notebooks.** Start from the official starter or from scratch. Public work may be read; cite it in `CITATIONS.md`.
- **No hand-labelling of test data.** Test passages are only ever read by our code at scoring time. Nobody, person or agent, labels them by hand or by asking an LLM. Balancing, centring and pseudo-labels are code in the notebook, not labelling.
- **Free and public only.** External data and pretrained models must be free and publicly available. Each one is listed in `CITATIONS.md`. No paid APIs.
- **Every AI session is logged.** Each Claude Code session, subagent run, workflow or other AI chat gets a row in `docs/ai-sessions.md` and is exported before submission. Missing logs mean disqualification.
- **Folds are frozen.** Cross-validation is `GroupKFold` on `prompt_id`, fixed in phase 0. Every experiment uses the same folds file. Changing it needs an ADR.
- **Every experiment is logged.** It saves out-of-fold and test predictions in the format of `src/nthl/preds.py`, scores them with `src/nthl/metric.py` and adds a row to `docs/experiments/LOG.md`: the change, local CV against the baseline, and the next step.
- **Seeds are fixed** in every notebook and script.
- **The repository stays private** until after the deadline. No code or ideas are shared with other participants.
- **What we make stays private on Kaggle too.** Our code dataset (`nabilshah/nthl-src`), notebooks and trained weights are private and shared only with the organisers at hand-in. What we take from others must be public and free.

## Phases

Work is organised into phases. Each phase maps to one milestone (`vX.Y.Z`) and one `phase-N` label. A phase ends when its milestone closes. `ROADMAP.md` holds the live phase table.

## Tracks

If the work splits into parallel workstreams, each gets a `track-*` label (for example `track-a`, `track-b`). Not in use until that happens.

## The wayfinder protocol

Each phase opens with one `wayfinder:map` issue: the phase's single source of truth for what's known, decided, and next.

A map issue body has three sections, kept up to date as children close:

- **Fog**: open unknowns.
- **Decisions so far**: what's been settled, and why.
- **Frontier**: the next unblocked tickets.

Child tickets han
... (4881 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && cat src/nthl/svc.py src/nthl/ngrams.py src/nthl/folds.py; ls notebooks/svc-pool-all-w025; cat notebooks/svc-pool-all-w025/kernel-metadata.json"
}
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && cat notebooks/svc-pool-all-w025/*.py 2>/dev/null | head -250; ls notebooks/svc-pool-all-w025 notebooks/svc-pool122k; cat src/nthl/metric.py | head -40; cat src/nthl/labels.py | head -40"
}
```

**Tool result**

```
"""The four-view n-gram LinearSVC, with or without the public pool.

One model per frozen fold gives out-of-fold predictions on host train, one
model on all of train gives test predictions. Pool rows join every training
fold at a lower weight. They never join validation, and they were filtered
against every train passage, so no fold sees its own prompts.

LinearSVC gives scores, not probabilities, so scores go through a softmax
with one temperature picked on out-of-fold log loss. It never changes the
top pick.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from nthl.labels import LABELS
from nthl.metric import balanced_accuracy
from nthl.ngrams import four_views
from nthl.preds import save_preds, to_labels


def _scores(model, texts):
    order = [list(model.classes_).index(label) for label in LABELS]
    return model.decision_function(texts)[:, order]


def _frame(ids, logp):
    df = pd.DataFrame(np.exp(logp), columns=list(LABELS))
    df.insert(0, "id", np.asarray(ids))
    return df


def run(train, test, pool=None, pool_weight=1.0, routes=None, drop_pool_labels=None, C=0.5, min_df=2, out=".", name="svc", seed=42):
    """Fit, write oof.parquet, test.parquet and summary.json to `out`, return the summary.

    `train` needs id, text, model and fold. `pool` needs text, model and route.
    `drop_pool_labels` leaves those models' pool rows out, so they learn from host train only.
    """
    start = time.time()
    out = Path(out)
    if pool is not None and routes is not None:
        pool = pool[pool["route"].isin(routes)]
    if pool is not None and drop_pool_labels:
        pool = pool[~pool["model"].isin(drop_pool_labels)]
    extra_text = [] if pool is None else pool["text"].tolist()
    extra_label = [] if pool is None else pool["model"].tolist()

    def fit(texts, labels):
        model = make_pipeline(four_views(min_df=min_df), LinearSVC(C=C, class_weight="balanced", random_state=seed))
        weights = np.r_[np.ones(len(texts)), np.full(len(extra_text), pool_weight)]
        model.fit(list(texts) + extra_text, list(labels) + extra_label, linearsvc__sample_weight=weights)
        return model

    oof = np.zeros((len(train), len(LABELS)))
    per_fold = {}
    for k in sorted(train["fold"].unique()):
        tr = (train["fold"] != k).to_numpy()
        va = ~tr
        model = fit(train.loc[tr, "text"], train.loc[tr, "model"])
        oof[va] = _scores(model, train.loc[va, "text"])
        pred = np.asarray(LABELS)[oof[va].argmax(axis=1)]
        per_fold[int(k)] = balanced_accuracy(train.loc[va, "model"], pred)
        print(f"{name} fold {k}: {per_fold[int(k)]:.4f} ({time.time() - start:.0f} s)", flush=True)

    truth = train["model"].to_numpy()
    true_idx = np.array([LABELS.index(m) for m in truth])
    temps = np.logspace(-1, 2, 61)
    losses = [-log_softmax(oof * t,
... (7012 more characters in the raw transcript)
```

**Tool result**

```
"""SVC run pool-all-w025 (#26). Config: {'pool': True, 'pool_weight': 0.25, 'routes': None, 'C': 0.5}"""

import glob
import json
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, f"expected one copy of nthl-src, found {code}"
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.folds import load_folds
from nthl.svc import run

CONFIG = {'pool': True, 'pool_weight': 0.25, 'routes': None, 'C': 0.5}
COMP = "/kaggle/input/competitions/name-that-llm"

train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
assert len(train) == 2699

pool = None
if CONFIG.pop("pool"):
    path = glob.glob("/kaggle/input/**/pool.parquet", recursive=True)
    assert len(path) == 1, f"expected one pool.parquet, found {path}"
    pool = pd.read_parquet(path[0])

summary = run(train, test, pool=pool, out="/kaggle/working", name="pool-all-w025", **CONFIG)
print(json.dumps(summary, indent=1))
notebooks/svc-pool-all-w025:
kernel-metadata.json
run.py

notebooks/svc-pool122k:
kernel-metadata.json
run.py
"""Balanced accuracy, as Kaggle scores this competition.

The mean, over the models that appear in the truth, of the share of that
model's passages labelled correctly. A model that is only ever predicted,
never true, adds no term.
"""

import numpy as np


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "truth and predictions differ in length"
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))
"""The 18 candidate models, in the order the competition page lists them.

Every prediction file uses this column order.
"""

LABELS = (
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && cat notebooks/svc-pool122k/run.py notebooks/svc-pool122k/kernel-metadata.json; cat src/nthl/preds.py | head -40; git log --oneline -5; grep -n \"122k\\|ntl-130k\" CITATIONS.md docs/experiments/LOG.md | head; sed -n 1,30p docs/experiments/LOG.md | cut -c1-250"
}
```

**Tool result**

```
"""SVC run pool122k: row 11 (pool weight 0.25, C 0.5) with nuridudak's 122k public pool (CC BY-SA 4.0)
in place of our 84k pool. The two sources we leave out of our own pool for licence reasons are dropped
here too. CV only, no submission."""

import glob
import json
import shutil
import sys

code = glob.glob("/kaggle/input/**/nthl-src", recursive=True)
assert len(code) == 1, code
shutil.copytree(code[0], "/tmp/pkg/nthl", dirs_exist_ok=True)
sys.path.insert(0, "/tmp/pkg")

import pandas as pd

from nthl.folds import load_folds
from nthl.labels import LABELS
from nthl.svc import run

COMP = "/kaggle/input/competitions/name-that-llm"
train = pd.read_csv(f"{COMP}/train.csv").merge(load_folds()[["id", "fold"]], on="id", validate="one_to_one")
test = pd.read_csv(f"{COMP}/test.csv")
path = glob.glob("/kaggle/input/**/ntl_encoder_pool_122k.parquet", recursive=True)
assert len(path) == 1, path
pool = pd.read_parquet(path[0])
print("pool", pool.shape, list(pool.columns), flush=True)
src = pool["source"].astype(str) if "source" in pool else pd.Series("", index=pool.index)
excluded = src.str.contains("olegzholobov|ankiteshtiwari", case=False)
print("dropped for licence:", int(excluded.sum()), flush=True)
pool = pool[~excluded & pool["model"].isin(LABELS)]
pool = pool[~pool["text"].isin(set(train["text"]))][["text", "model"]].reset_index(drop=True)
print("pool used", len(pool), pool["model"].value_counts().to_dict(), flush=True)

summary = run(train, test, pool=pool, pool_weight=0.25, C=0.5, out="/kaggle/working", name="pool122k")
print(json.dumps(summary, indent=1))
{
  "id": "nabilshah/nthl-svc-pool122k",
  "title": "nthl svc pool122k",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": ["nabilshah/nthl-src", "nuridudak/ntl-130k-clean-passages"],
  "competition_sources": ["name-that-llm"],
  "kernel_sources": [],
  "model_sources": []
}
"""One file format for every member's predictions.

A prediction file has an `id` column and one probability column per model,
in the order of `LABELS`. Rows sum to 1. Each experiment writes two files:

    outputs/<experiment>/oof.parquet    out-of-fold, one row per train passage
    outputs/<experiment>/test.parquet   one row per test passage

so any member can be blended with any other later.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nthl.labels import LABELS

COLUMNS = ["id", *LABELS]


def check_preds(df, ids=None):
    assert list(df.columns) == COLUMNS, "columns must be id then the 18 labels in order"
    assert df["id"].is_unique, "repeated id"
    p = df[list(LABELS)].to_numpy(dtype=float)
    assert np.isfinite(p).all(), "missing or infinite probability"
    assert (p >= 0).all() and (p <= 1).all(), "probability outside 0 to 1"
    assert np.allclose(p.sum(axis=1), 1, atol=1e-4), "a row does not sum to 1"
    if ids is not None:
        asse
... (4964 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && ls docs/research | head -50; grep -rn \"glm-5.3-flash\\|gemini-3.8-flash\" docs/research/*census* docs/experiments/LOG.md 2>/dev/null | cut -c1-300 | head -10; cat tests/test_svc.py | head -40; cat scripts/push_src.sh"
}
```

**Tool result**

```
README.md
issue-17-kaggle-cli-runs.md
issue-18-published-checkpoints.md
issue-19-public-pools.md
issue-20-blend-validation.md
issue-21-lora24-readout.md
issue-22-kaggle-pitfalls.md
issue-23-sinkhorn-and-centring.md
issue-24-what-top-teams-do.md
issue-31-witness-models.md
issue-32-likelihood-attribution.md
issue-33-likelihood-features.md
issue-38-glm-pair.md
issue-39-gemini-flash-pair.md
issue-40-gpt56-trio.md
docs/experiments/LOG.md:7:| 1 | 2026-10-24 | #25 | Four-view n-gram LinearSVC (C 0.5, balanced class weights), host train only. Code `nthl-src` @ 343c099, notebook `nthl-svc-train` v1, CPU, 114 s | 0.3485 | baseline | not submitted | Folds range 0.31 to 0.39. The lab is right 0.57 of the time (balan
import numpy as np
import pandas as pd

from nthl.labels import LABELS
from nthl.preds import load_preds
from nthl.svc import run


def toy(n_per=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, label in enumerate(LABELS):
        for j in range(n_per):
            words = rng.choice(["alpha", "beta", "gamma", "delta", "omega"], 30)
            rows.append({"id": f"{i}-{j}", "text": f"marker{i} " + " ".join(words), "model": label, "fold": j % 5})
    return pd.DataFrame(rows)


def test_run_writes_valid_files_and_learns_an_easy_task(tmp_path):
    train = toy()
    test = toy(n_per=1, seed=1).drop(columns=["model", "fold"]).assign(id=lambda d: "t" + d["id"])
    summary = run(train, test, out=tmp_path, name="toy")
    assert summary["cv"] > 0.9
    oof = load_preds(tmp_path / "oof.parquet")
    assert set(oof["id"]) == set(train["id"])
    assert len(load_preds(tmp_path / "test.parquet")) == len(test)


def test_run_keeps_only_the_asked_routes(tmp_path):
    train = toy()
    test = toy(n_per=1, seed=1).drop(columns=["model", "fold"]).assign(id=lambda d: "t" + d["id"])
    pool = toy(n_per=4, seed=2).drop(columns=["id", "fold"])
    pool["source"] = "x/y"
    pool["route"] = np.where(np.arange(len(pool)) % 2 == 0, "api", "proxy")
    summary = run(train, test, pool=pool, routes=["api"], pool_weight=0.5, out=tmp_path, name="toy")
    assert summary["pool_rows"] == (pool["route"] == "api").sum()
    assert summary["pool_weight"] == 0.5


def test_fit_scores_gives_raw_scores_in_our_label_order():
#!/usr/bin/env bash
# Upload src/nthl to the private Kaggle dataset nabilshah/nthl-src, so
# notebooks can import our code. The first run creates the dataset, later
# runs add a version named after the current commit.
#
#   scripts/push_src.sh [version message]
set -euo pipefail
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
OUT=build/nthl-src

rm -rf "$OUT" && mkdir -p "$OUT"
cp -R src/nthl "$OUT/nthl"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
cat > "$OUT/dataset-metadata.json" <<JSON
{"title": "nthl-src", "id": "nabilshah/nthl-src", "licenses": [{"name": "other"}]}
JSON

MSG="${1:-$(git rev-parse --short HEAD)}"
if $K datasets list --mine --search nthl-src 2>/dev/null | grep -q "^nabilshah/nthl-src "; then
  $K datasets
... (81 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && gh issue list --limit 40 --state all 2>/dev/null | grep -i \"pair\\|glm\\|gemini\" ; tail -c 1500 docs/research/issue-38-glm-pair.md; tail -c 1200 docs/research/issue-39-gemini-flash-pair.md"
}
```

**Tool result**

```
64	CLOSED	Leave the pool's gemini-3.7-flash and 3.8-flash rows out	phase-1, wayfinder:task	2026-10-24T16:18:42Z
42	CLOSED	A specialist for one hard pair	phase-3, wayfinder:prototype	2026-10-24T20:32:53Z
39	CLOSED	What separates gemini-3.7-flash from gemini-3.8-flash	phase-3, wayfinder:research	2026-10-24T20:11:46Z
38	CLOSED	What separates glm-5.3 from glm-5.3-flash	phase-3, wayfinder:research	2026-10-24T20:11:43Z
e.** The strongest log-odds scores are about 2.5, which is noise level across thousands of words.

| glm-5.3 favours | z | glm-5.3-flash favours | z |
|---|---|---|---|
| system | 2.4 | rather | 2.3 |
| so | 2.3 | we | 2.2 |
| stand | 2.2 | door | 2.2 |
| note | 2.1 | most | 2.2 |
| three | 2.0 | real | 2.1 |

Flash leans slightly towards "we", "rather", "whether" and semicolons, a little more essayistic. Too weak to use.

**Openings and closings match.** Both open with "The", "In", "Imagine", "Beneath", "Deep", "When". Every passage of both ends with a full stop. No title lines.

**The SVC barely splits them, and loses more to Claude than to each other.**

| True \ predicted | glm-5.3 | glm-5.3-flash | any Claude | other |
|---|---|---|---|---|
| glm-5.3 | 28 | 36 | 66 | 20 |
| glm-5.3-flash | 20 | 46 | 52 | 32 |

Recall is 0.19 for glm-5.3 and 0.31 for flash. Restricted to the two GLM columns, the pair is split at 0.54 balanced accuracy, close to a coin flip. Claude passages also flow the other way: 44 claude-fable-5 passages are read as GLM. A per-class offset on glm-5.3's score, tuned on four folds and applied to the fifth, lowers CV by 0.002.

## Recommendation

A clear no for the pair. Nothing in the surface text separates glm-5.3 from flash, and no offset or feature we can see would help. Treat the two as near-interchangeable and spend the effort on GLM against Claude, where 118 of the 300 GLM passages go. That is a larger and more tractable loss than the pair itself.
3.7 and 0.73 for 3.8. Restricted to the pair, balanced accuracy is 0.54. The 3.7 passages read as 3.8 are longer on average (121 words) than the ones read correctly (107), so the SVC is not using length.

**The routing story.** Discussion 751370 checked the host's labels and found no sign of 3.8 text under the 3.7 label. Its host-trained models err both ways, and its LoRA reads 3.8 as 3.7 more often than the reverse. Our LOG agrees: host-only, 3.7 recall was 0.41. It fell to 0.19 once the pool came in (row 6), and 7 of 12 pool sources have no 3.7 rows. Dropping the pool's 3.7 and 3.8 rows made both worse (row 8, -0.030). The pull towards 3.8 comes from the pool, not the host text.

A per-class offset on 3.7's score, tuned on four folds and applied to the fifth, picks offsets near zero and lowers CV by 0.002. The bias is not a simple prior shift.

## Recommendation

No text-level lever. Length is real but too spread to act on, and the SVC's errors do not follow it. The one route left is the pool's labels: weight or filter Gemini 3.7 and 3.8 pool rows by source, keepin
... (116 more characters in the raw transcript)
```

## Claude (2026-10-25 10:04:54 UTC)

Worktree is set up; I've read the SVC code and the earlier pair notes. Next I'll write `src/nthl/pairs.py` with its tests.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && gh issue view 42 --comments | head -60"
}
```

**Tool result**

```
author:	Nabzx
association:	owner
edited:	false
status:	none
--
Not planned. #38 found the GLM pair at a coin flip with no textual lever, and a per-class offset check lowered CV, so a specialist has nothing to learn.
--
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && gh issue create --title \"Pairwise specialists for the two hardest pairs, CV only\" --label \"phase-3,wayfinder:task\" --body \"Part of the hard-pairs work (#38, #39, #42).\n\nA binary four-view LinearSVC for glm-5.3 vs glm-5.3-flash and one for gemini-3.7-flash vs gemini-3.8-flash, fitted per frozen fold on host rows plus pool rows of the two models (weight 0.25). Re-rank row 11's out-of-fold SVC predictions only where its top two are exactly that pair, by swap or by adding the specialist's log-odds at a weight. A second variant uses nuridudak/ntl-130k-clean-passages (CC BY-SA 4.0) as the pool.\n\nDone when the notebook reports pairwise accuracy (base vs specialist), overall balanced accuracy at a few weights and the number of changed rows. Worth adding only if the OOF gain is clearly above noise (about +0.003). No submission.\" 2>&1 | tail -1"
}
```

**Tool result**

```
https://github.com/Nabzx/name-that-llm/issues/97
```

**Tool call: Write**

```json
{
 "file_path": "/Users/nabilshah/Random/rak-pairs/tests/test_pairs.py",
 "content": "import numpy as np\nimport pandas as pd\n\nfrom nthl.labels import LABELS\nfrom nthl.pairs import fit_pair, pair_margin, rerank, top_two_is\n\nA, B = \"glm-5.3\", \"glm-5.3-flash\"\nIA, IB = LABELS.index(A), LABELS.index(B)\n\n\ndef scores(rows):\n    \"\"\"One row per (first, second) pair of labels: first scores 0, second -1, the rest -5.\"\"\"\n    s = np.full((len(rows), len(LABELS)), -5.0)\n    for r, (first, second) in enumerate(rows):\n        s[r, LABELS.index(first)] = 0.0\n        s[r, LABELS.index(second)] = -1.0\n    return s\n\n\ndef test_top_two_is_ignores_order_and_needs_both():\n    s = scores([(A, B), (B, A), (A, \"kimi-k3\"), (\"kimi-k3\", B)])\n    assert top_two_is(s, A, B).tolist() == [True, True, False, False]\n\n\ndef test_rerank_swap_follows_the_sign_of_the_margin():\n    s = scores([(A, B), (A, B), (B, A), (\"kimi-k3\", A)])\n    margin = np.array([1.0, -1.0, 2.0, -3.0])  # positive favours A\n    out = rerank(s, A, B, margin)\n    top = np.asarray(LABELS)[out.argmax(axis=1)]\n    assert top.tolist() == [A, B, A, \"kimi-k3\"]\n    assert np.array_equal(out[3], s[3])\n    assert sorted(out[1]) == sorted(s[1])\n\n\ndef test_rerank_with_a_weight_adds_log_odds_on_pair_rows_only():\n    s = scores([(A, B), (\"kimi-k3\", A)])\n    margin = np.array([-0.4, -9.0])\n    small = rerank(s, A, B, margin, weight=1.0)\n    assert small.argmax(axis=1)[0] == IA\n    assert np.isclose(small[0, IA] - small[0, IB], 1.0 - 0.4)\n    big = rerank(s, A, B, margin, weight=5.0)\n    assert big.argmax(axis=1)[0] == IB\n    assert np.array_equal(big[1], s[1])\n    assert np.isclose(big[0].sum(), s[0].sum())\n\n\ndef test_fit_pair_learns_an_easy_pair_and_uses_extra_rows():\n    rng = np.random.default_rng(0)\n    words = lambda: \" \".join(rng.choice([\"alpha\", \"beta\", \"gamma\", \"delta\"], 20))\n    texts = [f\"left {words()}\" for _ in range(10)] + [f\"right {words()}\" for _ in range(10)]\n    labels = [A] * 10 + [B] * 10\n    extra = pd.DataFrame({\"text\": [f\"left {words()}\" for _ in range(5)], \"model\": [A] * 5})\n    model = fit_pair(texts, labels, A, B, extra=extra, extra_weight=0.25)\n    m = pair_margin(model, [\"left \" + words(), \"right \" + words()], A)\n    assert m[0] > 0 > m[1]\n"
}
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && ls .venv 2>/dev/null | head -2; ls ~/Random/research-arm-kaggle/.venv/bin/python && PYTHONPATH=src ~/Random/research-arm-kaggle/.venv/bin/python -m pytest -q tests/test_pairs.py 2>&1 | tail -3"
}
```

**Tool result**

```
File created successfully at: /Users/nabilshah/Random/rak-pairs/tests/test_pairs.py (file state is current in your context — no need to Read it back)
```

**Tool result**

```
/Users/nabilshah/Random/research-arm-kaggle/.venv/bin/python
ERROR tests/test_pairs.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.78s
```

## Claude (2026-10-25 10:05:18 UTC)

Red as expected. Committing the tests, then the module.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && git add tests/test_pairs.py && git commit -qm \"test(pairs): add failing tests for pairwise specialists\n\nrefs #97\" && git log --oneline -1"
}
```

**Tool result**

```
394b096 test(pairs): add failing tests for pairwise specialists
```

**Tool call: Write**

```json
{
 "file_path": "/Users/nabilshah/Random/rak-pairs/src/nthl/pairs.py",
 "content": "\"\"\"Pairwise specialists for the hardest pairs (#97).\n\nA binary four-view LinearSVC sees only passages of two models. Where the\nbase prediction's top two are exactly that pair, the specialist re-ranks\nthem: by its sign alone (a swap), or by adding its margin to the pair's\nlog-odds at a weight. Every other row is left as it was.\n\"\"\"\n\nimport numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.svm import LinearSVC\n\nfrom nthl.labels import LABELS\nfrom nthl.ngrams import four_views\n\n\ndef top_two_is(scores, a, b):\n    \"\"\"Rows whose two highest scores are a and b, in either order.\"\"\"\n    top2 = np.argsort(-scores, axis=1)[:, :2]\n    return np.sort(top2, axis=1).tolist() == [] or (np.sort(top2, axis=1) == sorted((LABELS.index(a), LABELS.index(b)))).all(axis=1)\n\n\ndef rerank(scores, a, b, margin, weight=None):\n    \"\"\"Re-rank a against b on rows where they are the top two. `margin` > 0 favours a.\n\n    weight None swaps the two scores where the margin's sign disagrees with their order.\n    A number adds weight * margin to the a-minus-b gap, split evenly so the row sum stays put.\n    \"\"\"\n    ia, ib = LABELS.index(a), LABELS.index(b)\n    out = np.array(scores, dtype=float)\n    rows = top_two_is(out, a, b)\n    margin = np.asarray(margin, dtype=float)\n    if weight is None:\n        flip = rows & ((out[:, ia] > out[:, ib]) != (margin > 0))\n        out[flip, ia], out[flip, ib] = out[flip, ib], out[flip, ia].copy()\n    else:\n        shift = np.where(rows, weight * margin / 2, 0.0)\n        out[:, ia] += shift\n        out[:, ib] -= shift\n    return out\n\n\ndef fit_pair(texts, labels, a, b, extra=None, extra_weight=0.25, C=0.5, min_df=2, seed=42):\n    \"\"\"A balanced binary LinearSVC on the rows labelled a or b, plus `extra` rows (text, model) of those two at a lower weight.\"\"\"\n    labels = np.asarray(labels)\n    keep = np.isin(labels, [a, b])\n    texts = [t for t, k in zip(texts, keep) if k]\n    labels = labels[keep].tolist()\n    if extra is not None:\n        extra = extra[extra[\"model\"].isin([a, b])]\n        weights = np.r_[np.ones(len(texts)), np.full(len(extra), extra_weight)]\n        texts, labels = texts + extra[\"text\"].tolist(), labels + extra[\"model\"].tolist()\n    else:\n        weights = np.ones(len(texts))\n    model = make_pipeline(four_views(min_df=min_df), LinearSVC(C=C, class_weight=\"balanced\", random_state=seed))\n    model.fit(texts, labels, linearsvc__sample_weight=weights)\n    return model\n\n\ndef pair_margin(model, texts, a):\n    \"\"\"The specialist's decision on `texts`, signed so that > 0 favours a.\"\"\"\n    d = model.decision_function(list(texts))\n    return d if model.classes_[1] == a else -d\n"
}
```

**Tool result**

```
File created successfully at: /Users/nabilshah/Random/rak-pairs/src/nthl/pairs.py (file state is current in your context — no need to Read it back)
```

**Tool call: Edit**

```json
{
 "replace_all": false,
 "file_path": "/Users/nabilshah/Random/rak-pairs/src/nthl/pairs.py",
 "old_string": "    top2 = np.argsort(-scores, axis=1)[:, :2]\n    return np.sort(top2, axis=1).tolist() == [] or (np.sort(top2, axis=1) == sorted((LABELS.index(a), LABELS.index(b)))).all(axis=1)",
 "new_string": "    top2 = np.sort(np.argsort(-np.asarray(scores), axis=1)[:, :2], axis=1)\n    return (top2 == sorted((LABELS.index(a), LABELS.index(b)))).all(axis=1)"
}
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && PYTHONPATH=src ~/Random/research-arm-kaggle/.venv/bin/python -m pytest -q tests/ 2>&1 | tail -3"
}
```

**Tool result**

```
The file /Users/nabilshah/Random/rak-pairs/src/nthl/pairs.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

**Tool result**

```
...............ss............s.......................................... [ 97%]
..                                                                       [100%]
71 passed, 3 skipped in 6.77s
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && git add src/nthl/pairs.py && git commit -qm \"feat(pairs): add binary pair specialists and a top-two re-rank\n\nrefs #97\" && mkdir -p notebooks/pairs-check && cat notebooks/pool-build/kernel-metadata.json | head -5"
}
```

**Tool result**

```
{
  "id": "nabilshah/nthl-pool-build",
  "title": "nthl pool build",
  "code_file": "build.py",
  "language": "python",
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && cat > notebooks/pairs-check/kernel-metadata.json <<'JSON'\n{\n  \"id\": \"nabilshah/nthl-pairs-check\",\n  \"title\": \"nthl pairs check\",\n  \"code_file\": \"run.py\",\n  \"language\": \"python\",\n  \"kernel_type\": \"script\",\n  \"is_private\": true,\n  \"enable_gpu\": false,\n  \"enable_tpu\": false,\n  \"enable_internet\": false,\n  \"dataset_sources\": [\"nabilshah/nthl-src\", \"nuridudak/ntl-130k-clean-passages\"],\n  \"competition_sources\": [\"name-that-llm\"],\n  \"kernel_sources\": [\"nabilshah/nthl-svc-pool-all-w025\", \"nabilshah/nthl-pool-build\"],\n  \"model_sources\": []\n}\nJSON\ncat > notebooks/pairs-check/run.py <<'PY'\n\"\"\"Pairs check (#97): do binary specialists fix glm-5.3 vs glm-5.3-flash and gemini-3.7-flash vs 3.8-flash?\n\nBase is row 11's out-of-fold SVC (pool weight 0.25). Per frozen fold, a binary four-view LinearSVC per pair\nis fitted on the other folds' host rows of the two models plus pool rows of them at weight 0.25, with three\npools: none, ours, nuridudak's 122k (CC BY-SA 4.0). Only rows whose base top two are exactly the pair are\nre-ranked. CV only, no submission.\"\"\"\n\nimport glob\nimport json\nimport os\nimport shutil\nimport sys\nimport time\n\ncode = glob.glob(\"/kaggle/input/**/nthl-src\", recursive=True)\nassert len(code) == 1, code\nshutil.copytree(code[0], \"/tmp/pkg/nthl\", dirs_exist_ok=True)\nsys.path.insert(0, \"/tmp/pkg\")\n\nimport numpy as np\nimport pandas as pd\n\nfrom nthl.folds import load_folds\nfrom nthl.labels import LABELS\nfrom nthl.metric import balanced_accuracy\nfrom nthl.pairs import fit_pair, pair_margin, rerank, top_two_is\n\nPAIRS = [(\"glm-5.3\", \"glm-5.3-flash\"), (\"gemini-3.7-flash\", \"gemini-3.8-flash\")]\nWEIGHTS = [None, 0.25, 0.5, 1.0, 2.0, 4.0]\nCOMP = \"/kaggle/input/competitions/name-that-llm\"\nstart = time.time()\n\ntrain = pd.read_csv(f\"{COMP}/train.csv\").merge(load_folds()[[\"id\", \"fold\"]], on=\"id\", validate=\"one_to_one\")\nassert len(train) == 2699\noof_path = glob.glob(\"/kaggle/input/**/oof.parquet\", recursive=True)\nassert len(oof_path) == 1, oof_path\noof = pd.read_parquet(oof_path[0]).set_index(\"id\").loc[train[\"id\"], list(LABELS)]\ntemperature = json.load(open(os.path.join(os.path.dirname(oof_path[0]), \"summary.json\")))[\"temperature\"]\nbase = np.log(np.clip(oof.to_numpy(), 1e-300, None))\ntruth = train[\"model\"].to_numpy()\nfolds = train[\"fold\"].to_numpy()\n\n\ndef ba(scores, rows=slice(None)):\n    return balanced_accuracy(truth[rows], np.asarray(LABELS)[scores[rows].argmax(axis=1)])\n\n\nbase_ba = ba(base)\nprint(f\"base oof balanced accuracy {base_ba:.4f}, temperature {temperature:.3f}\", flush=True)\n\nours = pd.read_parquet(glob.glob(\"/kaggle/input/**/pool.parquet\", recursive=True)[0])[[\"text\", \"model\"]]\np122 = glob.glob(\"/kaggle/input/**/ntl_encoder_pool_122k.parquet\", recursive=True)\nassert len(p122) == 1, p122\nbig = pd.read_parquet(p122[0])\nsrc = big[\"source\"].astype(str) if \"source\" in big else pd.Series(\"\", index=big.index)\nbig = big[~src.str.contains(\"olegzholobov|ankiteshtiwari\", case=False) & big[\"model\"].isin(LABELS)]\nbig = big[~big[\"text\"].isin(set(train[\"text\"]))][[\"text\", \"model\"]].reset_index(drop=True)\nPOOLS = {\"host\": None, \"ours\": ours, \"122k\": big}\npair_models = [m for p in PAIRS for m in p]\nfor name, pool in POOLS.items():\n    if pool is not None:\n        print(name, \"pair rows\", pool[pool[\"model\"].isin(pair_models)][\"model\"].value_counts().to_dict(), flush=True)\n\nmargins = {}\nfor name, pool in POOLS.items():\n    for a, b in PAIRS:\n        m = np.zeros(len(train))\n        for k in sorted(np.unique(folds)):\n            tr, va = folds != k, folds == k\n            model = fit_pair(train.loc[tr, \"text\"].tolist(), truth[tr], a, b, extra=pool, extra_weight=0.25)\n            m[va] = pair_margin(model, train.loc[va, \"text\"], a)\n        margins[(name, a)] = m\n        print(f\"{name} {a} vs {b} done ({time.time() - start:.0f} s)\", flush=True)\n\nreport = {\"base_ba\": base_ba, \"temperature\": temperature, \"pairwise\": [], \"overall\": []}\nia_of = {m: LABELS.index(m) for m in pair_models}\nfor a, b in PAIRS:\n    on_pair = np.isin(truth, [a, b])\n    top = top_two_is(base, a, b)\n    base_says_a = base[:, ia_of[a]] > base[:, ia_of[b]]\n    for subset, rows in [(\"all pair rows\", on_pair), (\"pair rows with base top two = pair\", on_pair & top)]:\n        def pair_ba(says_a):\n            return float(np.mean([np.mean(says_a[rows & (truth == a)]), np.mean(~says_a[rows & (truth == b)])]))\n        row = {\"pair\": f\"{a} vs {b}\", \"subset\": subset, \"n\": int(rows.sum()), \"base\": pair_ba(base_says_a)}\n        for name in POOLS:\n            row[name] = pair_ba(margins[(name, a)] > 0)\n        report[\"pairwise\"].append(row)\n        print(json.dumps(row), flush=True)\n    print(f\"{a} vs {b}: base top two = pair on {int(top.sum())} rows, of which truly in the pair {int((top & on_pair).sum())}\", flush=True)\n\nbase_pred = base.argmax(axis=1)\nfor name in POOLS:\n    for which in [\"both\", *[a for a, _ in PAIRS]]:\n        for w in WEIGHTS:\n            s = base\n            for a, b in PAIRS:\n                if which in (\"both\", a):\n                    s = rerank(s, a, b, margins[(name, a)], weight=w)\n            pred = s.argmax(axis=1)\n            per_fold = [ba(s, folds == k) - ba(base, folds == k) for k in sorted(np.unique(folds))]\n            row = {\"pool\": name, \"pairs\": which, \"weight\": \"swap\" if w is None else w, \"ba\": ba(s),\n                   \"gain\": ba(s) - base_ba, \"changed\": int((pred != base_pred).sum()),\n                   \"fixed\": int(((pred != base_pred) & (np.asarray(LABELS)[pred] == truth)).sum()),\n                   \"broken\": int(((pred != base_pred) & (np.asarray(LABELS)[base_pred] == truth)).sum()),\n                   \"per_fold_gain\": [round(g, 4) for g in per_fold]}\n            report[\"overall\"].append(row)\n            print(json.dumps(row), flush=True)\n\npd.DataFrame({f\"{k[0]}|{k[1]}\": v for k, v in margins.items()}).assign(id=train[\"id\"].to_numpy()).to_parquet(\"/kaggle/working/margins.parquet\")\njson.dump(report, open(\"/kaggle/working/report.json\", \"w\"), indent=1)\nbest = max(report[\"overall\"], key=lambda r: r[\"gain\"])\nprint(\"best\", json.dumps(best), f\"({time.time() - start:.0f} s)\", flush=True)\nPY\nPYTHONPATH=src ~/Random/research-arm-kaggle/.venv/bin/python -c \"import ast;ast.parse(open('notebooks/pairs-check/run.py').read())\" && echo ok"
}
```

**Tool result**

```
ok
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && K=~/Random/research-arm-kaggle/.venv/bin/kaggle; $K kernels files nabilshah/nthl-pool-build 2>&1 | head; $K kernels files nabilshah/nthl-svc-pool-all-w025 2>&1 | head; $K datasets files nuridudak/ntl-130k-clean-passages 2>&1 | head"
}
```

**Tool result**

```
name               size  creationDate                           
-----------------  ----  -------------------------------------  
pool.parquet        852  1:11 pm, Saturday 24 October 2026 UTC  
pool_counts.csv     855  1:11 pm, Saturday 24 October 2026 UTC  
pool_summary.json   862  1:11 pm, Saturday 24 October 2026 UTC  
name          size  creationDate                           
------------  ----  -------------------------------------  
oof.parquet    848  3:21 pm, Saturday 24 October 2026 UTC  
summary.json   847  3:21 pm, Saturday 24 October 2026 UTC  
test.parquet   850  3:21 pm, Saturday 24 October 2026 UTC  
name                               size  creationDate                
-----------------------------  --------  --------------------------  
ntl_encoder_pool_122k.parquet  57919716  2026-10-23 09:11:28.758000  
ntl_merged_external.parquet    66199642  2026-10-23 09:11:28.681000  
official_folds.csv                80983  2026-10-23 09:11:27.485000  
sources.csv                        3143  2026-10-23 09:11:27.468000
```

## Claude (2026-10-25 10:06:30 UTC)

Code, tests and notebook are written. I'm committing them, then asking the main session for the OK to upload `nthl-src`.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && git add notebooks/pairs-check && git commit -qm \"exp(pairs): add the pairs-check notebook\n\nrefs #97\" && git push -qu origin exp/hard-pairs 2>&1 | tail -1; git log --oneline -1"
}
```

**Tool call: ToolSearch**

```json
{
 "query": "select:SendMessage",
 "max_results": 1
}
```

**Tool result**

```
remote: 
d65946e exp(pairs): add the pairs-check notebook
```

**Tool result**

```

```

**Tool call: SendMessage**

```json
{
 "to": "local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459",
 "summary": "Ask OK to upload nthl-src for pairs",
 "message": "hard-pairs: may I upload nthl-src now from branch exp/hard-pairs (d65946e, adds src/nthl/pairs.py on top of exp/gauss-heads)? The pairs-check kernel imports nthl.pairs, so a later upload from a branch without it would break that run until it has started. Reply OK and I upload and push nabilshah/nthl-pairs-check (CPU, internet off).",
 "type": "message",
 "recipient": "local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459",
 "recipient_kind": "session",
 "content": "hard-pairs: may I upload nthl-src now from branch…"
}
```

**Tool result**

```
{"success":true,"message":"Forwarded to Claude Desktop's session messaging: Message queued for session local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459 (\"AI Soc Kaggle Tournament\"): that session is holding it (a turn is in progress there, or its host or process is away) and runs it when that clears. Don't wait on it; check back with list_events if the outcome matters. (delivery: queued; message_id: af323e2b-cedf-4283-9af2-c3f576d190a2); not confirmed read — that session may hold it (usually a different permission mode) or refuse it, and this route reports nothing back"}
```

## Claude (2026-10-25 10:06:53 UTC)

Nothing has run on Kaggle yet. The code is ready and I'm waiting for the main session's OK to upload `nthl-src`.

**Done so far**
- My own worktree `~/Random/rak-pairs` on branch `exp/hard-pairs`, branched from `origin/exp/gauss-heads` and pushed. Last commit is `d65946e`.
- New issue for this run: [Nabzx/name-that-llm#97](https://github.com/Nabzx/name-that-llm/issues/97).
- [src/nthl/pairs.py](src/nthl/pairs.py) has three parts:
  - `top_two_is` picks out the rows where the base's top two are exactly the pair, in either order.
  - `rerank` either swaps the two scores or adds the specialist's margin to their gap at a weight.
  - `fit_pair` and `pair_margin` fit the binary four-view LinearSVC and read its decision.
- I wrote the tests first and committed them failing. They now pass, and so does the rest of the suite (71 passed, 3 skipped).
- [notebooks/pairs-check/run.py](notebooks/pairs-check/run.py) is the private CPU kernel `nabilshah/nthl-pairs-check`, internet off. Its base is row 11's out-of-fold predictions, and it fits each pair's specialist per frozen fold on the other folds only. It tries three versions of each specialist:
  - host rows only;
  - host rows plus our pool at weight 0.25;
  - host rows plus nuridudak's 122k pool at weight 0.25, leaving out the two sources we exclude for licence reasons.
- It will report:
  - each pair's accuracy, base against specialist, on all rows of the pair and on the rows where the base's top two are the pair;
  - overall balanced accuracy for the swap and for weights 0.25 to 4, with both pairs together and each alone;
  - how many rows change, how many are fixed and how many are broken, plus the gain per fold.

**Next**
Once I have the OK, I'll upload `nthl-src`, push the kernel and read the results. Then I'll add the log row, open the PR and say whether this goes in the submission. It goes in only if the overall out-of-fold gain is clearly above +0.003.

The best weight is chosen on the same out-of-fold rows it is scored on, so I'll check the per-fold gains before counting any win as real.

**Upload risk**
Another session's upload could replace `nthl-src` with a copy that has no `pairs.py`. If that lands after my upload but before my kernel starts, the kernel will fail. I've told the main session.

## Nabil (2026-10-25 10:07:05 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): report-draft = local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900 (working); video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (done, PR #95); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (idle); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
OK, with one step first: git fetch, then merge origin/exp/qwen27b and, if it exists on origin, origin/feat/encoders-122k into exp/hard-pairs, so the upload keeps qwen27b.py (and encoders122k.py) that other running kernels need. Run tests, then push_src.sh, then push nabilshah/nthl-pairs-check (CPU). Running kernels already copied their code, so only future pushes depend on the upload.
</cross-session-message>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && git fetch -q && git branch -r | grep -E \"qwen27b|encoders-122k\"; git merge --no-edit origin/exp/qwen27b 2>&1 | tail -2; git rev-parse -q --verify origin/feat/encoders-122k && git merge --no-edit origin/feat/encoders-122k 2>&1 | tail -2; ls src/nthl; PYTHONPATH=src ~/Random/research-arm-kaggle/.venv/bin/python -m pytest -q tests/ 2>&1 | tail -2"
}
```

**Tool result**

```
  origin/exp/qwen27b
  origin/feat/encoders-122k
 create mode 100644 src/nthl/qwen27b.py
 create mode 100644 tests/test_qwen27b.py
63a20ae3bd1f49c0aeaa92a660ed4f3f40f9ffac
 create mode 100644 src/nthl/encoders122k.py
 create mode 100644 tests/test_encoders122k.py
__init__.py
atlas.py
blend.py
encoder.py
encoders122k.py
folds.csv
folds.py
gauss.py
labels.py
lora24.py
metric.py
ngrams.py
pairs.py
pool.py
preds.py
qwen27b.py
svc.py
train_lora.py
ultrabert.py
witness.py
................                                                         [100%]
85 passed, 3 skipped in 8.36s
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && ln -sfn ~/Random/research-arm-kaggle/.venv .venv 2>/dev/null; git status --short | head; scripts/push_src.sh \"$(git rev-parse --short HEAD) hard-pairs + qwen27b + encoders122k\" 2>&1 | tail -3"
}
```

**Tool result**

```
?? .venv
  0%|          | 0.00/160k [00:00<?, ?B/s]100%|██████████| 160k/160k [00:00<00:00, 1.56MB/s]100%|██████████| 160k/160k [00:00<00:00, 283kB/s] 
Upload successful: nthl.tar (160KB)
Dataset version is being created. Please check progress at https://www.kaggle.com/datasets/nabilshah/nthl-src
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && for i in $(seq 1 20); do s=$(.venv/bin/kaggle datasets status nabilshah/nthl-src 2>&1); echo \"$s\"; echo \"$s\" | grep -qi ready && break; sleep 10; done; .venv/bin/kaggle kernels push -p notebooks/pairs-check 2>&1 | tail -2; git push -q 2>&1 | tail -1"
}
```

**Tool result**

```
ready
Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/nabilshah/nthl-pairs-check
```

## Claude (2026-10-25 10:07:44 UTC)

Uploaded and pushed. I'll watch the kernel in the background while I draft the PR.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && while true; do s=$(.venv/bin/kaggle kernels status nabilshah/nthl-pairs-check 2>&1); echo \"$(date +%H:%M) $s\"; echo \"$s\" | grep -qiE \"complete|error|cancel\" && break; sleep 60; done",
 "run_in_background": true
}
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && tail -5 docs/ai-sessions.md | cut -c1-300; tail -3 docs/experiments/LOG.md | cut -c1-200; grep -c \"\" docs/experiments/LOG.md"
}
```

**Tool result**

```
Command running in background with ID: brp0z332d. Output is being written to: /private/tmp/claude-501/-Users-nabilshah-Random/66776953-8515-46ab-ba28-5071c1f5cd44/tasks/brp0z332d.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /Users/nabilshah/Random/rak-pairs; directory changes made by the backgrounded command do not apply to subsequent commands.
```

**Tool result**

```
| 6 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #38 to #40 (hard pairs). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 7 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #48 (ablation table and report skeleton). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 8 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #28 (Zbigniew's checkpoint reader). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 9 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament (fork)", app id `local_b253a26e-df68-47e4-8da8-023d80ce5091` | 2026-10-24 | An accidental fork of session 1. Interrupted after 4 turns, it changed nothing. Nabil deleted it, so its log cannot be exported | Not exportable (deleted
| 10 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #28 (UltraBERT member). Transcript in session 1's `subagents` folder | 2026-10-24 | Wrote the UltraBERT reader on its own branch and opened a PR | Exported with session 1 |
| 15 | 2026-10-24 | #30 | **Probe 4.** Probe 3 plus Zbigniew's in-style Qwen3-8B LoRA at 10% (spine 80%, prvsiyan 10%). Notebook `nthl-submit` v6 | n/a | n/a | 0.627 | -0.004. His model was trained on
| 16 | 2026-10-24 | #36 | UltraBERT 5-fold encoder through our loader (`nthl.ultrabert`), host train. Notebook `nthl-spine-ultrabert` v1, 3 min | in-sample 0.966 | n/a | not submitted | A loader check
| 17 | 2026-10-25 | #28 | **Probe 5.** Probe 3 plus UltraBERT at a fixed 10% and Joseph Adamski's ATLAS at 10% on rows whose top two are under 0.3 apart; Zbigniew off. Notebook `nthl-submit` v8 | n/a 
23
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && grep -rn \"local_\" docs/ai-sessions.md | cut -c1-200 | tail -5; git log origin/exp/qwen27b --oneline -3 -- docs/ | head"
}
```

**Tool result**

```
docs/ai-sessions.md:15:3. The transcript id is the file name under `~/.claude/projects/`. The app id (`local_...`) is not the same thing.
docs/ai-sessions.md:27:| 1 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament", app id `local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459`, transcript `5d7d3493-861c-4044-bd86-4e023fd27b66` |
docs/ai-sessions.md:35:| 9 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament (fork)", app id `local_b253a26e-df68-47e4-8da8-023d80ce5091` | 2026-10-24 | An accidental fork of session 
e21a1ad Merge remote-tracking branch 'origin/feat/probe7-ours' into exp/centred-svc
ca2fdd8 docs(log): add probe 5
cc4eaba docs(ai-sessions): record that no other AI chat was used
```

## Claude (2026-10-25 10:07:53 UTC)

Kernel `nthl-pairs-check` v1 is running on Kaggle CPU. I'll be notified when it finishes, then read the report.

## Nabil (2026-10-25 10:22:57 UTC)

<task-notification>
<task-id>brp0z332d</task-id>
<tool-use-id>toolu_01Gu4f82qEzpydNymiWSJzpy</tool-use-id>
<output-file>/private/tmp/claude-501/-Users-nabilshah-Random/66776953-8515-46ab-ba28-5071c1f5cd44/tasks/brp0z332d.output</output-file>
<status>completed</status>
<summary>Background command "cd ~/Random/rak-pairs &amp;&amp; while true; do s=$(.venv/bin/kaggle kernels status nabilshah/nthl-pairs-check 2&gt;&amp;1); echo "$(date +%H:%M) $s"; echo "$s" | grep -qiE "complete|error|cancel" &amp;&amp; break; sleep 60; done" completed (exit code 0)</summary>
</task-notification>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && tail -2 /private/tmp/claude-501/-Users-nabilshah-Random/66776953-8515-46ab-ba28-5071c1f5cd44/tasks/brp0z332d.output; D=/private/tmp/claude-501/-Users-nabilshah-Random/66776953-8515-46ab-ba28-5071c1f5cd44/scratchpad/pairs-out; mkdir -p $D && .venv/bin/kaggle kernels output nabilshah/nthl-pairs-check -p $D 2>&1 | tail -1; ls $D; grep -v '\"overall\"' $D/*.log 2>/dev/null | head -0; python3 -c \"\nimport json;r=json.load(open('$D/report.json'))\nprint(r['base_ba'],r['temperature'])\nfor p in r['pairwise']: print(p)\nfor o in sorted(r['overall'],key=lambda o:-o['gain'])[:12]: print(o)\n\""
}
```

**Tool result**

```
[exited with code 0]
Kernel log downloaded to /private/tmp/claude-501/-Users-nabilshah-Random/66776953-8515-46ab-ba28-5071c1f5cd44/scratchpad/pairs-out/nthl-pairs-check.log 
margins.parquet
nthl-pairs-check.log
report.json
head: illegal line count -- 0
0.4837658463832961 4.466835921509633
{'pair': 'glm-5.3 vs glm-5.3-flash', 'subset': 'all pair rows', 'n': 300, 'base': 0.5366666666666666, 'host': 0.5166666666666666, 'ours': 0.5700000000000001, '122k': 0.57}
{'pair': 'glm-5.3 vs glm-5.3-flash', 'subset': 'pair rows with base top two = pair', 'n': 43, 'base': 0.575657894736842, 'host': 0.5603070175438596, 'ours': 0.5548245614035088, '122k': 0.5285087719298246}
{'pair': 'gemini-3.7-flash vs gemini-3.8-flash', 'subset': 'all pair rows', 'n': 300, 'base': 0.5433333333333333, 'host': 0.6066666666666667, 'ours': 0.5666666666666667, '122k': 0.5533333333333333}
{'pair': 'gemini-3.7-flash vs gemini-3.8-flash', 'subset': 'pair rows with base top two = pair', 'n': 180, 'base': 0.5464396284829721, 'host': 0.6037151702786377, 'ours': 0.5674922600619196, '122k': 0.5609907120743034}
{'pool': 'host', 'pairs': 'gemini-3.7-flash', 'weight': 'swap', 'ba': 0.4863584389758886, 'gain': 0.0025925925925924798, 'changed': 99, 'fixed': 46, 'broken': 39, 'per_fold_gain': [0.0037, -0.0056, 0.0148, -0.0019, 0.0019]}
{'pool': 'host', 'pairs': 'both', 'weight': 'swap', 'ba': 0.48598806860551824, 'gain': 0.0022222222222221255, 'changed': 133, 'fixed': 54, 'broken': 48, 'per_fold_gain': [0.0019, -0.0074, 0.0167, -0.0037, 0.0037]}
{'pool': 'host', 'pairs': 'glm-5.3', 'weight': 0.5, 'ba': 0.4845065871240368, 'gain': 0.0007407407407407085, 'changed': 7, 'fixed': 3, 'broken': 1, 'per_fold_gain': [0.0019, 0.0019, 0.0, 0.0, 0.0]}
{'pool': 'ours', 'pairs': 'gemini-3.7-flash', 'weight': 'swap', 'ba': 0.48450658712403677, 'gain': 0.000740740740740653, 'changed': 46, 'fixed': 20, 'broken': 18, 'per_fold_gain': [-0.0019, -0.0019, 0.0037, 0.0056, -0.0019]}
{'pool': 'host', 'pairs': 'both', 'weight': 0.25, 'ba': 0.4841362167536664, 'gain': 0.00037037037037029874, 'changed': 6, 'fixed': 2, 'broken': 1, 'per_fold_gain': [0.0019, 0.0, -0.0019, 0.0, 0.0019]}
{'pool': 'host', 'pairs': 'both', 'weight': 0.5, 'ba': 0.4841362167536664, 'gain': 0.00037037037037029874, 'changed': 13, 'fixed': 5, 'broken': 4, 'per_fold_gain': [0.0, 0.0037, 0.0, -0.0019, 0.0]}
{'pool': 'host', 'pairs': 'gemini-3.7-flash', 'weight': 0.25, 'ba': 0.4841362167536664, 'gain': 0.00037037037037029874, 'changed': 1, 'fixed': 1, 'broken': 0, 'per_fold_gain': [0.0, 0.0, 0.0, 0.0, 0.0019]}
{'pool': 'ours', 'pairs': 'both', 'weight': 'swap', 'ba': 0.4841362167536664, 'gain': 0.00037037037037029874, 'changed': 65, 'fixed': 24, 'broken': 23, 'per_fold_gain': [-0.0019, -0.0019, 0.0037, 0.0056, -0.0037]}
{'pool': '122k', 'pairs': 'gemini-3.7-flash', 'weight': 'swap', 'ba': 0.4841362167536664, 'gain': 0.00037037037037029874, 'changed': 53, 'fixed': 23, 'broken': 22, 'per_fold_gain': [0.0074, -0.0037, 0.0037, 0.0019, -0.0074]}
{'pool': '1
... (569 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "D=/private/tmp/claude-501/-Users-nabilshah-Random/66776953-8515-46ab-ba28-5071c1f5cd44/scratchpad/pairs-out; python3 -c \"\nimport json;L=json.load(open('$D/nthl-pairs-check.log'))\nfor e in L:\n  t=e['data']\n  if 'pair rows' in t or 'done' in t or 'top two' in t or 'best' in t: print(t.strip()[:300])\n\" 2>/dev/null || grep -oE '(pair rows|top two = pair on|best)[^\"]{0,200}' $D/nthl-pairs-check.log | head; python3 -c \"\nimport json;r=json.load(open('$D/report.json'))\nfor o in sorted(r['overall'],key=lambda o:o['gain'])[:3]: print(o['pool'],o['pairs'],o['weight'],round(o['gain'],4),o['changed'])\nprint(sum(o['gain']>0 for o in r['overall']), len(r['overall']))\""
}
```

**Tool result**

```
ours pair rows {'glm-5.3-flash': 6274, 'gemini-3.8-flash': 5861, 'glm-5.3': 5269, 'gemini-3.7-flash': 2666}
122k pair rows {'gemini-3.8-flash': 10178, 'glm-5.3-flash': 9476, 'glm-5.3': 8113, 'gemini-3.7-flash': 4921}
host glm-5.3 vs glm-5.3-flash done (13 s)
host gemini-3.7-flash vs gemini-3.8-flash done (23 s)
ours glm-5.3 vs glm-5.3-flash done (205 s)
ours gemini-3.7-flash vs gemini-3.8-flash done (332 s)
122k glm-5.3 vs glm-5.3-flash done (610 s)
122k gemini-3.7-flash vs gemini-3.8-flash done (842 s)
{"pair": "glm-5.3 vs glm-5.3-flash", "subset": "all pair rows", "n": 300, "base": 0.5366666666666666, "host": 0.5166666666666666, "ours": 0.5700000000000001, "122k": 0.57}
{"pair": "glm-5.3 vs glm-5.3-flash", "subset": "pair rows with base top two = pair", "n": 43, "base": 0.575657894736842, "host": 0.5603070175438596, "ours": 0.5548245614035088, "122k": 0.5285087719298246}
glm-5.3 vs glm-5.3-flash: base top two = pair on 71 rows, of which truly in the pair 43
{"pair": "gemini-3.7-flash vs gemini-3.8-flash", "subset": "all pair rows", "n": 300, "base": 0.5433333333333333, "host": 0.6066666666666667, "ours": 0.5666666666666667, "122k": 0.5533333333333333}
{"pair": "gemini-3.7-flash vs gemini-3.8-flash", "subset": "pair rows with base top two = pair", "n": 180, "base": 0.5464396284829721, "host": 0.6037151702786377, "ours": 0.5674922600619196, "122k": 0.5609907120743034}
gemini-3.7-flash vs gemini-3.8-flash: base top two = pair on 209 rows, of which truly in the pair 180
best {"pool": "host", "pairs": "gemini-3.7-flash", "weight": "swap", "ba": 0.4863584389758886, "gain": 0.0025925925925924798, "changed": 99, "fixed": 46, "broken": 39, "per_fold_gain": [0.0037, -0.0056, 0.0148, -0.0019, 0.0019]} (843 s)
host both 1.0 -0.0026 27
host gemini-3.7-flash 1.0 -0.0026 15
122k both 1.0 -0.0019 13
10 54
```

## Claude (2026-10-25 10:23:14 UTC)

Verdict: no gain worth taking. Writing the log row and the PR.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && grep -c \"ntl-130k\" CITATIONS.md; rm -f .venv; cat >> docs/experiments/LOG.md <<'EOF'\n| 18 | 2026-10-25 | #97 | Pairwise specialists on row 11's OOF: binary four-view LinearSVC per fold for glm-5.3 vs flash and gemini-3.7 vs 3.8-flash, host rows plus pool rows at 0.25 (no pool, ours, nuridudak's 122k), re-ranking only rows whose top two are the pair, by swap or margin at weight 0.25 to 4. Code `nthl-src` @ 2a6f1c4 (`nthl.pairs`), notebook `nthl-pairs-check` v1, CPU 14 min | best 0.4864 | +0.0026 | not submitted | Below the +0.003 bar and picked in-sample from 54 settings, 10 of which gain at all. Best is the host-only Gemini swap: 99 rows change, 46 fixed, 39 broken, folds +0.004, -0.006, +0.015, -0.002, +0.002. Pairwise accuracy on the base's top-two rows: GLM 0.58 base vs 0.53 to 0.56, Gemini 0.55 vs 0.56 to 0.60. Do not add. Confirms #38 and #39: no text-level lever. |\nEOF\ngit add docs/experiments/LOG.md && git commit -qm \"docs(log): add the pairs check\n\nrefs #97\" && git push -q && git rev-parse --short HEAD~1"
}
```

**Tool result**

```
0
7659799
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && grep -n \"nuridudak\" CITATIONS.md | head -3; git log --oneline -4"
}
```

**Tool result**

```
6f2a2ce docs(log): add the pairs check
7659799 Merge remote-tracking branch 'origin/feat/encoders-122k' into exp/hard-pairs
3932e84 Merge remote-tracking branch 'origin/exp/qwen27b' into exp/hard-pairs
63a20ae Merge remote-tracking branch 'origin/exp/qwen27b' into feat/encoders-122k
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && sed -i '' 's/`nthl-src` @ 2a6f1c4/`nthl-src` @ 7659799/' docs/experiments/LOG.md && grep -n \"^#\\|^|\" CITATIONS.md | head -20"
}
```

**Tool result**

```
1:# Citations
5:## Notebooks and discussions
27:## Datasets
53:## Pretrained models
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && sed -n 27,52p CITATIONS.md"
}
```

**Tool result**

```
## Datasets

- Name That LLM: extra GPT and Gemini generations (abdullah1afify/ntl-generations), CC BY 4.0. https://www.kaggle.com/datasets/abdullah1afify/ntl-generations. Read in the pool census (#19).
- NTL Vertex Gemini + proxy GPT generations (alperen5252525/ntl-vertex-gemini-proxy-gpt-generations), CC BY 4.0. https://www.kaggle.com/datasets/alperen5252525/ntl-vertex-gemini-proxy-gpt-generations. Read in the pool census (#19).
- NTL Curated Pool (86.5k, route-filtered) (ankiteshtiwari/ntl-curated-pool-routed), unknown. https://www.kaggle.com/datasets/ankiteshtiwari/ntl-curated-pool-routed. Read in the pool census (#19).
- ntl-kaggle-ai-gen (darragh1walsh/ntl-kaggle-ai-gen-run-1), CC0-1.0. https://www.kaggle.com/datasets/darragh1walsh/ntl-kaggle-ai-gen-run-1. Read in the pool census (#19).
- Name That LLM: free-proxy generations (dophuchung/ntl-proxy-generations), CC BY 4.0. https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations. Read in the pool census (#19).
- NTL Quality Selected Public SVC Pool (fleongg/ntl-quality-selected-public-svc-pool), other. https://www.kaggle.com/datasets/fleongg/ntl-quality-selected-public-svc-pool. Metadata read for #19.
- Name That LLM: Same-Prompt Passages (Free Quota) (haileycheng/name-that-llm-same-prompt-passages), CC BY 4.0. https://www.kaggle.com/datasets/haileycheng/name-that-llm-same-prompt-passages. Read in the pool census (#19).
- NTL: extra GPT passages via free Kaggle proxy (kaplinskiy/ntl-gpt-proxy-passages), CC BY 4.0. https://www.kaggle.com/datasets/kaplinskiy/ntl-gpt-proxy-passages. Read in the pool census (#19).
- New GPT Responses on 100 Independent Prompts (keshavatearth/ntl-gpt-new-prompts-v1), CC BY 4.0. https://www.kaggle.com/datasets/keshavatearth/ntl-gpt-new-prompts-v1. Read in the pool census (#19).
- NTL free-proxy GPT and Gemini generations (lucasboesen/ntl-free-proxy-generations), CC BY 4.0. https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations. Read in the pool census (#19).
- NTL synthetic generations v1 (13 labels) (lxdlxdlxd/ntl-synth-generations-v1), CC BY 4.0. https://www.kaggle.com/datasets/lxdlxdlxd/ntl-synth-generations-v1. Read in the pool census (#19).
- Name That LLM: OpenRouter + OpenAI Generations (maniceet/name-that-llm-openrouter-openai-generations), CC BY 4.0. https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations. Read in the pool census (#19).
- NTL own extra v1 (michaelsemenoff/ntl-own-extra-v1), CC BY 4.0. https://www.kaggle.com/datasets/michaelsemenoff/ntl-own-extra-v1. Read in the pool census (#19).
- NTL two-stage LoRA part two (nishantdahal/ntl-handwriting-blend), CC-BY-SA-4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend. Read in the pool census (#19).
- NTL two-stage LoRA checkpoints and public pool (nishantdahal/ntl-two-stage-lora), CC-BY-SA-4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora. Metadata read for #19.
- Name That LLM: Response Corpus (olegzholobov/
... (1542 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && python3 - <<'PY'\np=\"CITATIONS.md\";s=open(p).read()\nanchor=\"- Name That LLM: Response Corpus (olegzholobov\"\nline=\"- NTL 130k clean passages (nuridudak/ntl-130k-clean-passages), CC BY-SA 4.0. https://www.kaggle.com/datasets/nuridudak/ntl-130k-clean-passages. Its 122k encoder pool feeds the pair specialists in #97, with the olegzholobov and ankiteshtiwari rows left out.\\n\"\ns=s.replace(anchor,line+anchor,1);open(p,\"w\").write(s)\nPY\ngit add CITATIONS.md docs/experiments/LOG.md && git commit -qm \"docs(citations): cite nuridudak's 130k passages for the pairs check\n\nrefs #97\" && git push -q && git log --oneline -1"
}
```

**Tool result**

```
7adc441 docs(citations): cite nuridudak's 130k passages for the pairs check
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && gh pr list --state all --limit 6 --json number,baseRefName,headRefName,state | head -c 800"
}
```

**Tool result**

```
[{"baseRefName":"main","headRefName":"docs/readme","number":96,"state":"MERGED"},{"baseRefName":"main","headRefName":"docs/video-script-50","number":95,"state":"OPEN"},{"baseRefName":"main","headRefName":"docs/report-final-49","number":94,"state":"OPEN"},{"baseRefName":"feat/probe6-peer","headRefName":"feat/own-lora-member","number":93,"state":"OPEN"},{"baseRefName":"main","headRefName":"docs/ai-session-export","number":92,"state":"MERGED"},{"baseRefName":"main","headRefName":"feat/own-lora","number":91,"state":"OPEN"}]
```

## Claude (2026-10-25 10:23:48 UTC)

The result is in: no gain worth taking. Opening the PR against `exp/gauss-heads`, the branch this work started from.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/rak-pairs && gh pr create --base exp/gauss-heads --head exp/hard-pairs --title \"exp(pairs): pairwise specialists for the two hardest pairs, CV only (#97)\" --body \"$(cat <<'EOF'\n## What this does\n\n- `src/nthl/pairs.py`: a binary four-view LinearSVC for one pair (`fit_pair`, `pair_margin`), and `rerank`, which changes only rows whose top two are exactly that pair, by swap or by adding the margin at a weight. Tests in `tests/test_pairs.py`, written first.\n- `notebooks/pairs-check`: private CPU kernel `nabilshah/nthl-pairs-check` v1. Base is row 11's OOF (`nthl-svc-pool-all-w025`). Specialists fitted per frozen fold on the other folds, with three pools at weight 0.25: none, ours, nuridudak's 122k (CC BY-SA 4.0, the olegzholobov and ankiteshtiwari rows left out).\n- LOG row 18, and a citation for `nuridudak/ntl-130k-clean-passages`.\n- Merges `origin/exp/qwen27b` and `origin/feat/encoders-122k`, so the `nthl-src` upload from this branch (@ 7659799) kept `qwen27b.py` and `encoders122k.py`.\n\n## Why\n\nglm-5.3 vs glm-5.3-flash and gemini-3.7-flash vs 3.8-flash are our worst pairs. #38 and #39 found no lever by hand; this checks whether a dedicated model finds one.\n\n## Result\n\nBase OOF balanced accuracy 0.4838 (row 11 reproduced).\n\nPairwise balanced accuracy, base vs specialist:\n\n| Pair | Rows | Base | Host only | Our pool | 122k pool |\n|---|---|---|---|---|---|\n| GLM, all pair rows | 300 | 0.537 | 0.517 | 0.570 | 0.570 |\n| GLM, base top two = pair | 43 | 0.576 | 0.560 | 0.555 | 0.529 |\n| Gemini, all pair rows | 300 | 0.543 | 0.607 | 0.567 | 0.553 |\n| Gemini, base top two = pair | 180 | 0.546 | 0.604 | 0.567 | 0.561 |\n\nOverall, 54 settings (3 pools x both/GLM/Gemini x swap and weights 0.25, 0.5, 1, 2, 4). 10 gain at all. The best:\n\n| Setting | BA | Gain | Rows changed | Fixed | Broken | Per-fold gain |\n|---|---|---|---|---|---|---|\n| Host-only, Gemini, swap | 0.4864 | +0.0026 | 99 | 46 | 39 | +0.004, -0.006, +0.015, -0.002, +0.002 |\n| Host-only, both, swap | 0.4860 | +0.0022 | 133 | 54 | 48 | +0.002, -0.007, +0.017, -0.004, +0.004 |\n| Our pool, Gemini, swap | 0.4845 | +0.0007 | 46 | 20 | 18 | |\n\nThe worst lose 0.0026 (host-only, weight 1).\n\n**Not worth adding.** The best gain is below the +0.003 bar, chosen in-sample from 54 settings, and negative on two of five folds. The GLM specialist is no better than the base where it matters. The Gemini one is a little better on pair rows (0.60 vs 0.55), but fixes and breaks almost as many rows.\n\n## Checklist\n\n- [x] Tests first, all pass (85 passed, 3 skipped)\n- [x] Frozen folds, fitted on Kaggle only, seeds fixed\n- [x] LOG row\n- [x] Data cited\n- [x] No submission\n- [ ] AI session row: app id `local_`-prefixed side session of session 1, transcript `66776953-8515-46ab-ba28-5071c1f5cd44`. Left to the main session to avoid clashes in `docs/ai-sessions.md`\n\n## Test plan\n\n`PYTHONPATH=src pytest -q tests/`; kernel `nthl-pairs-check` v1 ran to completion in 14 min, `report.json` read back.\n\nrefs #97\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\n)\" 2>&1 | tail -1"
}
```

**Tool result**

```
https://github.com/Nabzx/name-that-llm/pull/98
```

**Tool call: Bash**

```json
{
 "command": "cat > \"/Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459/hard-pairs.md\" <<'EOF'\n# hard-pairs (PR #98, issue #97)\n\nPairwise specialists on row 11's OOF: no. Best +0.0026 (host-only Gemini swap, 99 rows changed, 46 fixed, 39 broken, folds +0.004 -0.006 +0.015 -0.002 +0.002), picked in-sample from 54 settings. GLM specialist no better than base on top-two rows. Do not add to the submission.\n\nnthl-src uploaded from exp/hard-pairs @ 7659799, which includes qwen27b.py, encoders122k.py and pairs.py.\nEOF"
}
```

**Tool result**

```
(Bash completed with no output)
```

## Claude (2026-10-25 10:24:17 UTC)

PR: [Nabzx/name-that-llm#98](https://github.com/Nabzx/name-that-llm/pull/98) (issue [#97](https://github.com/Nabzx/name-that-llm/issues/97)). **The pair specialists are not worth adding to the submission.** The best overall gain is +0.0026, under the +0.003 bar.

**Best setting:** host rows only, Gemini pair, swap.
- Out-of-fold balanced accuracy goes from 0.4838 to 0.4864.
- 99 rows change: 46 fixed, 39 broken.
- Two of five folds lose: per-fold gains are +0.004, -0.006, +0.015, -0.002 and +0.002.
- It is the best of 54 settings scored on the same rows, and only 10 of them gain at all. So the true gain is probably smaller.
- The worst settings lose 0.0026.

**Pair accuracy where re-ranking applies** (rows whose base top two are the pair):

| Pair | Rows | Base | Host only | Our pool | 122k pool |
|---|---|---|---|---|---|
| GLM | 43 | 0.58 | 0.56 | 0.55 | 0.53 |
| Gemini | 180 | 0.55 | 0.60 | 0.57 | 0.56 |

- **GLM:** no specialist beats the base where it matters. This matches what the earlier GLM notes (#38) found.
- **Gemini:** the specialist is a little better on these rows, but it fixes and breaks almost as many rows overall.

**Setup:**
- Kernel `nthl-pairs-check` v1 ran on CPU in 14 minutes and reproduced row 11's 0.4838 as its base.
- Before uploading I merged `origin/exp/qwen27b` and `origin/feat/encoders-122k`, as you asked. The 85 tests passed (3 skipped). `nthl-src` was then uploaded from `exp/hard-pairs` at 7659799, so it has `qwen27b.py`, `encoders122k.py` and `pairs.py`.

**In the PR:**
- `src/nthl/pairs.py`, with tests written first.
- The notebook.
- LOG row 18. Other branches may also use that number when they merge.
- A citation for nuridudak's 130k passages.

The PR targets `exp/gauss-heads`, the branch this work started from. I left the `docs/ai-sessions.md` row to you to avoid clashes; my transcript id is `66776953-8515-46ab-ba28-5071c1f5cd44`. A short hand-over note is in the shared notes folder as `hard-pairs.md`.
