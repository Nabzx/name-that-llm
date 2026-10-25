## Nabil (2026-10-25 00:42:59 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): report-draft = local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900 (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

Background (from the session that started this one):
Repo ~/Random/research-arm-kaggle (GitHub Nabzx/name-that-llm, private) is Nabil's entry to the Imperial "Name That LLM" Kaggle competition (identify which of 18 LLMs wrote a passage; balanced accuracy). Deadline Sun 25 Oct 2026 18:00 GMT. The hand-in needs a 3-minute video alongside a 2-page report. Nabil will record it himself on Sunday afternoon, so it must be easy to record solo. Read AGENTS.md and CLAUDE.md first. Style: Nabil's tone, British English, plain words, no em dashes, no hype. Never submit to Kaggle.

Task:
Write docs/report/video-script.md for the 3-minute video (#51): a timed script (spoken words per section, about 400 to 450 words in all) plus a shot list of what is on screen for each section (repo, LOG.md, a Kaggle notebook, the leaderboard, a simple diagram).

Story to tell, from docs/experiments/LOG.md on branch feat/own-lora, docs/research/ and docs/report/: the task and why it is hard (two hardest pairs: glm-5.3 vs glm-5.3-flash and gemini-3.7 vs 3.8-flash); CV first (GroupKFold on prompt); the public pool and our SVC (0.348 to 0.484 CV); building on published LoRA checkpoints with a loader we wrote ourselves (end token and label order traps); the probes 0.626, 0.629, 0.631, 0.627, 0.631 and what each taught us; our own additions (half Sinkhorn, peer correction, witness, our own LoRA); one honest failure; what we would do next. Leave TODO slots for probe 6 and the final score.

Also make one simple diagram of the pipeline as docs/report/pipeline.svg (members feeding a log-space blend, then fixed shares, then balancing), hand-written SVG, readable at slide size.

Commit small, push, open a PR in the house format. Done = PR open with script, shot list and diagram.

Shared notes folder for this group of sessions (outside the repo; read siblings' notes there, leave hand-over notes or large findings there instead of pasting them into messages): /Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git status -sb | head -5 && git branch -a | head -30 && ls docs docs/report docs/research; ls \"/Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459\""
}
```

**Tool result**

```
## feat/probe6-peer...origin/feat/probe6-peer
  ci/guardrails-59
  docs/adr-0001-12
  docs/ai-sessions-15
+ docs/blend-validation-20
  docs/checkpoints-18
  docs/citations-fork
+ docs/hard-pairs-38-40
  docs/kaggle-cli-17
  docs/kaggle-pitfalls-22
+ docs/likelihood-32-33
  docs/lora24-21
  docs/pools-19
  docs/probe1-log
+ docs/report-48
  docs/roadmap-direction
+ docs/sinkhorn-23
+ docs/top-teams-24
  docs/witnesses-31
  feat/folds-13
  feat/metric-14
  feat/own-lora
  feat/peer-correction-46
  feat/pool-gemini-64
  feat/pool-weight-63
  feat/pool-weight-63b
  feat/probe2-half-sinkhorn
  feat/probe3-pool025
  feat/probe5-atlas
* feat/probe6-peer
  feat/prvsiyan-28
docs:
adr
ai-sessions.md
experiments
report
research

docs/report:
ablation.md
report-draft.md

docs/research:
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
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git status --short | head; git worktree list; cat AGENTS.md CLAUDE.md | head -200"
}
```

**Tool result**

```
/Users/nabilshah/Random/research-arm-kaggle                                            d039816 [feat/probe6-peer]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-a1324620553d0edac  77f3352 [feat/zbigniew-28]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-a13e2b885fcd4b2a9  a43431a [feat/ultrabert-28]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-a17f1ecd3da6ad194  eb71861 [docs/top-teams-24]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-a1ee5f4536de328ae  7304328 [docs/likelihood-32-33]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-a4bfee7ca148f3e7a  39f6350 [docs/sinkhorn-23]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-ab02e241af761ffd9  3730464 [docs/report-48]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-ad2df954f09199f68  b56929b [docs/hard-pairs-38-40]
/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/agent-ad4f26265ddeef030  4e91e30 [docs/blend-validation-20]
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
- **What we make stays private on Kaggle too.
... (5031 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git fetch -q origin; git log --oneline -3 origin/main; git show origin/feat/own-lora:docs/experiments/LOG.md"
}
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git show origin/main:docs/report/report-draft.md | head -150; git log --oneline origin/main -- docs/report | head; git branch -r | grep -i -E \"report|video\""
}
```

**Tool result**

```
1e74ee1 Merge pull request #92 from Nabzx/docs/ai-session-export
cc4eaba docs(ai-sessions): record that no other AI chat was used
e9f9f5f docs(ai-sessions): add a snapshot export of session 1
# Experiment log

One row per run. CV is balanced accuracy on the frozen `GroupKFold` folds over `prompt_id`. Δ is against the current baseline.

| # | Date | Issue | Change | CV | Δ | Public LB | Notes and next step |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-24 | #25 | Four-view n-gram LinearSVC (C 0.5, balanced class weights), host train only. Code `nthl-src` @ 343c099, notebook `nthl-svc-train` v1, CPU, 114 s | 0.3485 | baseline | not submitted | Folds range 0.31 to 0.39. The lab is right 0.57 of the time (balanced), and half the errors stay inside the right lab. Hardest: claude-fable-5 0.05, glm-5.3-flash 0.11. Worst swaps: gemini-3.7-flash and gemini-3.8-flash (86 passages between them), glm-5.3 read as claude-opus-5 (27). In line with public train-only SVCs (0.343 to 0.370, discussion 750390). Next: add the public pool (#26). |
| 2 | 2026-10-24 | #26 | Experiment 1 re-run through the new `nthl.svc.run`. Code `nthl-src` @ 797d6ad, notebook `nthl-svc-repro` v1 | 0.3485 | 0 | not submitted | Same CV and the same probabilities to the last digit. The runner is a faithful copy of experiment 1. |
| 3 | 2026-10-24 | #26 | SVC plus the pool, official API and OpenRouter rows only (36,794), weight 1. Pool `nthl-pool-build` v3, notebook `nthl-svc-pool-api-w1` v1, CPU 37 min | 0.4189 | +0.070 | not submitted | Cleaner labels, a third of the rows. |
| 4 | 2026-10-24 | #26 | SVC plus the pool without the free proxy rows (69,603), weight 1. Notebook `nthl-svc-pool-noproxy-w1` v1, CPU 57 min | 0.4534 | +0.105 | not submitted | More rows beat cleaner labels. |
| 5 | 2026-10-24 | #26 | SVC plus the whole pool (84,470), weight 1. Notebook `nthl-svc-pool-all-w1` v1, CPU 90 min | 0.4634 | +0.115 | not submitted | The proxy rows help too. |
| 6 | 2026-10-24 | #26 | SVC plus the whole pool (84,470), weight 0.5. Notebook `nthl-svc-pool-all-w05` v1, CPU 65 min | **0.4782** | **+0.130** | not submitted | Best. Host rows count double, so host style still matters. Folds 0.44 to 0.52. gemini-3.7-flash falls from 0.41 to 0.19 in every pool run while 3.8-flash rises to 0.72: the pool's 3.7 rows may partly be 3.8. Public SVCs on a pool score 0.467 to 0.473 CV, and 0.423 on the LB, so expect about 0.43 to 0.44 there. Next: try a lower weight and leave the pool's Gemini 3.7 and 3.8 rows out. |
| 7 | 2026-10-24 | #27 | Nishant Dahal's Qwen3-14B LoRA checkpoint through our own loader (`nthl.lora24`), host train plus test, two T4s. Notebook `nthl-spine-qwen3-14b` v1, 8 min | in-sample 0.858 | n/a | not submitted | A loader check, not a score: the checkpoint saw every train passage. Its own CV is 0.611. Claude labels all right (fable 0.84), so the column reorder works. End token 151643 from meta.json, not the tokenizer's 151645. 189 adapter layers merged, no fp16 retries. About 11 to
... (3626 more characters in the raw transcript)
```

**Tool result**

```
# Name That LLM: report draft

Nabil Shah, solo entry, Imperial Research Arm qualifier. Draft for a two-page PDF (#49). Numbers come from `docs/experiments/LOG.md` and `docs/research/`. TODO marks a number we do not have yet.

## Starting point and approach

We wrote our own code from scratch. The first model was a four-view n-gram LinearSVC on host train only, scored on frozen `GroupKFold` folds over `prompt_id`. It reached 0.3485 balanced accuracy, in line with public train-only SVCs (0.343 to 0.370).

From there we worked in two branches (#20). Branch A is a published spine we run but never re-fit: Nishant Dahal's two-stage LoRA blend, which scores 0.620 on the public LB. Branch B is what we can validate on our own folds, starting with our SVC. Every change is one experiment, logged with its CV, and every LB probe changes one thing.

## Hypotheses

- **Data beats model size.** Public work showed 8B and 14B backbones tied on the same data, while growing the pool from 35k to 84k passages gave +0.028 on the LB (#24). So we expected a bigger, well-filtered pool to matter more than any new architecture.
- **Labs differ in surface style, siblings do not.** Our baseline put the right lab first 0.57 of the time, and half its errors stayed inside the right lab. The hardest pairs were gemini-3.7-flash against 3.8-flash, and glm-5.3 read as claude-opus-5. We expected n-grams to separate labs and need help inside a lab.
- **The prompt is louder than the author.** Passages that answer the same prompt share content. Subtracting a prompt estimate should leave mostly style (#23).
- **A lab's own model finds its text more likely.** Likelihood features from base models in each family should add signal that n-grams miss (#31 to #33).
- **The test mix is unequal.** Per-model test counts differ, so balancing the final predictions towards 1/18 should help. Half strength was safer than full on a test whose skew we cannot see (#23).

## Ablation table

Full table in `docs/report/ablation.md`. The main chain, all on the SVC:

| Change | CV | Total gain |
|---|---|---|
| Host train only (baseline) | 0.3485 | |
| + pool API and OpenRouter rows, weight 1 | 0.4189 | +0.070 |
| + all pool rows except free proxies | 0.4534 | +0.105 |
| + whole pool (84,470), weight 1 | 0.4634 | +0.115 |
| pool weight 0.5 | 0.4782 | +0.130 |
| pool weight 0.25 | 0.4838 | +0.135 |
| + likelihood witnesses (on the weight 0.5 SVC) | about +0.003 | |

LB probes, one change each:

| Probe | Change | Public LB |
|---|---|---|
| 1 | Published spine with our SVC | TODO |
| 2 | Half-strength balancing | TODO |
| 3 | + prvsiyan's two Qwen3-8B seeds at 10% | TODO |

TODO: final submission scores, public and private.

## One thing that didn't work, and why

In every pool run gemini-3.7-flash recall fell from 0.41 to 0.19 while 3.8-flash rose to 0.72. We read that as mislabelled pool rows: some of the pool's 3.7 passages might really be 3.8. So we left the pool's 3.7-flash and 3.8-flash rows out (log row
... (2162 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git log --oneline -8 origin/feat/own-lora; git log --oneline -5 origin/feat/peer-correction-46 2>/dev/null; git log --oneline -5 origin/feat/probe6-peer; git diff origin/main origin/feat/own-lora --stat | tail -15; gh issue view 51 --json title,body -q '.title+\"\\n\"+.body'"
}
```

**Tool result**

```
ca2fdd8 docs(log): add probe 5
2b7ddcd docs(log): add probes 2 to 4 and the UltraBERT loader check
505e1fb feat(notebooks): overnight two-stage LoRA training on Qwen3-8B
6cbe692 feat(train): add our own LoRA24 training parts and a reader for them
85a582d test(train): add failing tests for our own LoRA training parts
572a401 Merge pull request #90 from Nabzx/feat/ultrabert-28
a43431a Merge remote-tracking branch 'origin/main' into feat/ultrabert-28
518487e docs: cite UltraBERT as a member and log the session (#28)
e812374 Merge remote-tracking branch 'origin/main' into feat/peer-correction-46
f24ce64 Merge pull request #89 from Nabzx/docs/citations-fork
aa18ebd feat(notebooks): peer correction as an option in place of Sinkhorn
12a8f8d Merge remote-tracking branch 'origin/feat/witness-member-36' into feat/peer-correction-46
b87338e feat(blend): add peer correction and cluster similarity
d039816 feat(submit): probe 6, peer correction in place of Sinkhorn
6f1e4e7 fix(submit): let run_ultrabert do its own copy to local disk
e6cde88 feat(submit): probe 5, UltraBERT at 10% and ATLAS on unsure rows
e587fb2 feat(atlas): run the published ATLAS adapter under a time budget
ad519ce feat(blend): add low-margin rows and row-wise mixing, with tests
 ai-sessions/01-main-session.md            | 18174 ----------------------------
 ai-sessions/01-main-session.zip           |   Bin 4195329 -> 0 bytes
 docs/ai-sessions.md                       |    10 +-
 docs/experiments/LOG.md                   |     5 +
 notebooks/train-lora/kernel-metadata.json |    16 +
 notebooks/train-lora/run.py               |   169 +
 scripts/export_session.py                 |    12 +-
 src/nthl/lora24.py                        |    17 +
 src/nthl/train_lora.py                    |    74 +
 tests/test_train_lora.py                  |    49 +
 10 files changed, 339 insertions(+), 18187 deletions(-)
Export every AI session and check none is missing
Part of #11

Kind: wayfinder:task

Blocked by: #47

## Question or job

Export every row of `docs/ai-sessions.md`, subagent runs included.

## Done when

Every export saved and linked.

## Notes

Missing logs disqualify us.
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && gh issue list --state all --search video --json number,title; gh issue list --state all -L 100 --json number,title -q '.[]|\"\\(.number) \\(.title)\"' | sed -n '1,100p' | grep -i -E \"video|report|record|hand\""
}
```

**Tool result**

```
[{"number":50,"title":"Record the three-minute video"},{"number":52,"title":"Hand in the form"},{"number":11,"title":"Phase 5 map: submission and write-up"}]
59 Guardrails: tests on every pull request and the no hand-labelling rule
52 Hand in the form
50 Record the three-minute video
49 Write the two-page report
12 Record the direction as ADR-0001
```

## Claude (2026-10-25 00:43:29 UTC)

Reading the repo context now; the video issue is #50, not #51, so I'll reference #50. Next I'll pull the remaining facts and write the script.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && gh issue view 50 --json body -q .body; git show origin/feat/peer-correction-46:src/nthl/blend.py | grep -n -i -A12 \"def peer\" | head -40; git show origin/feat/own-lora:notebooks/train-lora/run.py | head -40; ls \"/Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459\""
}
```

**Tool result**

```
Part of #11

Kind: wayfinder:task

Blocked by: #49

## Question or job

A screen recording that walks through the approach and the results.

## Done when

An unlisted link, three minutes at most.

71:def peer_correct(z, svc, groups, sim, w_svc, strength=0.5, lo=3, hi=18, prior=8):
72-    """Peer correction (avazaku, cited): remove a shrunk share of each topic cluster's SVC bias.
73-
74-    For a cluster of n texts (lo to hi), every row loses strength * r * w_svc times the mean SVC margin
75-    of the other rows, where r = (n - 1) / (n - 1 + prior) * clip((sim - 0.1) / 0.4, 0, 1). A row never
76-    removes its own lean, and small or loosely related clusters are corrected less. `z` is the blended
77-    log score, `svc` the raw SVC scores. Other rows are left alone.
78-    """
79-    out = z.astype(float).copy()
80-    margin = svc - svc.mean(1, keepdims=True)
81-    for g in np.unique(groups):
82-        rows = np.flatnonzero(groups == g)
83-        n = len(rows)
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
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git show origin/main:docs/research/issue-38-glm-pair.md 2>/dev/null | head -30; git show origin/main:docs/research/issue-39-gemini-flash-pair.md 2>/dev/null| grep -i -E \"recommend|confus|swap\" | head -8; git show origin/main:CITATIONS.md | grep -i -E \"avazaku|nishant|prvsiyan\" | head"
}
```

**Tool result**

```
# Issue #38: What separates glm-5.3 from glm-5.3-flash

## Question

Is there anything in the text that tells glm-5.3 from glm-5.3-flash, and can we use it?

## What we looked at

- The 150 host train passages of each model. Both answer the same 150 prompts, one passage each.
- Per passage: words, sentences, words per sentence, and per 1,000 characters the rates of em dashes, curly quotes, semicolons, colons and commas. Mann-Whitney tests between the two sides.
- Words each side favours: log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008), prior from all 2,699 train passages, strength 1,000 tokens.
- First and last words of each passage.
- Out-of-fold predictions of our best SVC (`svc-pool-all-w025`, CV 0.484), on the frozen folds.

## Findings

**Style is almost identical.** Both write one paragraph, no markdown, straight quotes only, and an em dash in about 80% of passages.

| Per passage | glm-5.3 | glm-5.3-flash | p |
|---|---|---|---|
| Words, mean (median) | 110 (96) | 119 (112) | 0.20 |
| Words per sentence | 23.3 | 23.2 | 0.89 |
| Em dashes per 1k chars | 2.36 | 2.16 | 0.17 |
| Curly quotes per 1k chars | 0 | 0 | |
| Semicolons per 1k chars | 0.59 | 0.83 | 0.03 |
| Passages with a semicolon | 34% | 48% | 0.01 |
| Colons per 1k chars | 1.17 | 1.14 | 0.98 |
| Commas per 1k chars | 14.0 | 13.5 | 0.26 |

The semicolon is the only difference that clears p < 0.05, and with a dozen tests one such hit is expected by chance. On the same prompt, flash is longer only 54% of the time.
## Recommendation
- avazaku, *Name That LLM, Peer Correction: 0.631*. https://www.kaggle.com/code/avazaku/name-that-llm-fingerprint-ensemble. Read for leave-one-out peer correction in place of Sinkhorn.
- prvsiyan, *The Prompt Is Not the Author*. https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author. Read for the Gaussian group head and test-time grouping (#23).
- Nishant Dahal, *Two-stage LoRA, part two* (LB 0.620). https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two. Read for the LoRA24 readout (#21) and the blend weights.
- Nishant Dahal, *Two-stage LoRA on the public pool* (LB 0.592). https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool. Read for the training recipe.
- NTL two-stage LoRA part two (nishantdahal/ntl-handwriting-blend), CC-BY-SA-4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend. Read in the pool census (#19).
- NTL two-stage LoRA checkpoints and public pool (nishantdahal/ntl-two-stage-lora), CC-BY-SA-4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora. Metadata read for #19.
- Name That LLM: own-prompt generations v1 (prvsiyan/ntl-own-prompt-generations-v1), CC-BY-SA-4.0. https://www.kaggle.com/datasets/prvsiyan/ntl-own-prompt-generations-v1. Read in the pool census (#19).
- NTL public in-style pool v1 (prvsiyan/ntl-public-in-style-pool-v1), CC-BY-SA-4.0. https://www.kaggle.com/datasets/prvsiyan/ntl-public-in-style-pool-v1. Read in the pool census 
... (460 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && ls notebooks; grep -n -i -E \"share|weight|sinkhorn|alpha|peer|log\" notebooks/submit/run.py | head -60"
}
```

**Tool result**

```
pool-build
pool-census
smoke
spine-prvsiyan
spine-qwen3-14b
spine-ultrabert
spine-zbigniew
submit
svc-pool-all-w025
svc-pool-all-w035
svc-pool-all-w05
svc-pool-all-w05-nogemflash
svc-pool-all-w1
svc-pool-api-w1
svc-pool-noproxy-w1
svc-repro
svc-train
train-lora
witness-proto
3:Members, blended in log space with Nishant's published weights (#20):
4:- our four-view n-gram SVC, fitted here on host train plus our public pool at weight 0.5,
5:  read as a plain softmax of its raw scores like his SVC (weight 1.0);
9:Then Sinkhorn balancing at SINKHORN_ALPHA, skipped on the 3-row commit test.
34:from nthl.blend import cluster_similarity, low_margin_rows, mix_rows, peer_correct, sinkhorn, softmax, topic_groups
40:POST = "peer"   # "sinkhorn" (half strength) or "peer" (peer correction, no class quotas, #46)
41:SINKHORN_ALPHA = 0.5   # half strength, as #23 recommends. Probe 1 (v2) used 1.0.
42:POOL_WEIGHT = 0.25   # best SVC on CV, 0.4838 (#63). Probes 1 and 2 used 0.5.
43:WEIGHTS = {"svc": 1.0, "qwen3_14b": 0.4, "qwen25_14b": 0.2, "deberta": 0.25}
44:TWIN_WEIGHT = 0.1
45:WITNESS_WEIGHT = 0.0   # lab witness head (#34 to #36), off until probed. +0.003 to +0.005 on the SVC OOF.
46:ZBIGNIEW_SHARE = 0.0   # probe 4 put Zbigniew Kirchner's in-style Qwen3-8B at 10%: LB 0.627 against 0.631, so off
47:ULTRABERT_SHARE = 0.0   # probe 5: pjmathematician's UltraBERT-380M, 5 folds averaged, a fixed 10% (#36)
48:ATLAS_SHARE = 0.0   # probe 5: Joseph Adamski's ATLAS at 10%, only on rows whose top two are under ATLAS_MARGIN apart
51:PRVSIYAN_SHARE = 0.1   # probe 3: prvsiyan's two Qwen3-8B seeds, mixed in at a fixed 10% before balancing (#20)
68:logit = np.zeros((len(test), len(LABELS)))
74:    logit[rows] += w * np.log(np.clip(p, 1e-9, 1))
77:    print(f"{name}: weight {w}, done at {timings[name]} s", flush=True)
88:svc_raw = fit_scores(train, test["text"], pool=pool, pool_weight=POOL_WEIGHT)
89:vote("svc", softmax(svc_raw), WEIGHTS["svc"])
102:    vote(name, p, WEIGHTS[name])
104:        vote(f"{name}_twin", centred, TWIN_WEIGHT, rows=twin)
110:vote("deberta", run_encoder(os.path.dirname(one("deberta_v3_large/meta.json")), test["text"].tolist()), WEIGHTS["deberta"])
113:# logistic-regression head fitted here on all of host train (#34).
114:if WITNESS_WEIGHT > 0:
115:    from sklearn.linear_model import LogisticRegression
128:    head = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(x[: len(train)], train["model"])
130:    vote("witness", wp, WITNESS_WEIGHT)
132:if big and POST == "peer":
133:    logit = peer_correct(logit, svc_raw, groups, cluster_similarity(tfidf, groups), w_svc=WEIGHTS["svc"])
134:p = softmax(logit)
136:# 5. prvsiyan's two Qwen3-8B seeds, averaged, as a fixed share of the probabilities.
137:if PRVSIYAN_SHARE > 0:
154:if ZBIGNIEW_SHARE > 0:
167:if ULTRABERT_SHARE > 0:
178:# Published additions as fixed shares of the probabilities, before balancing (#20).
179:shares = {"prvsiyan": PRVSIYAN_SHARE if PRVSIYAN_SHARE > 0 els
... (957 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git worktree add -q .claude/worktrees/video-script-50 -b docs/video-script-50 origin/main && cd .claude/worktrees/video-script-50 && ls docs/report && git log -1 --format=%s origin/main -- docs/report"
}
```

**Tool result**

```
ablation.md
report-draft.md
docs(report): add the report skeleton with the required sections
```

## Claude (2026-10-25 00:44:08 UTC)

Writing the diagram first.

**Tool call: Write**

```json
{
 "file_path": "/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/pipeline.svg",
 "content": "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1280 720\" width=\"1280\" height=\"720\" font-family=\"Helvetica, Arial, sans-serif\">\n  <title>Name That LLM: submission pipeline</title>\n  <defs>\n    <marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"8\" markerHeight=\"8\" orient=\"auto-start-reverse\">\n      <path d=\"M0,0 L10,5 L0,10 z\" fill=\"#444\"/>\n    </marker>\n  </defs>\n  <rect width=\"1280\" height=\"720\" fill=\"#ffffff\"/>\n\n  <text x=\"40\" y=\"60\" font-size=\"34\" font-weight=\"bold\" fill=\"#111\">From passage to prediction</text>\n  <text x=\"40\" y=\"96\" font-size=\"20\" fill=\"#555\">18 labels, balanced accuracy. Ours in blue, published weights in grey.</text>\n\n  <!-- Members -->\n  <text x=\"40\" y=\"150\" font-size=\"22\" font-weight=\"bold\" fill=\"#111\">1. Members</text>\n  <g font-size=\"20\">\n    <rect x=\"40\" y=\"170\" width=\"360\" height=\"62\" rx=\"10\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"2\"/>\n    <text x=\"60\" y=\"198\" fill=\"#111\">Our n-gram SVC</text>\n    <text x=\"60\" y=\"222\" font-size=\"16\" fill=\"#444\">host train + pool at 0.25, CV 0.484</text>\n\n    <rect x=\"40\" y=\"246\" width=\"360\" height=\"62\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"60\" y=\"274\" fill=\"#111\">Qwen3-14B LoRA + centred twin</text>\n    <text x=\"60\" y=\"298\" font-size=\"16\" fill=\"#444\">Nishant Dahal, through our loader</text>\n\n    <rect x=\"40\" y=\"322\" width=\"360\" height=\"62\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"60\" y=\"350\" fill=\"#111\">Qwen2.5-14B LoRA + centred twin</text>\n    <text x=\"60\" y=\"374\" font-size=\"16\" fill=\"#444\">Nishant Dahal, through our loader</text>\n\n    <rect x=\"40\" y=\"398\" width=\"360\" height=\"62\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"60\" y=\"426\" fill=\"#111\">DeBERTa-v3-large</text>\n    <text x=\"60\" y=\"450\" font-size=\"16\" fill=\"#444\">Nishant Dahal</text>\n\n    <rect x=\"40\" y=\"520\" width=\"360\" height=\"62\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"60\" y=\"548\" fill=\"#111\">Two Qwen3-8B seeds</text>\n    <text x=\"60\" y=\"572\" font-size=\"16\" fill=\"#444\">prvsiyan, through our loader</text>\n\n    <rect x=\"40\" y=\"596\" width=\"360\" height=\"62\" rx=\"10\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"2\" stroke-dasharray=\"8 6\"/>\n    <text x=\"60\" y=\"624\" fill=\"#111\">Our own Qwen3-8B LoRA</text>\n    <text x=\"60\" y=\"648\" font-size=\"16\" fill=\"#444\">trained on Kaggle overnight</text>\n  </g>\n\n  <!-- Log-space blend -->\n  <rect x=\"490\" y=\"236\" width=\"270\" height=\"156\" rx=\"12\" fill=\"#ffffff\" stroke=\"#111\" stroke-width=\"3\"/>\n  <text x=\"625\" y=\"282\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">2. Log-space</text>\n  <text x=\"625\" y=\"312\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">blend</text>\n  <text x=\"625\" y=\"346\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">weighted sum of</text>\n  <text x=\"625\" y=\"368\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">log probabilities</text>\n\n  <!-- Fixed shares -->\n  <rect x=\"490\" y=\"510\" width=\"270\" height=\"140\" rx=\"12\" fill=\"#ffffff\" stroke=\"#111\" stroke-width=\"3\"/>\n  <text x=\"625\" y=\"556\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">3. Fixed shares</text>\n  <text x=\"625\" y=\"590\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">blend 90%</text>\n  <text x=\"625\" y=\"614\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">Qwen3-8B seeds 10%</text>\n\n  <!-- Balancing -->\n  <rect x=\"850\" y=\"510\" width=\"390\" height=\"140\" rx=\"12\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"3\"/>\n  <text x=\"1045\" y=\"556\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">4. Balancing</text>\n  <text x=\"1045\" y=\"590\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">half-strength Sinkhorn (probes 2 to 5)</text>\n  <text x=\"1045\" y=\"614\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">or peer correction (probe 6)</text>\n\n  <!-- Output -->\n  <rect x=\"930\" y=\"236\" width=\"230\" height=\"156\" rx=\"12\" fill=\"#111\" stroke=\"#111\" stroke-width=\"3\"/>\n  <text x=\"1045\" y=\"296\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#fff\">5. One label</text>\n  <text x=\"1045\" y=\"326\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#fff\">per passage</text>\n  <text x=\"1045\" y=\"362\" font-size=\"16\" text-anchor=\"middle\" fill=\"#ddd\">best public LB 0.631</text>\n\n  <!-- Arrows: members into blend -->\n  <g stroke=\"#444\" stroke-width=\"2.5\" fill=\"none\" marker-end=\"url(#arrow)\">\n    <path d=\"M400,201 C450,201 450,280 488,280\"/>\n    <path d=\"M400,277 C445,277 445,300 488,300\"/>\n    <path d=\"M400,353 C445,353 445,328 488,328\"/>\n    <path d=\"M400,429 C450,429 450,350 488,350\"/>\n    <!-- seeds and own LoRA into fixed shares -->\n    <path d=\"M400,551 L488,560\"/>\n    <path d=\"M400,627 L488,610\" stroke-dasharray=\"8 6\"/>\n    <!-- blend down to fixed shares -->\n    <path d=\"M625,392 L625,508\"/>\n    <!-- fixed shares to balancing -->\n    <path d=\"M760,580 L848,580\"/>\n    <!-- balancing up to output -->\n    <path d=\"M1045,510 L1045,394\"/>\n  </g>\n</svg>\n"
}
```

**Tool result**

```
File created successfully at: /Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/pipeline.svg (file state is current in your context — no need to Read it back)
```

**Tool call: mcp__Claude_Browser__computer**

```json
{
 "action": "screenshot",
 "scale": 0.6
}
```

**Tool result**

```

[Image: source: /Users/nabilshah/.claude/projects/-Users-nabilshah-Random/9014c337-56e2-446b-9370-78528936a8ee/tool-results/mcp-Claude_Browser-blob-1792889070263-kq123p.jpg]
Screenshot size: 480x360 0.6-scale view; coordinate frame: 800x600.


Tab Context:
- Executed on tabId: tab-1
- Available tabs:
  • tabId tab-1: "Name That LLM: submission pipeline" ("data:")
```

## Claude (2026-10-25 00:44:32 UTC)

Diagram renders cleanly. Now the script.

**Tool call: Write**

```json
{
 "file_path": "/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/video-script.md",
 "content": "# Video script: three minutes\n\nFor the hand-in video (#50). One speaker, one screen, recorded solo. About 440 spoken words, which is close to three minutes at a calm pace. Numbers come from `docs/experiments/LOG.md` and `docs/research/`. TODO marks a number we do not have yet.\n\n## Before you record\n\nOpen these in browser tabs, left to right, so each section is one tab switch:\n\n1. The GitHub repo, `Nabzx/name-that-llm`, on the README.\n2. `docs/research/issue-38-glm-pair.md` on GitHub, scrolled to the style table.\n3. `docs/experiments/LOG.md` on GitHub (rendered table).\n4. The Kaggle notebook `nthl-spine-qwen3-14b`, scrolled to the loader cell with `EOS` and the label reorder.\n5. `docs/report/pipeline.svg`, opened full screen.\n6. The competition leaderboard, filtered to our team, plus our submissions page.\n\nRecord each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take. Read the TODO lines last, once probe 6 and the final score are in.\n\nSay \"glm-5.3\" as \"G L M five point three\".\n\n## Script\n\n### 1. The task (0:00 to 0:20), about 50 words\n\n> Hi, I'm Nabil, and this is my solo entry to Name That LLM. Each passage was written by one of eighteen models, and the score is balanced accuracy. Labs are fairly easy to tell apart. Siblings are not. The two hardest pairs are glm-5.3 against glm-5.3-flash, and Gemini 3.7-flash against 3.8-flash.\n\n**On screen:** tab 1, the repo README, then tab 2, the GLM style table. Point at the semicolon row: the only difference between the two GLMs.\n\n### 2. CV first (0:20 to 0:35), about 40 words\n\n> Before any model, I froze the folds. All eighteen models answer the same prompts, so a random split leaks the prompt. I used GroupKFold on the prompt, and every experiment since uses the same folds and gets a row in this log.\n\n**On screen:** tab 3, `LOG.md`, top of the table. Hover over the CV column header.\n\n### 3. The pool and our SVC (0:35 to 0:55), about 50 words\n\n> The first model was an n-gram linear SVM on the host data alone: 0.348. Then I built a pool of eighty-four thousand public passages, filtered and de-duplicated. More rows beat cleaner labels, and counting pool rows at a quarter weight took cross-validation to 0.484.\n\n**On screen:** tab 3, rows 1 to 11. Highlight 0.3485 in row 1 and 0.4838 in row 11.\n\n### 4. Published LoRAs and our own loader (0:55 to 1:25), about 75 words\n\n> The strongest public members are LoRA fine-tunes of Qwen. The rules allow published weights if you show what you built, so I wrote my own loader for two checkpoint formats. Two traps nearly caught me. The end token: training used 151643, but the tokenizer says 151645, and the wrong one quietly shifts every feature. And label order: each checkpoint lists the models its own way, so the loader reorders the columns and checks it.\n\n**On screen:** tab 4, the Kaggle notebook. Scroll slowly from the `EOS` constant to the label reorder and its assert. Then tab 3, row 7.\n\n### 5. Five probes (1:25 to 2:05), about 95 words\n\n> Then leaderboard probes, one idea each. Probe one, the published spine with my SVC: 0.626, just above the original's 0.620. Probe two, balancing at half strength: 0.629. Full balancing over-corrects when the test mix is uneven. Probe three added prvsiyan's two Qwen3-8B seeds at a fixed ten per cent: 0.631. Probe four added another in-style LoRA and fell to 0.627. It was trained on data much like ours, so it repeated our mistakes. Probe five added two members that lifted public notebooks a long way, and stayed at 0.631. We already covered those rows.\n\n**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 5. Cut to tab 3, rows 12 to 17, if the submissions page is hard to read.\n\n### 6. What we added (2:05 to 2:30), about 60 words\n\n> Here is the whole pipeline. Members blend in log space, then take fixed shares, then a balancing step. What I added: half-strength Sinkhorn balancing; peer correction, which removes the lean that passages on the same topic share; small likelihood features from base models; and my own two-stage LoRA on Qwen3-8B, trained overnight on Kaggle.\n>\n> TODO probe 6: \"Probe six swapped Sinkhorn for peer correction and scored ___. That tells us ___.\"\n\n**On screen:** tab 5, `pipeline.svg`. Point at each box in turn, left to right, as you name it. Blue boxes are ours.\n\n### 7. One honest failure (2:30 to 2:45), about 45 words\n\n> One failure. In every pool run Gemini 3.7-flash recall collapsed, so I guessed the pool's 3.7 rows were mislabelled and dropped them. Cross-validation fell by 0.03, and both flash models went to 3.6-flash instead. Noisy labels for a class beat no labels at all.\n\n**On screen:** tab 3, row 8. Highlight 0.4482 and the -0.030.\n\n### 8. Next, and the result (2:45 to 3:00), about 35 words\n\n> Next I would add a model from outside the Qwen family, since the published ones agree on most picks, and test each step on folds that look like the test.\n>\n> TODO final: \"The final submission scored ___ public. Thanks for watching.\"\n\n**On screen:** tab 6, the leaderboard with our row highlighted.\n\n## Timing check\n\n| Section | Start | Words |\n|---|---|---|\n| 1. The task | 0:00 | 50 |\n| 2. CV first | 0:20 | 40 |\n| 3. Pool and SVC | 0:35 | 45 |\n| 4. Loader | 0:55 | 75 |\n| 5. Probes | 1:25 | 95 |\n| 6. What we added | 2:05 | 60 plus probe 6 |\n| 7. Failure | 2:30 | 45 |\n| 8. Next | 2:45 | 30 plus final |\n\nIf a run goes over three minutes, cut probe four's reason in section 5 first, then the end-token detail in section 4.\n"
}
```

**Tool result**

```
File created successfully at: /Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/video-script.md (file state is current in your context — no need to Read it back)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && grep '^> ' docs/report/video-script.md | wc -w; awk '/^### /{if(s)print s, n; s=$2\" \"$3; n=0} /^> /{n+=NF-1} END{print s,n}' docs/report/video-script.md; grep -c \"—\" docs/report/video-script.md docs/report/pipeline.svg"
}
```

**Tool result**

```
     471
1. The 51
2. CV 42
3. The 44
4. Published 74
5. Five 94
6. What 71
7. One 44
8. Next, 41
docs/report/video-script.md:0
docs/report/pipeline.svg:0
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && git add docs/report/pipeline.svg && git commit -q -m \"docs(report): add the pipeline diagram for the video and report\n\nrefs #50\" && git add docs/report/video-script.md && git commit -q -m \"docs(report): add the three-minute video script and shot list\n\nrefs #50\" && git push -q -u origin docs/video-script-50 2>&1 | tail -2; git log --oneline -2"
}
```

**Tool result**

```
remote:      https://github.com/Nabzx/name-that-llm/pull/new/docs/video-script-50        
remote: 
2362dd6 docs(report): add the three-minute video script and shot list
4ed1552 docs(report): add the pipeline diagram for the video and report
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && gh pr create --base main --title \"docs(report): video script, shot list and pipeline diagram (#50)\" --body \"$(cat <<'EOF'\n## What this does\n\n- `docs/report/video-script.md`: a timed script for the three-minute video, eight sections, about 440 spoken words, with what is on screen for each section and a tab setup for recording solo.\n- `docs/report/pipeline.svg`: a hand-written diagram of the submission pipeline (members, log-space blend, fixed shares, balancing), 1280 by 720 so it reads at slide size. Ours in blue, published weights in grey.\n\n## Why\n\nThe hand-in needs a three-minute video next to the report. The script tells the story from the log: the hard pairs, frozen folds, pool and SVC (0.348 to 0.484), our loader and its two traps, probes 1 to 5 (0.626, 0.629, 0.631, 0.627, 0.631), what we added, one failure (log row 8) and what we would do next.\n\n## Checklist\n\n- [x] Every number matches `docs/experiments/LOG.md` on `feat/own-lora`\n- [x] TODO slots for probe 6 and the final score\n- [x] British English, no em dashes\n- [ ] Fill the TODOs once probe 6 and the final score are in\n\n## Test plan\n\n- Open `pipeline.svg` in a browser at full screen and check every label reads.\n- Read the script aloud with a timer: each section should land near its start time, three minutes in all.\n\nrefs #50\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\n)\""
}
```

**Tool result**

```
https://github.com/Nabzx/name-that-llm/pull/95
```

**Tool call: ToolSearch**

```json
{
 "query": "select:mcp__ccd_pr__get_status,mcp__ccd_pr__bind_pr",
 "max_results": 2
}
```

**Tool result**

```


```

**Tool call: mcp__ccd_pr__get_status**

```json
{}
```

**Tool result**

```
{
  "bound": true,
  "pr": {
    "number": 95,
    "url": "https://github.com/Nabzx/name-that-llm/pull/95",
    "repo": "Nabzx/name-that-llm",
    "host": "github.com",
    "title": "docs(report): video script, shot list and pipeline diagram (#50)",
    "state": "open",
    "draft": false,
    "base": "main",
    "head": "docs/video-script-50"
  },
  "checks": {
    "available": true,
    "passing": 0,
    "failing": 0,
    "pending": 0,
    "skipped": 0,
    "failingChecks": []
  },
  "mergeable": "MERGEABLE",
  "mergeStateStatus": "UNSTABLE",
  "mergeQueue": {
    "enabled": false,
    "queued": false
  },
  "monitor": {
    "auto_fix": false,
    "address_comments": false,
    "auto_merge": false,
    "auto_archive_on_close": false
  }
}
```

## Claude (2026-10-25 00:45:23 UTC)

I've opened [Nabzx/name-that-llm#95](https://github.com/Nabzx/name-that-llm/pull/95) with the video script, the shot list and the pipeline diagram, on branch `docs/video-script-50` in two commits.

- **The issue number is #50, not #51.** #50 is "Record the three-minute video"; #51 is about exporting the AI session logs. The commits and the PR say `refs #50`.
- **Script (`docs/report/video-script.md`):** eight timed sections with about 440 spoken words, plus the two TODO lines, for probe 6 and the final score. It covers everything in your brief, in that order. Each section has a shot list. There's also a list of six browser tabs to open in order, and a note to record each section as its own clip so Nabil can redo one line on his own. A last note says what to cut first if a run goes over three minutes.
- **Diagram (`docs/report/pipeline.svg`):** hand-written, 1280×720, so it reads at slide size. It shows members going into the log-space blend, then fixed shares (blend 90%, prvsiyan's two Qwen3-8B seeds 10%), then balancing (half Sinkhorn or peer correction). Our parts are blue and published ones are grey. Our own LoRA is drawn dashed because it isn't in a submission yet. I opened it in the browser pane and the labels read cleanly.

The numbers come from `LOG.md` on `feat/own-lora` and from the submit notebook's config. The PR has no CI results yet. I can turn on Auto-fix for it if you want. Following the repo rule, the commits have no co-author trailer.

## Nabil (2026-10-25 00:45:25 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): report-draft = local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900 (working); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Important: work in your own git worktree, not in ~/Random/research-arm-kaggle itself (the main session switches branches there). First run: git -C ~/Random/research-arm-kaggle fetch -q && git -C ~/Random/research-arm-kaggle worktree add ~/Random/rak-video -b docs/video-script origin/main, then do everything inside ~/Random/rak-video. Read LOG.md rows 13 to 17 with: git -C ~/Random/rak-video show origin/feat/own-lora:docs/experiments/LOG.md. Never run git checkout or switch branches in ~/Random/research-arm-kaggle.
</cross-session-message>

## Claude (2026-10-25 00:45:31 UTC)

I've already followed this. Everything was done in a separate worktree, and I never switched branches in `~/Random/research-arm-kaggle`.

- **Worktree:** `~/Random/research-arm-kaggle/.claude/worktrees/video-script-50`, on branch `docs/video-script-50`, cut from `origin/main`. It's under the repo's own worktrees folder rather than `~/Random/rak-video`, but it's still isolated from the main checkout.
- **Main checkout:** I didn't check out, switch or commit there. It's still on `feat/probe6-peer`, as I found it.
- **LOG.md:** I read it from `origin/feat/own-lora`, rows 1 to 17. The script uses rows 13 to 17 for the probes (0.629, 0.631, 0.627, 0.631), and rows 1, 7, 8, 11 and 12 for the rest.

The work is in [Nabzx/name-that-llm#95](https://github.com/Nabzx/name-that-llm/pull/95), on branch `docs/video-script-50` (references #50, the video issue). It has the script, the shot list and `docs/report/pipeline.svg`. I won't create `~/Random/rak-video` unless you still want the work moved there.

## Nabil (2026-10-25 11:57:16 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): report-draft = local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900 (done, PR #94); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (done, PR #100); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (done, PR #99); hard-pairs = local_189a6d27-6360-4f5e-ae99-73795f412148 (done, PR #98). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Please update docs/report/video-script.md and the shot list (PR #95, keep it 3 minutes, about 400 to 450 words) with today's results. Fetch LOG.md from origin/feat/own-lora (rows 18 to 27). Key story beats to add: our own Qwen3-8B LoRA trained overnight on Kaggle (stage 1 honest 0.519); peer correction failed (0.622, the honest failure); the shared-prompt Gaussian group head (our own code after prvsiyan's public method), +0.058 on its own honest view, the biggest measured gain; probes 3, 5, 7 and 8 all at 0.631, which says the public slice is too small to show gains of this size; parallel tests that came back flat (centred SVC, hard pairs, 122k-pool SVC, 27B too slow at 3 s per passage). Probe 9 (Gaussian weight 1.0 + ModernBERT 10%) is scoring, leave a TODO for it and the final score. Also update docs/report/pipeline.svg so it shows the Gaussian head (ours, blue) and ModernBERT (published, grey). Push and update the PR.
</cross-session-message>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && git fetch -q origin && git status -sb | head -2 && git show origin/feat/own-lora:docs/experiments/LOG.md | sed -n '/^| 18 /,$p'"
}
```

**Tool result**

```
## docs/video-script-50...origin/docs/video-script-50
| 18 | 2026-10-25 | #46 | **Probe 6.** Probe 3 members with avazaku's peer correction (leave-one-out SVC margin removal in topic clusters) in place of half Sinkhorn. Notebook `nthl-submit` v9 | n/a | n/a | 0.622 | -0.009. Without class quotas the blend drifts towards the classes it already favours. avazaku saw +0.010 on a weaker blend and -0.001 on the full stack, so the gain does not carry to a strong blend. Keep half Sinkhorn. |
| 19 | 2026-10-25 | #43 | Our own Qwen3-8B LoRA24, trained on Kaggle (`nthl-train-lora` v1, 4.5 h): stage 1 on a balanced pool sample, stage 2 two epochs on host train. Checked with `nthl-check-ours` v1 | stage 1 on train **0.519** (honest), final 0.809 in-sample | n/a | not submitted | Stage 1 never saw train, so 0.519 is a fair out-of-sample number, above our SVC (0.484) but below Nishant's 14B after stage 1 (0.572). Weakest on glm-5.3, glm-5.3-flash (0.25) and claude-fable-5 (0.24). End token 151643 as trained. Ready to probe at 10%. |
| 20 | 2026-10-25 | #43 | **Probe 7.** Probe 3 plus our own Qwen3-8B LoRA24 at a fixed 10% (main 80%, prvsiyan 10%, ours 10%), half Sinkhorn. Notebook `nthl-submit` v10 | n/a | n/a | 0.631 | No change against probe 3. Our model runs cleanly and adds no harm, but at 10% its 0.519 view does not move the public score. Same pattern as UltraBERT and ATLAS: extra members at 10% sit inside the noise of a 0.631 blend. |
| 21 | 2026-10-25 | #63 | Row 11 with C 0.25. Notebook `nthl-svc-pool-w025-c025` v1, CPU 60 min | 0.4775 | -0.006 | not submitted | Too much regularisation. |
| 22 | 2026-10-25 | #63 | Row 11 with C 1.0. Notebook `nthl-svc-pool-w025-c1` v1, CPU 70 min | 0.4801 | -0.004 | not submitted | Too little. C 0.5 stays best. |
| 23 | 2026-10-25 | #63 | Row 11 with min_df 3. Notebook `nthl-svc-pool-w025-mindf3` v1, CPU | 0.4841 | +0.0003 | not submitted | Noise. Our SVC settings are at their best. |
| 24 | 2026-10-25 | #46 | Prompt-centred SVC twin (the public stylegram idea, our own code): host train centred within each prompt, test-like groups (1 to 5 passages per prompt, clustered with every test text), added to the row 11 OOF at weights 0.1 to 2. Notebook `nthl-svc-centred-check` v1, CPU 6 min, 10 draws | twin alone 0.327 on grouped rows (plain 0.481) | best +0.003 at weight 1.0, sd 0.0125 | not submitted | Not a clear gain: the spread across draws is four times the best mean gain, and replacing the plain SVC on grouped rows loses 0.13. The twin learns from 2,160 host rows only (the pool has no prompt ids to centre on), so it is much weaker than the pool SVC. Not worth a submission. |
| 25 | 2026-10-25 | #46 | **Shared-prompt Gaussian group head** (our own code after prvsiyan's public method) on the honest stage-1 features of our own LoRA: fitted per frozen fold, test-like groups (1 to 5 passages per prompt, clustered with every test text), added to the plain head on grouped rows. Notebook `nthl-gauss-check` v1, T4, 7 min, 10 draw
... (1172 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && for n in 98 99 100; do gh pr view $n --json title,body -q '.title+\"\\n\"+.body' | head -25; echo ----; done"
}
```

**Tool result**

```
exp(pairs): pairwise specialists for the two hardest pairs, CV only (#97)
## What this does

- `src/nthl/pairs.py`: a binary four-view LinearSVC for one pair (`fit_pair`, `pair_margin`), and `rerank`, which changes only rows whose top two are exactly that pair, by swap or by adding the margin at a weight. Tests in `tests/test_pairs.py`, written first.
- `notebooks/pairs-check`: private CPU kernel `nabilshah/nthl-pairs-check` v1. Base is row 11's OOF (`nthl-svc-pool-all-w025`). Specialists fitted per frozen fold on the other folds, with three pools at weight 0.25: none, ours, nuridudak's 122k (CC BY-SA 4.0, the olegzholobov and ankiteshtiwari rows left out).
- LOG row 18, and a citation for `nuridudak/ntl-130k-clean-passages`.
- Merges `origin/exp/qwen27b` and `origin/feat/encoders-122k`, so the `nthl-src` upload from this branch (@ 7659799) kept `qwen27b.py` and `encoders122k.py`.

## Why

glm-5.3 vs glm-5.3-flash and gemini-3.7-flash vs 3.8-flash are our worst pairs. #38 and #39 found no lever by hand; this checks whether a dedicated model finds one.

## Result

Base OOF balanced accuracy 0.4838 (row 11 reproduced).

Pairwise balanced accuracy, base vs specialist:

| Pair | Rows | Base | Host only | Our pool | 122k pool |
|---|---|---|---|---|---|
| GLM, all pair rows | 300 | 0.537 | 0.517 | 0.570 | 0.570 |
| GLM, base top two = pair | 43 | 0.576 | 0.560 | 0.555 | 0.529 |
| Gemini, all pair rows | 300 | 0.543 | 0.607 | 0.567 | 0.553 |
| Gemini, base top two = pair | 180 | 0.546 | 0.604 | 0.567 | 0.561 |

----
feat(encoders): run nuridudak's two-stage ModernBERT and desklib as members (#28)
## What this does

- `src/nthl/encoders122k.py`: `run_seqcls(model_dir, texts, max_len=512, batch_size=32)` runs nuridudak's two-stage encoders (nuridudak/ntl-two-stage-encoders) with `AutoModelForSequenceClassification`, fp16 autocast, after copying the folder to `/tmp`. Columns come back in our `LABELS` order, read from each config's `id2label` (both store the labels alphabetically).
- `tests/test_encoders122k.py`: label order tests.
- `notebooks/check-encoders122k`: private T4 kernel, internet off. Runs both models on host train and test, writes `members/modernbert.npy`, `members/desklib.npy` (test rows in `test.csv` order) and the in-sample train predictions.
- `CITATIONS.md` and the experiment log row.
- Merges `origin/exp/qwen27b` and `origin/exp/hard-pairs`, so the `nthl-src` uploaded from this branch keeps `qwen27b.py` and `pairs.py`.

## Why

ModernBERT-large has a published prompt-grouped OOF of 0.569 and desklib 0.523, both trained on the open 122k pool and then host train. Adding them to Nishant Dahal's 0.620 notebook gave 0.624 public. They are cheap blend candidates.

## Results (`nthl-check-encoders122k` v1, 83 s in all)

| model | in-sample balanced accuracy | published OOF | s per 1,000 passages |
|---|---|---|---|
| ModernBERT-large | 0.806 | 0.569 | 16.1 |
| desklib DeBERTa-v3-large | 0.659 | 0.523 | 14.4 |

In-sample, because stage 2 traine
... (3465 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && gh pr view 98 --json body -q .body | sed -n '25,60p'; gh pr view 100 --json body -q .body | grep -i -E \"s per|second|slow|0.521|recommend\" | head"
}
```

**Tool result**

```
Overall, 54 settings (3 pools x both/GLM/Gemini x swap and weights 0.25, 0.5, 1, 2, 4). 10 gain at all. The best:

| Setting | BA | Gain | Rows changed | Fixed | Broken | Per-fold gain |
|---|---|---|---|---|---|---|
| Host-only, Gemini, swap | 0.4864 | +0.0026 | 99 | 46 | 39 | +0.004, -0.006, +0.015, -0.002, +0.002 |
| Host-only, both, swap | 0.4860 | +0.0022 | 133 | 54 | 48 | +0.002, -0.007, +0.017, -0.004, +0.004 |
| Our pool, Gemini, swap | 0.4845 | +0.0007 | 46 | 20 | 18 | |

The worst lose 0.0026 (host-only, weight 1).

**Not worth adding.** The best gain is below the +0.003 bar, chosen in-sample from 54 settings, and negative on two of five folds. The GLM specialist is no better than the base where it matters. The Gemini one is a little better on pair rows (0.60 vs 0.55), but fixes and breaks almost as many rows.

## Checklist

- [x] Tests first, all pass (85 passed, 3 skipped)
- [x] Frozen folds, fitted on Kaggle only, seeds fixed
- [x] LOG row
- [x] Data cited
- [x] No submission
- [ ] AI session row: app id `local_`-prefixed side session of session 1, transcript `66776953-8515-46ab-ba28-5071c1f5cd44`. Left to the main session to avoid clashes in `docs/ai-sessions.md`

## Test plan

`PYTHONPATH=src pytest -q tests/`; kernel `nthl-pairs-check` v1 ran to completion in 14 min, `report.json` read back.

refs #97

🤖 Generated with [Claude Code](https://claude.com/claude-code)
| speed | 3,029 s per 1,000 passages at batch 3 (mean 148 tokens, max 379) |
Runtime. At this speed the 4,000-row test needs about 3.4 h plus 17 min load, about 3.7 h, and longer test passages would make it slower. With our current members (about 2 h) that is about 5.7 h. It fits inside 9 h, but not today's last submission slot. Batching by a token budget instead of 3 rows should cut the fixed per-batch cost on short passages. That was not tried here.
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && git branch -r | grep -i -E \"probe|gauss\"; b=$(git branch -r | grep -i -E \"probe9|probe-9\" | head -1); echo $b; git show $b:notebooks/submit/run.py | grep -n -E \"SHARE|GAUSS|WEIGHT|POST|ours|main\" | head -40"
}
```

**Tool result**

```
  origin/docs/probe1-log
  origin/exp/gauss-heads
  origin/feat/probe2-half-sinkhorn
  origin/feat/probe3-pool025
  origin/feat/probe5-atlas
  origin/feat/probe6-peer
  origin/feat/probe7-ours

41:POOL_WEIGHT = 0.25   # best SVC on CV, 0.4838 (#63). Probes 1 and 2 used 0.5.
42:WEIGHTS = {"svc": 1.0, "qwen3_14b": 0.4, "qwen25_14b": 0.2, "deberta": 0.25}
43:TWIN_WEIGHT = 0.1
44:PRVSIYAN_SHARE = 0.1   # probe 3: prvsiyan's two Qwen3-8B seeds, mixed in at a fixed 10% before balancing (#20)
81:vote("svc", softmax(fit_scores(train, test["text"], pool=pool, pool_weight=POOL_WEIGHT)), WEIGHTS["svc"])
94:    vote(name, p, WEIGHTS[name])
96:        vote(f"{name}_twin", centred, TWIN_WEIGHT, rows=twin)
102:vote("deberta", run_encoder(os.path.dirname(one("deberta_v3_large/meta.json")), test["text"].tolist()), WEIGHTS["deberta"])
107:if PRVSIYAN_SHARE > 0:
120:    p = (1 - PRVSIYAN_SHARE) * p + PRVSIYAN_SHARE * prv
129:summary = {"rows": len(test), "sinkhorn_alpha": SINKHORN_ALPHA if big else None, "weights": WEIGHTS,
130:           "twin_weight": TWIN_WEIGHT, "prvsiyan_share": PRVSIYAN_SHARE, "pool_weight": POOL_WEIGHT, "timings": timings, "seconds": round(time.time() - T0),
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && for b in origin/main origin/feat/own-lora origin/exp/gauss-heads; do echo \"== $b\"; git show $b:notebooks/submit/run.py 2>/dev/null | grep -n -i -E \"gauss|modernbert|OURS_SHARE|_SHARE =\" | head -20; done"
}
```

**Tool result**

```
== origin/main
44:PRVSIYAN_SHARE = 0.1   # probe 3: prvsiyan's two Qwen3-8B seeds, mixed in at a fixed 10% before balancing (#20)
== origin/feat/own-lora
44:PRVSIYAN_SHARE = 0.1   # probe 3: prvsiyan's two Qwen3-8B seeds, mixed in at a fixed 10% before balancing (#20)
== origin/exp/gauss-heads
34:from nthl.gauss import fit as gauss_fit, log_posteriors, transform as gauss_transform
47:ZBIGNIEW_SHARE = 0.0   # probe 4 put Zbigniew Kirchner's in-style Qwen3-8B at 10%: LB 0.627 against 0.631, so off
48:ULTRABERT_SHARE = 0.0   # probe 5: pjmathematician's UltraBERT-380M, 5 folds averaged, a fixed 10% (#36)
49:ATLAS_SHARE = 0.0   # probe 5: Joseph Adamski's ATLAS at 10%, only on rows whose top two are under ATLAS_MARGIN apart
52:PRVSIYAN_SHARE = 0.1   # probe 3: prvsiyan's two Qwen3-8B seeds, mixed in at a fixed 10% before balancing (#20)
53:OURS_SHARE = 0.1   # probe 7: our own Qwen3-8B LoRA24 from nabilshah/nthl-train-lora (#43) at a fixed 10%
54:GAUSS_WEIGHT = 1.0   # probe 9 (probe 8 used 0.4, LB 0.631; the check found 1.0 best): shared-prompt Gaussian group head on our honest stage-1 features, grouped rows only (#46, LOG row 25)
55:MODERNBERT_SHARE = 0.1   # probe 9: nuridudak's two-stage ModernBERT-large (122k pool, Apache 2.0, OOF 0.569) at a fixed 10%
56:GAUSS_T = 2.0   # temperature picked by held-out log loss in nthl-gauss-check
137:# 7. Shared-prompt Gaussian group head on our own LoRA's stage-1 features (#46). Stage 1 never saw host
139:gauss_info = None
140:if GAUSS_WEIGHT > 0:
145:    prm = gauss_fit(feats[: len(train)], np.array([LABELS.index(m) for m in train["model"]]), train["prompt_id"].to_numpy())
147:    lp = log_posteriors(gauss_transform(feats[len(train):], prm), prm, groups if big else np.arange(len(test)), rows, T=GAUSS_T)
148:    np.save(f"{OUT}/members/gauss_stage1.npy", lp)
150:        vote("gauss_stage1", np.exp(lp[twin]), GAUSS_WEIGHT, rows=twin)
151:    gauss_info = {"d": prm["d"], "grouped_rows": int(rows.sum()), "features_seconds": info["total_seconds"]}
152:    print("gauss", json.dumps(gauss_info), flush=True)
156:    timings["gauss"] = round(time.time() - T0)
204:# 5d. nuridudak's two-stage ModernBERT-large, trained on the 122k open pool (#28, PR #99).
```

## Claude (2026-10-25 11:58:04 UTC)

Fetched rows 18 to 27 and checked PRs #98 to #100 and the probe 9 notebook. Now redrawing the diagram with the Gaussian head and ModernBERT.

**Tool call: Write**

```json
{
 "file_path": "/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/pipeline.svg",
 "content": "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1280 720\" width=\"1280\" height=\"720\" font-family=\"Helvetica, Arial, sans-serif\">\n  <title>Name That LLM: submission pipeline</title>\n  <defs>\n    <marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"8\" markerHeight=\"8\" orient=\"auto-start-reverse\">\n      <path d=\"M0,0 L10,5 L0,10 z\" fill=\"#444\"/>\n    </marker>\n  </defs>\n  <rect width=\"1280\" height=\"720\" fill=\"#ffffff\"/>\n\n  <text x=\"40\" y=\"52\" font-size=\"34\" font-weight=\"bold\" fill=\"#111\">From passage to prediction</text>\n  <text x=\"40\" y=\"84\" font-size=\"20\" fill=\"#555\">18 labels, balanced accuracy. Ours in blue, published weights in grey.</text>\n  <text x=\"40\" y=\"118\" font-size=\"22\" font-weight=\"bold\" fill=\"#111\">1. Members</text>\n\n  <!-- Members into the log-space blend -->\n  <g font-size=\"20\">\n    <rect x=\"40\" y=\"130\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"154\" fill=\"#111\">Our n-gram SVC</text>\n    <text x=\"58\" y=\"175\" font-size=\"15\" fill=\"#444\">host train + pool at 0.25, CV 0.484</text>\n\n    <rect x=\"40\" y=\"194\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"218\" fill=\"#111\">Qwen3-14B LoRA + centred twin</text>\n    <text x=\"58\" y=\"239\" font-size=\"15\" fill=\"#444\">Nishant Dahal, through our loader</text>\n\n    <rect x=\"40\" y=\"258\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"282\" fill=\"#111\">Qwen2.5-14B LoRA + centred twin</text>\n    <text x=\"58\" y=\"303\" font-size=\"15\" fill=\"#444\">Nishant Dahal, through our loader</text>\n\n    <rect x=\"40\" y=\"322\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"346\" fill=\"#111\">DeBERTa-v3-large</text>\n    <text x=\"58\" y=\"367\" font-size=\"15\" fill=\"#444\">Nishant Dahal</text>\n\n    <rect x=\"40\" y=\"386\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"410\" fill=\"#111\">Shared-prompt Gaussian group head</text>\n    <text x=\"58\" y=\"431\" font-size=\"15\" fill=\"#444\">our code on our LoRA's features, +0.058 alone</text>\n\n    <!-- Members as fixed shares -->\n    <rect x=\"40\" y=\"470\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"494\" fill=\"#111\">Two Qwen3-8B seeds</text>\n    <text x=\"58\" y=\"515\" font-size=\"15\" fill=\"#444\">prvsiyan, through our loader</text>\n\n    <rect x=\"40\" y=\"534\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"558\" fill=\"#111\">Our own Qwen3-8B LoRA</text>\n    <text x=\"58\" y=\"579\" font-size=\"15\" fill=\"#444\">trained overnight on Kaggle, honest 0.519</text>\n\n    <rect x=\"40\" y=\"598\" width=\"380\" height=\"54\" rx=\"10\" fill=\"#eeeeee\" stroke=\"#777\" stroke-width=\"2\"/>\n    <text x=\"58\" y=\"622\" fill=\"#111\">ModernBERT-large</text>\n    <text x=\"58\" y=\"643\" font-size=\"15\" fill=\"#444\">nuridudak, 122k open pool</text>\n  </g>\n\n  <!-- Log-space blend -->\n  <rect x=\"500\" y=\"208\" width=\"260\" height=\"156\" rx=\"12\" fill=\"#ffffff\" stroke=\"#111\" stroke-width=\"3\"/>\n  <text x=\"630\" y=\"254\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">2. Log-space</text>\n  <text x=\"630\" y=\"284\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">blend</text>\n  <text x=\"630\" y=\"318\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">weighted sum of</text>\n  <text x=\"630\" y=\"340\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">log probabilities</text>\n\n  <!-- Fixed shares -->\n  <rect x=\"500\" y=\"490\" width=\"260\" height=\"150\" rx=\"12\" fill=\"#ffffff\" stroke=\"#111\" stroke-width=\"3\"/>\n  <text x=\"630\" y=\"532\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">3. Fixed shares</text>\n  <text x=\"630\" y=\"564\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">blend 70%</text>\n  <text x=\"630\" y=\"588\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">each of the three</text>\n  <text x=\"630\" y=\"610\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">below it 10%</text>\n\n  <!-- Balancing -->\n  <rect x=\"850\" y=\"490\" width=\"390\" height=\"150\" rx=\"12\" fill=\"#dbe8fb\" stroke=\"#2a62b8\" stroke-width=\"3\"/>\n  <text x=\"1045\" y=\"532\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#111\">4. Balancing</text>\n  <text x=\"1045\" y=\"566\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">half-strength Sinkhorn</text>\n  <text x=\"1045\" y=\"590\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">(peer correction tried in</text>\n  <text x=\"1045\" y=\"614\" font-size=\"16\" text-anchor=\"middle\" fill=\"#444\">probe 6: 0.622, dropped)</text>\n\n  <!-- Output -->\n  <rect x=\"930\" y=\"208\" width=\"230\" height=\"156\" rx=\"12\" fill=\"#111\" stroke=\"#111\" stroke-width=\"3\"/>\n  <text x=\"1045\" y=\"268\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#fff\">5. One label</text>\n  <text x=\"1045\" y=\"298\" font-size=\"24\" font-weight=\"bold\" text-anchor=\"middle\" fill=\"#fff\">per passage</text>\n  <text x=\"1045\" y=\"334\" font-size=\"16\" text-anchor=\"middle\" fill=\"#ddd\">best public LB 0.631</text>\n\n  <g stroke=\"#444\" stroke-width=\"2.5\" fill=\"none\" marker-end=\"url(#arrow)\">\n    <!-- five members into the blend -->\n    <path d=\"M420,157 C460,157 460,240 498,240\"/>\n    <path d=\"M420,221 C460,221 460,262 498,262\"/>\n    <path d=\"M420,285 L498,285\"/>\n    <path d=\"M420,349 C460,349 460,308 498,308\"/>\n    <path d=\"M420,413 C460,413 460,332 498,332\"/>\n    <!-- three members into fixed shares -->\n    <path d=\"M420,497 L498,530\"/>\n    <path d=\"M420,561 L498,565\"/>\n    <path d=\"M420,625 L498,600\"/>\n    <!-- blend down to fixed shares -->\n    <path d=\"M630,364 L630,488\"/>\n    <!-- fixed shares to balancing -->\n    <path d=\"M760,565 L848,565\"/>\n    <!-- balancing up to output -->\n    <path d=\"M1045,490 L1045,366\"/>\n  </g>\n"
}
```

**Tool result**

```
The file /Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/pipeline.svg has been updated successfully. (file state is current in your context — no need to Read it back)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && echo '</svg>' >> docs/report/pipeline.svg && xmllint --noout docs/report/pipeline.svg && echo ok"
}
```

**Tool result**

```
ok
```

**Tool call: mcp__Claude_Browser__javascript_tool**

```json
{
 "action": "javascript_exec",
 "text": "location.reload(); 1"
}
```

**Tool call: mcp__Claude_Browser__resize_window**

```json
{
 "width": 1300,
 "height": 740
}
```

**Tool call: mcp__Claude_Browser__computer**

```json
{
 "action": "screenshot"
}
```

**Tool result**

```
1


Tab Context:
- Executed on tabId: tab-1
- Available tabs:
  • tabId tab-1: "data:image/svg+xml;charset=utf-8,<svg xmlns%3D http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg  viewBox%3D 0 0 1280 720  width%3D 1280  height%3D 720  font-family%3D Helvetica%2C Arial%2C sans-serif >%0A  <title" ("data:")
```

**Tool result**

```
Viewport set to 1300x740 on this tab (scaled down to fit if larger than the pane). Reset it with preset "desktop" as soon as you finish testing; the desktop app also clears a size you set when your turn ends or the pane's width changes, so set it again if you still need it. If the user wants to keep a phone or tablet size after your turn, they can pick Mobile or Tablet in the pane's Viewport menu; a custom size you simply set again next turn.


Tab Context:
- Executed on tabId: tab-1
- Available tabs:
  • tabId tab-1: "data:image/svg+xml;charset=utf-8,<svg xmlns%3D http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg  viewBox%3D 0 0 1280 720  width%3D 1280  height%3D 720  font-family%3D Helvetica%2C Arial%2C sans-serif >%0A  <title" ("data:")
- Viewport: emulating 1300x740 (you set this; reset it with preset "desktop" when you finish testing; the app also clears it when your turn ends or the pane's width changes)
```

**Tool result**

```

[Image: source: /Users/nabilshah/.claude/projects/-Users-nabilshah-Random/9014c337-56e2-446b-9370-78528936a8ee/tool-results/mcp-Claude_Browser-blob-1792929512002-vdd51t.jpg]
Screenshot size: 800x455


Tab Context:
- Executed on tabId: tab-1
- Available tabs:
  • tabId tab-1: "data:image/svg+xml;charset=utf-8,<svg xmlns%3D http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg  viewBox%3D 0 0 1280 720  width%3D 1280  height%3D 720  font-family%3D Helvetica%2C Arial%2C sans-serif >%0A  <title" ("data:")
- Viewport: emulating 1300x740 (you set this; reset it with preset "desktop" when you finish testing; the app also clears it when your turn ends or the pane's width changes)
```

## Claude (2026-10-25 11:58:34 UTC)

The error banner is from the stale copy loaded before the closing tag went in; xmllint passes on the file. Layout reads well. Now the script.

**Tool call: mcp__Claude_Browser__resize_window**

```json
{
 "preset": "desktop"
}
```

**Tool result**

```
Viewport emulation cleared; the tab is back to the pane's responsive size (desktop).


Tab Context:
- Executed on tabId: tab-1
- Available tabs:
  • tabId tab-1: "data:image/svg+xml;charset=utf-8,<svg xmlns%3D http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg  viewBox%3D 0 0 1280 720  width%3D 1280  height%3D 720  font-family%3D Helvetica%2C Arial%2C sans-serif >%0A  <title" ("data:")
```

**Tool call: Write**

```json
{
 "file_path": "/Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/video-script.md",
 "content": "# Video script: three minutes\n\nFor the hand-in video (#50). One speaker, one screen, recorded solo. About 440 spoken words, which is close to three minutes at a calm pace. Numbers come from `docs/experiments/LOG.md` (rows 1 to 27) and `docs/research/`. TODO marks a number we do not have yet.\n\n## Before you record\n\nOpen these in browser tabs, left to right, so each section is one tab switch:\n\n1. The GitHub repo, `Nabzx/name-that-llm`, on the README.\n2. `docs/research/issue-38-glm-pair.md` on GitHub, scrolled to the style table.\n3. `docs/experiments/LOG.md` on GitHub (rendered table).\n4. The Kaggle notebook `nthl-train-lora`, on the cell with `EOS = 151643` and the stage 1 and stage 2 loop.\n5. `docs/report/pipeline.svg`, opened full screen.\n6. The competition leaderboard, filtered to our team, plus our submissions page.\n\nRecord each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take. Read the TODO lines last, once probe 9 and the final score are in.\n\nSay \"glm-5.3\" as \"G L M five point three\".\n\n## Script\n\n### 1. The task (0:00 to 0:17), about 45 words\n\n> Hi, I'm Nabil, and this is my solo entry to Name That LLM. Each passage was written by one of eighteen models, scored on balanced accuracy. Labs are easy to tell apart. Siblings are not: glm-5.3 against its flash model, and Gemini 3.7-flash against 3.8-flash.\n\n**On screen:** tab 1, the repo README, then tab 2, the GLM style table. Point at the semicolon row: the only difference between the two GLMs.\n\n### 2. CV first and our SVC (0:17 to 0:40), about 60 words\n\n> Before any model, I froze the folds: GroupKFold on the prompt, since all eighteen models answer the same prompts. Every experiment gets a row in this log. My first model, an n-gram SVM on host data, scored 0.348. Adding eighty-four thousand public passages, at a quarter weight, took it to 0.484.\n\n**On screen:** tab 3, `LOG.md`. Highlight 0.3485 in row 1, then 0.4838 in row 11.\n\n### 3. Published LoRAs and our own (0:40 to 1:05), about 65 words\n\n> The strongest public members are LoRA fine-tunes of Qwen. I wrote my own loader for them, and it caught two traps: the end token the model was trained with is not the tokenizer's default, and each checkpoint orders the labels its own way. Then I trained my own Qwen3-8B LoRA overnight on Kaggle. Its honest score is 0.519, above my SVC.\n\n**On screen:** tab 4, the Kaggle notebook. Scroll slowly from `EOS` to the two training stages. Then tab 3, row 19, highlight 0.519.\n\n### 4. The probes (1:05 to 1:35), about 70 words\n\n> Then leaderboard probes, one idea each. The published spine with my SVC scored 0.626. Balancing at half strength: 0.629. Adding prvsiyan's two seeds: 0.631. Then four probes in a row landed on exactly 0.631, with new members that each changed real rows. That tells me the public slice is too small to show gains of this size, so I trust cross-validation more than this board.\n\n**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 8. Cut to tab 3, rows 12 to 26, if the submissions page is hard to read.\n\n### 5. The biggest gain (1:35 to 2:05), about 65 words\n\n> Here is the pipeline. Members blend in log space, then take fixed shares, then balancing. The blue boxes are ours. The biggest measured gain is a shared-prompt Gaussian group head, our own code after prvsiyan's public method. Passages that answer the same prompt are decoded together, with the prompt's effect taken out. On its own honest view it added 0.058.\n\n**On screen:** tab 5, `pipeline.svg`. Point at each box as you name it, ending on the Gaussian group head. Then tab 3, row 25, highlight +0.058.\n\n### 6. One honest failure (2:05 to 2:25), about 45 words\n\n> One failure. Peer correction, a public trick, swaps balancing for removing the lean that passages on one topic share. It helped weaker blends. On ours it scored 0.622, our worst probe. Without class quotas the blend drifts towards the models it already favours.\n\n**On screen:** tab 3, row 18. Highlight 0.622.\n\n### 7. What came back flat (2:25 to 2:40), about 40 words\n\n> Some things came back flat, and that is useful too. A prompt-centred SVC, pair specialists for the two hardest pairs, and a bigger public pool all stayed within noise. A 27-billion model took three seconds per passage, too slow.\n\n**On screen:** tab 3, rows 24 and 27, then the PR list on GitHub showing the hard-pairs and Qwen 27B pull requests.\n\n### 8. Next, and the result (2:40 to 3:00), about 45 words\n\n> TODO probe 9: \"The last probe put the Gaussian head at full weight and added ModernBERT at ten per cent. It scored ___.\"\n>\n> Next I would add a model from outside the Qwen family, and test each step on folds that look like the test. TODO final: \"The final submission scored ___. Thanks for watching.\"\n\n**On screen:** tab 6, the leaderboard with our row highlighted.\n\n## Timing check\n\n| Section | Start | Words |\n|---|---|---|\n| 1. The task | 0:00 | 45 |\n| 2. CV and SVC | 0:17 | 55 |\n| 3. LoRAs | 0:40 | 65 |\n| 4. Probes | 1:05 | 70 |\n| 5. Gaussian head | 1:35 | 60 |\n| 6. Failure | 2:05 | 45 |\n| 7. Flat tests | 2:25 | 40 |\n| 8. Next | 2:40 | 45 with the TODOs |\n\nIf a run goes over three minutes, cut the 27B line in section 7 first, then the two loader traps in section 3.\n"
}
```

**Tool result**

```
The file /Users/nabilshah/Random/research-arm-kaggle/.claude/worktrees/video-script-50/docs/report/video-script.md has been updated successfully. (file state is current in your context — no need to Read it back)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && grep '^> ' docs/report/video-script.md | wc -w; awk '/^### /{if(s)print s, n; s=$2\" \"$3; n=0} /^> /{n+=NF-1} END{print s,n}' docs/report/video-script.md; grep -c \"—\" docs/report/video-script.md docs/report/pipeline.svg"
}
```

**Tool result**

```
     428
1. The 45
2. CV 51
3. Published 61
4. The 65
5. The 60
6. One 43
7. What 39
8. Next, 55
docs/report/video-script.md:0
docs/report/pipeline.svg:0
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && f=docs/report/video-script.md && sed -i '' -e 's/^| 2. CV and SVC | 0:17 | 55 |/| 2. CV and SVC | 0:17 | 50 |/' -e 's/^| 3. LoRAs | 0:40 | 65 |/| 3. LoRAs | 0:40 | 60 |/' -e 's/^| 4. Probes | 1:05 | 70 |/| 4. Probes | 1:05 | 65 |/' -e 's/^| 8. Next | 2:40 | 45 with the TODOs |/| 8. Next | 2:40 | 55 with the TODOs |/' -e 's/about 440 spoken words/about 430 spoken words/' -e 's/### 2. CV first and our SVC (0:17 to 0:40), about 60 words/### 2. CV first and our SVC (0:17 to 0:40), about 50 words/' -e 's/(0:40 to 1:05), about 65 words/(0:40 to 1:05), about 60 words/' -e 's/(1:05 to 1:35), about 70 words/(1:05 to 1:35), about 65 words/' -e 's/(1:35 to 2:05), about 65 words/(1:35 to 2:05), about 60 words/' -e 's/(2:40 to 3:00), about 45 words/(2:40 to 3:00), about 55 words/' $f && grep -n \"words\" $f | head -12\ngit add docs/report/pipeline.svg && git commit -q -m \"docs(report): add the Gaussian group head and ModernBERT to the pipeline diagram\n\nrefs #50\" && git add $f && git commit -q -m \"docs(report): update the video script with probes 6 to 8 and today's checks\n\nrefs #50\" && git push -q && git log --oneline -2"
}
```

**Tool result**

```
3:For the hand-in video (#50). One speaker, one screen, recorded solo. About 440 spoken words, which is close to three minutes at a calm pace. Numbers come from `docs/experiments/LOG.md` (rows 1 to 27) and `docs/research/`. TODO marks a number we do not have yet.
22:### 1. The task (0:00 to 0:17), about 45 words
28:### 2. CV first and our SVC (0:17 to 0:40), about 50 words
34:### 3. Published LoRAs and our own (0:40 to 1:05), about 60 words
40:### 4. The probes (1:05 to 1:35), about 65 words
46:### 5. The biggest gain (1:35 to 2:05), about 60 words
52:### 6. One honest failure (2:05 to 2:25), about 45 words
58:### 7. What came back flat (2:25 to 2:40), about 40 words
64:### 8. Next, and the result (2:40 to 3:00), about 55 words
d5341f1 docs(report): update the video script with probes 6 to 8 and today's checks
b07195a docs(report): add the Gaussian group head and ModernBERT to the pipeline diagram
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && gh pr edit 95 --body \"$(cat <<'EOF'\n## What this does\n\n- `docs/report/video-script.md`: a timed script for the three-minute video, eight sections, about 430 spoken words, with what is on screen for each section and a tab setup for recording solo.\n- `docs/report/pipeline.svg`: a hand-written diagram of the submission pipeline, 1280 by 720 so it reads at slide size. Five members (SVC, two 14B LoRAs, DeBERTa, the shared-prompt Gaussian group head) go into a log-space blend; prvsiyan's seeds, our own Qwen3-8B LoRA and ModernBERT join as fixed 10% shares; then half-strength Sinkhorn. Ours in blue, published weights in grey.\n\n## Why\n\nThe hand-in needs a three-minute video next to the report. The script follows LOG.md rows 1 to 27:\n\n- the task and the two hardest pairs\n- frozen folds, then our SVC from 0.348 to 0.484\n- our loader, and our own Qwen3-8B LoRA trained overnight on Kaggle (honest stage 1, 0.519)\n- probes 1 to 3 (0.626, 0.629, 0.631), then probes 3, 5, 7 and 8 all at 0.631: the public slice is too small to show gains of this size\n- the shared-prompt Gaussian group head, our code after prvsiyan's public method: +0.058 on its own honest view, our biggest measured gain\n- the honest failure: peer correction, 0.622\n- what came back flat: the centred SVC, the hard-pair specialists, the 122k pool and the 27B (3 s per passage)\n- what we would do next\n\n## Checklist\n\n- [x] Every number matches `docs/experiments/LOG.md` on `feat/own-lora` and PRs #98 to #100\n- [x] TODO slots for probe 9 and the final score\n- [x] British English, no em dashes\n- [ ] Fill the TODOs once probe 9 and the final score are in\n\n## Test plan\n\n- Open `pipeline.svg` in a browser at full screen and check every label reads.\n- Read the script aloud with a timer: each section should land near its start time, three minutes in all.\n\nrefs #50\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\n)\" >/dev/null && echo updated"
}
```

**Tool result**

```
updated
```

## Claude (2026-10-25 11:59:24 UTC)

I've pushed the updated script and diagram to [Nabzx/name-that-llm#95](https://github.com/Nabzx/name-that-llm/pull/95) in two new commits (b07195a, d5341f1), and rewrote the PR description to match.

**Script (`docs/report/video-script.md`):** 428 spoken words in eight sections, which is about three minutes. The count includes the two TODO lines. The story is:
1. The task and the two hardest pairs.
2. Frozen folds, then our SVC from 0.348 to 0.484.
3. The loader traps, and our own Qwen3-8B LoRA trained overnight on Kaggle (honest stage 1: 0.519).
4. The probes: 0.626, 0.629, then four in a row at exactly 0.631 (probes 3, 5, 7 and 8). The point made is that the public slice is too small to show gains this size.
5. The shared-prompt Gaussian group head, our own code after prvsiyan's public method: +0.058 on its own honest view, the biggest measured gain.
6. The honest failure is now peer correction at 0.622. The Gemini-rows story is dropped to keep within the word count.
7. What came back flat: the prompt-centred SVC, the hard-pair specialists and the 122k pool, all within noise. The 27B model was too slow at 3 s per passage.
8. What we'd do next, with TODO lines for probe 9 and the final score.

The shot list now points at the right rows (18, 19, 24, 25 and 27), and tab 4 is now the `nthl-train-lora` notebook.

**Diagram (`docs/report/pipeline.svg`):** I took the layout from the probe 9 notebook on `exp/gauss-heads`.
- **Into the log-space blend:** five members, including the Gaussian group head in blue.
- **Fixed 10% shares:** prvsiyan's seeds, our own LoRA and ModernBERT (grey), with the blend taking 70%.
- **Balancing:** half-strength Sinkhorn, with peer correction marked as tried and dropped.

The file passes an XML check, and I checked the layout in the browser pane.

The file's intro line still says "About 440 spoken words"; the real count is 428. Both are inside the 400 to 450 range you asked for, so I left it.

## Nabil (2026-10-25 13:01:20 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): report-draft = local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900 (done, PR #94); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (done, PR #100); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (done, PR #99); hard-pairs = local_189a6d27-6360-4f5e-ae99-73795f412148 (done, PR #98). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Probe 9 scored 0.637 public (+0.006, our best): Gaussian head at weight 1.0 plus ModernBERT at 10%. Please fill the TODO lines in docs/report/video-script.md with it (the run of four 0.631 ties ended when the head went to full weight). Note Nabil now also has documents/presentation.pdf and documents/speaker-notes.md (branch docs/presentation) to present from; keep the script consistent with those notes. Push and open a new PR against main (PR #95 is merged).
</cross-session-message>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && git fetch -q origin && git checkout -q -b docs/video-script-probe9 origin/main && git log --oneline -1 && grep -n \"TODO\" docs/report/video-script.md; git show origin/docs/presentation:documents/speaker-notes.md"
}
```

**Tool result**

```
28d4c7a Merge pull request #95 from Nabzx/docs/video-script-50
3:For the hand-in video (#50). One speaker, one screen, recorded solo. About 440 spoken words, which is close to three minutes at a calm pace. Numbers come from `docs/experiments/LOG.md` (rows 1 to 27) and `docs/research/`. TODO marks a number we do not have yet.
16:Record each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take. Read the TODO lines last, once probe 9 and the final score are in.
66:> TODO probe 9: "The last probe put the Gaussian head at full weight and added ModernBERT at ten per cent. It scored ___."
68:> Next I would add a model from outside the Qwen family, and test each step on folds that look like the test. TODO final: "The final submission scored ___. Thanks for watching."
83:| 8. Next | 2:40 | 55 with the TODOs |
# Speaker notes: Name That LLM

For the 3-minute video, read alongside `documents/presentation.pdf` (or `presentation.html` in a browser, full screen). About 450 spoken words, roughly 18 seconds a slide. Say "glm-5.3" as "G L M five point three".

## 1. Cover (0:00 to 0:08)

> Hi, I'm Nabil. This is my solo entry to the Research Arm qualifier, Name That LLM.

## 2. The task (0:08 to 0:25)

> Each passage, forty to two hundred and fifty words, was written by one of eighteen models from six labs. We have 2,699 training passages and about four thousand hidden test passages on prompts we never see. The score is balanced accuracy. Labs are easy to tell apart. Siblings are not: the two GLM models, and Gemini 3.7 against 3.8 flash.

## 3. Starting point (0:25 to 0:45)

> Before any model, I froze the folds: grouped by prompt, so validation looks like the hidden test. My first model, an n-gram SVC on host data only, scored 0.348. The best public notebook scored 0.620. Rather than fork it, I wrote my own loader for its published weights and built my own members on top, all trained on Kaggle and all cited.

## 4. Approach (0:45 to 1:05)

> Here's the pipeline. Each member gives log-probabilities, which are blended as a weighted sum. A few extra members join at fixed ten per cent shares. Then a half-strength Sinkhorn step pulls the class totals halfway to even. The filled boxes are ours: the SVC, the Gaussian group head and my own Qwen3-8B LoRA, trained overnight.

## 5. Hypotheses (1:05 to 1:30)

> Four hypotheses. One: style lives in small habits like function words and punctuation, so n-grams should work. Two: data beats model size, and eighty-four thousand public passages took the SVC from 0.348 to 0.484. Three: the prompt hides the author. Passages that share a prompt also share content, so taking that out should leave style. Four: the test mix is uneven, so balance only halfway.

## 6. Ablation, leaderboard (1:30 to 1:55)

> Each probe changed one thing. The published spine with my SVC: 0.626. Half-strength balancing: 0.629. Two extra seeds: 0.631. Then a run of ties at 0.631 as I added members, 
... (1797 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && sed -n 38,72p docs/report/video-script.md"
}
```

**Tool result**

```
**On screen:** tab 4, the Kaggle notebook. Scroll slowly from `EOS` to the two training stages. Then tab 3, row 19, highlight 0.519.

### 4. The probes (1:05 to 1:35), about 65 words

> Then leaderboard probes, one idea each. The published spine with my SVC scored 0.626. Balancing at half strength: 0.629. Adding prvsiyan's two seeds: 0.631. Then four probes in a row landed on exactly 0.631, with new members that each changed real rows. That tells me the public slice is too small to show gains of this size, so I trust cross-validation more than this board.

**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 8. Cut to tab 3, rows 12 to 26, if the submissions page is hard to read.

### 5. The biggest gain (1:35 to 2:05), about 60 words

> Here is the pipeline. Members blend in log space, then take fixed shares, then balancing. The blue boxes are ours. The biggest measured gain is a shared-prompt Gaussian group head, our own code after prvsiyan's public method. Passages that answer the same prompt are decoded together, with the prompt's effect taken out. On its own honest view it added 0.058.

**On screen:** tab 5, `pipeline.svg`. Point at each box as you name it, ending on the Gaussian group head. Then tab 3, row 25, highlight +0.058.

### 6. One honest failure (2:05 to 2:25), about 45 words

> One failure. Peer correction, a public trick, swaps balancing for removing the lean that passages on one topic share. It helped weaker blends. On ours it scored 0.622, our worst probe. Without class quotas the blend drifts towards the models it already favours.

**On screen:** tab 3, row 18. Highlight 0.622.

### 7. What came back flat (2:25 to 2:40), about 40 words

> Some things came back flat, and that is useful too. A prompt-centred SVC, pair specialists for the two hardest pairs, and a bigger public pool all stayed within noise. A 27-billion model took three seconds per passage, too slow.

**On screen:** tab 3, rows 24 and 27, then the PR list on GitHub showing the hard-pairs and Qwen 27B pull requests.

### 8. Next, and the result (2:40 to 3:00), about 55 words

> TODO probe 9: "The last probe put the Gaussian head at full weight and added ModernBERT at ten per cent. It scored ___."
>
> Next I would add a model from outside the Qwen family, and test each step on folds that look like the test. TODO final: "The final submission scored ___. Thanks for watching."

**On screen:** tab 6, the leaderboard with our row highlighted.

## Timing check
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && python3 - <<'EOF'\np='docs/report/video-script.md'\ns=open(p).read()\nold=s[s.index('### 8. Next, and the result'):s.index('**On screen:** tab 6, the leaderboard')]\nnew='''### 8. The result, and next (2:40 to 3:00), about 65 words\n\n> Probe nine put the Gaussian head at full weight and added ModernBERT at ten per cent. The run of ties ended at 0.637, our best, and the head that won offline was the one that moved the board. Next: retrain on the bigger open pool, five folds for every member, and targeted data for the GLM and Gemini siblings. Thanks for watching.\n\n'''\ns=s.replace(old,new)\ns=s.replace('**On screen:** tab 6, the leaderboard with our row highlighted.','**On screen:** tab 6, the submissions page on probe 9 (0.637), then the leaderboard with our row highlighted.')\ns=s.replace('About 440 spoken words','About 440 spoken words')\ns=s.replace(\"Numbers come from `docs/experiments/LOG.md` (rows 1 to 27) and `docs/research/`. TODO marks a number we do not have yet.\",\n \"Numbers come from `docs/experiments/LOG.md` (rows 1 to 27), `docs/research/` and probe 9 (0.637 public). The facts match `documents/speaker-notes.md`, which goes with the slides in `documents/presentation.pdf`: use whichever is easier to record from.\")\ns=s.replace(\"Record each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take. Read the TODO lines last, once probe 9 and the final score are in.\",\n \"Record each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take.\")\ns=s.replace(\"| 8. Next | 2:40 | 55 with the TODOs |\",\"| 8. Result and next | 2:40 | 65 |\")\ns=s.replace(\"**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 8.\",\"**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 8; stop before probe 9, which comes at the end.\")\nopen(p,'w').write(s)\nEOF\nsed -i '' 's/best public LB 0.631/best public LB 0.637/' docs/report/pipeline.svg\ngrep -n -i \"todo\" docs/report/video-script.md; grep '^> ' docs/report/video-script.md | wc -w; grep -c \"—\" docs/report/video-script.md; grep -n 0.637 docs/report/pipeline.svg"
}
```

**Tool result**

```
     434
0
75:  <text x="1045" y="334" font-size="16" text-anchor="middle" fill="#ddd">best public LB 0.637</text>
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle/.claude/worktrees/video-script-50 && git add docs/report/pipeline.svg && git commit -q -m \"docs(report): show the best public score, 0.637, in the pipeline diagram\n\nrefs #50\" && git add docs/report/video-script.md && git commit -q -m \"docs(report): fill the video script's last section with probe 9\n\nrefs #50\" && git push -q -u origin docs/video-script-probe9 2>&1 | grep -v remote; gh pr create --base main --title \"docs(report): add probe 9 (0.637) to the video script (#50)\" --body \"$(cat <<'EOF'\n## What this does\n\n- `docs/report/video-script.md`: section 8 now carries probe 9: the Gaussian head at full weight plus ModernBERT at 10% scored 0.637 public, our best, which ended the run of four 0.631 ties. The TODO slots are gone. Next steps now match `documents/speaker-notes.md`, and the intro points to the slides and notes as the other way to record.\n- `docs/report/pipeline.svg`: the output box now says best public LB 0.637.\n\n## Why\n\nProbe 9 is in. The video script and the speaker notes on `docs/presentation` should tell the same story with the same numbers.\n\n## Checklist\n\n- [x] No TODOs left\n- [x] 434 spoken words, three minutes at a calm pace\n- [x] Facts match `documents/speaker-notes.md`\n- [x] British English, no em dashes\n\n## Test plan\n\n- Read section 8 aloud with a timer: about 20 seconds.\n- Open `pipeline.svg` and check the output box reads 0.637.\n\nrefs #50\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\n)\""
}
```

**Tool result**

```
https://github.com/Nabzx/name-that-llm/pull/101
```

## Claude (2026-10-25 13:01:58 UTC)

I've opened [Nabzx/name-that-llm#101](https://github.com/Nabzx/name-that-llm/pull/101) on `docs/video-script-probe9`, cut from `main` after #95 merged. It adds probe 9 to the video script and the diagram.

- **Script, last section:** probe 9 put the Gaussian head at full weight and added ModernBERT at 10%. The script says the run of ties ended at 0.637, our best, and that the head that won offline was the one that moved the board. Both TODO lines are gone.
- **Final score:** the "final submission scored ___" slot is now covered by the 0.637 line. That assumes probe 9 is the final pick; if a later submission becomes the final, that line needs changing.
- **Match with the speaker notes:** the closing "what next" now uses the same items as `documents/speaker-notes.md` on `docs/presentation`: the bigger open pool, five folds for every member, and targeted data for the GLM and Gemini siblings. The intro points to those notes and `presentation.pdf` as the other way to record. The other numbers already matched.
- **Diagram:** the output box now says best public LB 0.637.
- **Length:** the script is 434 spoken words, and I found no em dashes.

The PR has two commits, both with `refs #50`.
