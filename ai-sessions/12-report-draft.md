## Nabil (2026-10-25 00:42:59 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

Background (from the session that started this one):
Repo ~/Random/research-arm-kaggle (GitHub Nabzx/name-that-llm, private) is Nabil's entry to the Imperial "Name That LLM" Kaggle competition (18-way LLM attribution, balanced accuracy). Deadline Sun 25 Oct 2026 18:00 GMT. Scoring is 50% public LB and 50% a report of at most 2 pages (approach, hypotheses, ablation table, one failure, next steps), plus a notebook link, a 3-minute video and all AI logs. Read AGENTS.md and CLAUDE.md first. Writing style is strict: Nabil's tone, British English, plain simple words, concise, few rule-of-three lists, few Oxford commas, no em dashes, no hype words. Never submit to Kaggle, never touch Kaggle tokens.

Task:
Bring docs/report/report-draft.md and docs/report/ablation.md up to date with every result so far, ready for the final numbers on Sunday afternoon (#49, #50).

Sources of truth: docs/experiments/LOG.md on branch feat/own-lora (rows 1 to 17; merge that branch's LOG.md rows into yours, main lacks rows 13 to 17), the research notes in docs/research/, CITATIONS.md, docs/ai-sessions.md.

Public LB so far: probe 1 0.626 (our SVC + Nishant Dahal's published Qwen3-14B, Qwen2.5-14B, DeBERTa, his weights, full Sinkhorn), probe 2 0.629 (half-strength Sinkhorn), probe 3 0.631 (+ prvsiyan's two Qwen3-8B seeds at 10%, SVC pool weight 0.25), probe 4 0.627 (+ Zbigniew's in-style Qwen3-8B at 10%, worse), probe 5 0.631 (probe 3 + UltraBERT 10% + ATLAS 10% on rows with top-two gap < 0.3: no change). Probe 6 (peer correction in place of Sinkhorn) is being scored. Our own Qwen3-8B LoRA is training overnight.

Must-haves:
- The organisers ruled published weights are allowed if the report shows "you understand what's happening and what you've changed/built on top". Write a short, concrete section on how the LoRA24 readout works (docs/research/issue-21-lora24-readout.md) and list exactly what is ours: the loader written from scratch, the end-token and label-order checks, the pool build, our SVC, half Sinkhorn, peer correction, the witness, our own LoRA, the blend experiments.
- Ablation table from the probes with deltas, plus CV rows for the SVC work.
- One failure, told honestly (probe 4 Zbigniew, or the probe 5 tie, or the Gemma witness crash: pick the most instructive).
- Next steps.
- Leave clearly marked TODO slots for probe 6, our own LoRA and the final pick.
- Must fit 2 pages: check with a word count (about 900 to 1,000 words plus the table).

Commit small, push the branch, open a PR in the house format (What this does / Why / Result / Checklist / Test plan). Done = PR open and a short note of what is still TODO.

Shared notes folder for this group of sessions (outside the repo; read siblings' notes there, leave hand-over notes or large findings there instead of pasting them into messages): /Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git status -sb && git branch -a && ls docs docs/report docs/research docs/experiments; ls \"/Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459\"; git log --oneline -5"
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
  feat/spine-load-27
  feat/submission-28-29
  feat/svc-pool-26
  feat/svc-train-25
+ feat/ultrabert-28
  feat/witness-member-36
  feat/witness-proto-34
+ feat/zbigniew-28
  feat/zbigniew-check-28
  main
  worktree-agent-a1324620553d0edac
  worktree-agent-a13e2b885fcd4b2a9
  worktree-agent-a17f1ecd3da6ad194
  worktree-agent-a1ee5f4536de328ae
  worktree-agent-a4bfee7ca148f3e7a
  worktree-agent-ab02e241af761ffd9
  worktree-agent-ad2df954f09199f68
  worktree-agent-ad4f26265ddeef030
  remotes/origin/HEAD -> origin/main
  remotes/origin/ci/guardrails-59
  remotes/origin/docs/adr-0001-12
  remotes/origin/docs/ai-session-export
  remotes/origin/docs/ai-sessions-15
  remotes/origin/docs/blend-validation-20
  remotes/origin/docs/checkpoints-18
  remotes/origin/docs/citations-fork
  remotes/origin/docs/hard-pairs-38-40
  remotes/origin/docs/kaggle-cli-17
  remotes/origin/docs/kaggle-pitfalls-22
  remotes/origin/docs/likelihood-32-33
  remotes/origin/docs/lora24-21
  remotes/origin/docs/pools-19
  remotes/origin/docs/probe1-log
  remotes/origin/docs/report-48
  remotes/origin/docs/roadmap-direction
  remotes/origin/docs/sinkhorn-23
  remotes/origin/docs/top-teams-24
  remotes/origin/docs/witnesses-31
  remotes/origin/feat/folds-13
  remotes/origin/feat/metric-14
  remotes/origin/feat/own-lora
  remotes/origin/feat/peer-correction-46
  remotes/origin/feat/pool-gemini-64
  remotes/origin/feat/pool-weight-63b
  remotes/origin/feat/probe2-half-sinkhorn
  remotes/origin/feat/probe3-pool025
  remotes/origin/feat/probe5-atlas
  remotes/origin/feat/probe6-peer
  remotes/origin/feat/prvsiyan-28
  remotes/origin/feat/spine-load-27
  remotes/origin/feat/svc-pool-26
  remotes/origin/feat/svc-train-25
  remotes/origin/feat/ultrabert-28
  remotes/origin/feat/witness-member-36
  remotes/origin/feat/witness-proto-34
  remotes/origin/feat/zbigniew-28
  remotes/origin/feat/zbigniew-check-28
  remotes/origin/main
docs:
adr
ai-sessions.md
experiments
report
research

docs/experiments:
LOG.md

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

... (496 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && cat AGENTS.md CLAUDE.md | head -200; git log --oneline main -3; git log --oneline origin/docs/report-48 -5; git diff --stat main origin/docs/report-48"
}
```

**Tool result**

```
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

Child tickets hang off the map issue with a `Part of: #n` line, and carry one of:

- `wayfinder:research`: investigate and report back, no implementation.
- `wayfinde
... (5953 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/research-arm-kaggle && git fetch -q && git ls-tree main docs/report/ && git merge-base --is-ancestor origin/docs/report-48 main && echo merged; git worktree add -q ../nthl-report -b docs/report-final-49 origin/main && cd ../nthl-report && cat docs/report/report-draft.md docs/report/ablation.md && wc -w docs/report/*"
}
```

**Tool result**

```
100644 blob 628494c9a18141992b69ac1c61f102be7dc69801	docs/report/ablation.md
100644 blob 6508df8ffb00400b2fbdf3885726960b2890cbbe	docs/report/report-draft.md
merged
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

In every pool run gemini-3.7-flash recall fell from 0.41 to 0.19 while 3.8-flash rose
... (4793 more characters in the raw transcript)
```

## Claude (2026-10-25 00:43:18 UTC)

Now the log on feat/own-lora and the research notes.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git diff origin/main origin/feat/own-lora -- docs/experiments/LOG.md; echo ======; git show origin/feat/own-lora:docs/experiments/LOG.md"
}
```

**Tool result**

```
diff --git a/docs/experiments/LOG.md b/docs/experiments/LOG.md
index dc7730c..3e65171 100644
--- a/docs/experiments/LOG.md
+++ b/docs/experiments/LOG.md
@@ -16,3 +16,8 @@ One row per run. CV is balanced accuracy on the frozen `GroupKFold` folds over `
 | 10 | 2026-10-24 | #63 | Row 6 with the pool at weight 0.35. Notebook `nthl-svc-pool-all-w035` v1, CPU 68 min | 0.4827 | +0.134 | not submitted | Lower weight helps a little more. |
 | 11 | 2026-10-24 | #63 | Row 6 with the pool at weight 0.25. Notebook `nthl-svc-pool-all-w025` v1, CPU 63 min | **0.4838** | **+0.135** | not submitted | Best SVC. The curve is flattening (1: 0.4634, 0.5: 0.4782, 0.35: 0.4827, 0.25: 0.4838), so stop here. Use 0.25 in the submission. |
 | 12 | 2026-10-24 | #29 | **Probe 1, first submission.** Our SVC (pool weight 0.5) plus Nishant Dahal's Qwen3-14B and Qwen2.5-14B LoRAs with centred twins and his DeBERTa, through our loader, his published weights, full Sinkhorn. Notebook `nthl-submit` v2 | n/a | n/a | **0.626** | 0.006 above Nishant's own 0.620, so the pipeline reproduces his and our SVC (0.478 against his 0.473) adds a little. Above the uni leader's 0.622. Scoring took about 2 hours. |
+| 13 | 2026-10-24 | #30 | **Probe 2.** Probe 1 with Sinkhorn at half strength (alpha 0.5). Notebook `nthl-submit` v3 | n/a | n/a | **0.629** | +0.003. Full balancing over-corrects when the test mix is uneven, as #23 expected. Keep alpha 0.5. |
+| 14 | 2026-10-24 | #30 | **Probe 3.** Probe 2 plus prvsiyan's two Qwen3-8B seeds at a fixed 10% share, and the SVC pool weight down to 0.25 (row 11). Notebook `nthl-submit` v5 | n/a | n/a | **0.631** | +0.002, best so far. Two changes at once, so the gain is shared between them. |
+| 15 | 2026-10-24 | #30 | **Probe 4.** Probe 3 plus Zbigniew's in-style Qwen3-8B LoRA at 10% (spine 80%, prvsiyan 10%). Notebook `nthl-submit` v6 | n/a | n/a | 0.627 | -0.004. His model was trained on in-style pool passages much like ours, so it may add the same errors rather than new ones. Drop it. |
+| 16 | 2026-10-24 | #36 | UltraBERT 5-fold encoder through our loader (`nthl.ultrabert`), host train. Notebook `nthl-spine-ultrabert` v1, 3 min | in-sample 0.966 | n/a | not submitted | A loader check, not a score. 65 s per 1,000 passages, so about 4 to 5 min on the test. Ready to probe at 10%. |
+| 17 | 2026-10-25 | #28 | **Probe 5.** Probe 3 plus UltraBERT at a fixed 10% and Joseph Adamski's ATLAS at 10% on rows whose top two are under 0.3 apart; Zbigniew off. Notebook `nthl-submit` v8 | n/a | n/a | 0.631 | No change against probe 3. The two members that took the public 0.620 to 0.636 add nothing on top of our blend, so our SVC, half Sinkhorn and prvsiyan's seeds likely already cover the rows they fix. Two changes at once, so one may help and the other hurt. |
======
# Experiment log

One row per run. CV is balanced accuracy on the frozen `GroupKFold` folds over `prompt_id`. Δ is against the current baseline.

| # | Date | Issue | Change | CV | Δ | Public LB | Notes
... (6232 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && cat docs/research/issue-21-lora24-readout.md; git log --oneline origin/main..origin/feat/own-lora; git log --oneline origin/main..origin/feat/peer-correction-46; git log --oneline origin/main..origin/feat/probe6-peer | head; git diff --stat origin/main origin/feat/own-lora | tail -15"
}
```

**Tool result**

```
# Issue #21: How the LoRA24 readout works

## Question

How do the published LoRA checkpoints turn a passage into 18 probabilities? We need to write our own loader (#27), so we need every step and every shape, without copying anyone's code.

## What we looked at

Nishant Dahal's *Two-stage LoRA, part two* notebook (read only) and the `meta.json` and `adapter_config.json` files of his checkpoints in `nishantdahal/ntl-handwriting-blend`. prvsiyan's and Zbigniew's checkpoints follow the same recipe on Qwen3-8B, with their own file layouts.

## The recipe, step by step

Shapes are for `lora_qwen3_14b`. H is the hidden size, 5,120 for both 14B models.

| Step | What happens | Shape |
|---|---|---|
| 1. Tokenise | No special tokens, cut at `max_len` 447, then append the end token from `meta.json` (`eos_id` 151643, `<|endoftext|>`) | up to 448 ids per passage |
| 2. Trunk | Only the first `keep` decoder layers: 27 of 40 for Qwen3-14B, 32 of 48 for Qwen2.5-14B. The final norm is switched off (identity), because training ran without it | |
| 3. Merge the adapter | For every adapted linear layer (q, k, v, o, gate, up, down): W ← W + B·A × alpha/r. Here r is 16 and alpha 32, so the scale is 2 | A is [16, in], B is [out, 16] |
| 4. Run | Hidden states of every layer. Index 0 is the embeddings, index l is after decoder layer l | [batch, tokens, H] per layer |
| 5. Pool | At 8 depths from `meta.json` (2, 4, 6, 9, 13, 17, 22, 27 for Qwen3-14B), a masked mean over tokens. The first token is left out of the mean, because it soaks up attention whatever it is | [batch, H] per depth |
| 6. Normalise | One LayerNorm per depth, in fp32, with weights stored in `heads.safetensors` (`norms.*`) | [batch, H] per depth |
| 7. Concatenate | The 8 normalised vectors side by side | [batch, 40,960] |
| 8. Plain head | Linear: Z·Wᵀ + b, then softmax | W is [18, 40,960], output [batch, 18] |
| 9. Centred head (optional) | For test passages in a topic cluster of 2 to 18, subtract the cluster's mean Z, then a second linear head trained on such differences | same shapes |

DeBERTa-v3-large is simpler: the full encoder, a masked mean of the last layer (first token kept), one linear head [18, 1,024], `max_len` 384, run in fp32.

The 0.620 notebook blends in log space: SVC 1.0, Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25, each centred twin 0.1, then Sinkhorn balancing.

## Traps

- **Label order.** The checkpoints list `claude-fable-5` after `claude-sonnet-5`. Ours lists it first among the Claude labels. Columns must be reordered by the `labels` field of each `meta.json`. Skip that and claude-fable-5's probabilities land under another Claude model's name with no error.
- **End token.** Kaggle's Qwen3-14B-Base ships the chat model's tokenizer, whose end token is `<|im_end|>` (151645). Training used `<|endoftext|>` (151643). Use `eos_id` from `meta.json`, never the tokenizer's. Getting this wrong changed 14% of top picks in Nishant's tests.
- **Final norm.** Leave it switched off. With 
... (3283 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git show origin/feat/probe6-peer:src/nthl/blend.py | grep -n -A30 \"def peer_correct\\|def cluster_sim\\|def sinkhorn\\|def low_margin\" | head -150; git show origin/feat/own-lora:notebooks/train-lora/run.py | head -50; git show origin/feat/own-lora:src/nthl/train_lora.py | head -40"
}
```

**Tool result**

```
22:def sinkhorn(p, alpha=0.5, iters=300):
23-    """Scale each column towards (1 - alpha) * its total + alpha * n / K, renormalising rows."""
24-    target = (1 - alpha) * p.sum(0) + alpha * len(p) / p.shape[1]
25-    c = np.ones(p.shape[1])
26-    for _ in range(iters):
27-        r = p * c
28-        r /= r.sum(1, keepdims=True)
29-        c *= target / r.sum(0)
30-    r = p * c
31-    return r / r.sum(1, keepdims=True)
32-
33-
34-def topic_groups(texts, cut=0.85, lo=2, hi=18, return_matrix=False):
35-    """Cluster by content words. Returns cluster ids and which rows sit in a cluster of lo to hi
36-    (and the TF-IDF matrix, if asked)."""
37-    x = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(list(texts))
38-    d = np.clip(1 - (x @ x.T).toarray(), 0, None)
39-    np.fill_diagonal(d, 0)
40-    groups = fcluster(linkage(squareform(d, checks=False), method="average"), cut, criterion="distance")
41-    size = np.bincount(groups)[groups]
42-    twin = (size >= lo) & (size <= hi)
43-    return (groups, twin, x) if return_matrix else (groups, twin)
44-
45-
46-def centred_probs(z, groups, twin, w, b):
47-    """Centred-head probabilities for the twin rows: features minus their cluster mean."""
48-    zc = z.astype(np.float32).copy()
49-    for g in np.unique(groups[twin]):
50-        rows = groups == g
51-        zc[rows] -= zc[rows].mean(0)
52-    return softmax(zc[twin] @ w.T + b)
--
55:def cluster_similarity(x, groups):
56-    """For every row, the mean pairwise cosine between different members of its cluster (0 for singletons).
57-
58-    `x` holds L2-normalised TF-IDF rows, as TfidfVectorizer gives by default.
59-    """
60-    sim = np.zeros(len(groups))
61-    for g in np.unique(groups):
62-        rows = np.flatnonzero(groups == g)
63-        n = len(rows)
64-        if n < 2:
65-            continue
66-        block = (x[rows] @ x[rows].T).toarray()
67-        sim[rows] = (block.sum() - np.trace(block)) / (n * (n - 1))
68-    return sim
69-
70-
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
84-        if n < lo or n > hi:
85-            continue
86-        r = (n - 1) / (n - 1 + prior) * np.clip((sim[rows[0]] - 0.1) / 0.4, 0, 1)
87-        total = margin[rows].sum(0)
88-        others = (total
... (3668 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git show origin/feat/probe6-peer:CITATIONS.md | sed -n 1,200p | cut -c1-300; git show origin/feat/probe6-peer:notebooks/submit/run.py | grep -n -i \"witness\\|peer\\|PROBE\\|share\\|alpha\" | head -40"
}
```

**Tool result**

```
# Citations

Every public notebook, discussion, dataset and pretrained model used or read for this entry.

## Notebooks and discussions

- avazaku, *Name That LLM, Peer Correction: 0.631*. https://www.kaggle.com/code/avazaku/name-that-llm-fingerprint-ensemble. Read for leave-one-out peer correction in place of Sinkhorn.
- Roman Tamrazov, *Name That LLM, Who am I?*. https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i. Read for UltraBERT use and consensus-gated reranking.
- Vladislav Lavrentev, *0.637 LoRA Llama Qwen3.8-27B*. https://www.kaggle.com/code/lightningv08/0-637-lora-llama-qwen3-8-27b. Read to see what the 0.637 to 0.639 stacks contain.
- gougou1, *ntllm v9 final*. https://www.kaggle.com/code/gougou1/ntllm-v9-final. Read for in-style LoRA plus Gaussian group heads.
- s0chn1y, *Sinkhorn balancing on an uneven test*. https://www.kaggle.com/code/jusill/sinkhorn-balancing-on-an-uneven-test. Read for the risk of full balancing.
- zhuyu2, *ntl 607 l24x w10*, and arlearonreidma's 0.636 to 0.639 forks. https://www.kaggle.com/code/zhuyu2/ntl-607-l24x-w10. Read to confirm the public stacks share one lineage.
- Discussions 752365 (Sinkhorn on an uneven test), 752357 (what 134 public versions show), 751871 (what transferred to the LB) and 752197 (the GLM and Fable corner). https://www.kaggle.com/competitions/name-that-llm/discussion/752357. Read for the plan to 0.64.

- Lucas Boesen, *LoRA x Adaptive Centering*. https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering. Read for half-strength balancing and centring (#23).
- prvsiyan, *The Prompt Is Not the Author*. https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author. Read for the Gaussian group head and test-time grouping (#23).
- Zbigniew Kirchner, *0.574 In-Style Qwen3-8B LoRA (fully public)*. https://www.kaggle.com/code/zbigniewkirchner/0-574-in-style-qwen3-8b-lora-fully-public. Read for how his in-style checkpoint is loaded (#28).

- HALX2158 and replies, *Forked a 0.600+ notebook and hit the 9-hour limit? Check the Qwen3.5-9B load time*, Name That LLM discussion 751268. https://www.kaggle.com/competitions/name-that-llm/discussion/751268. Read for load times from Datasets against Kaggle Models (#22).

- Nishant Dahal, *Two-stage LoRA, part two* (LB 0.620). https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two. Read for the LoRA24 readout (#21) and the blend weights.
- Nishant Dahal, *Two-stage LoRA on the public pool* (LB 0.592). https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool. Read for the training recipe.

- Adisak Sukul and replies (GoPain), *New here? Grouped CV that tracks the LB, CPU baseline code (LB 0.353), pitfalls, 2xT4 tips*, Name That LLM discussion 750390. https://www.kaggle.com/competitions/name-that-llm/discussion/750390. Read for train-only baseline scores and the CV to LB gap.

## Datasets

- Name That LLM: extra GPT and Gemini generations (abdullah1afify/ntl-generations), CC BY 4.0. https://www.kaggle.com/d
... (9557 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "ls -la \"/Users/nabilshah/Library/Application Support/Claude/side-session-notes/local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459\"; cd ~/Random/nthl-report && head -30 docs/ai-sessions.md | cut -c1-250; grep -n -i \"witness\\|gemma\" docs/research/issue-31-witness-models.md | head -15"
}
```

**Tool result**

```
total 0
drwx------@ 2 nabilshah  staff  64 Oct 25 01:42 .
drwx------@ 3 nabilshah  staff  96 Oct 25 01:42 ..
# AI sessions

Every AI session used for this entry, as required by the competition rules. Each one is exported before submission.

## How to export a session

1. Try the app first: the session menu's Export writes a zip to Downloads. For session 1 it failed twice on 2026-10-24.
2. Otherwise run our script with the transcript id:

   ```bash
   .venv/bin/python scripts/export_session.py <transcript-id> --name <nn-short-name>
   ```

   It finds `~/.claude/projects/*/<transcript-id>.jsonl` and writes `ai-sessions/<name>.zip` and `ai-sessions/<name>.md` in the repo, so the logs travel with the code. The zip holds the raw transcript untouched, the subagent transcripts if there are 
3. The transcript id is the file name under `~/.claude/projects/`. The app id (`local_...`) is not the same thing.

Re-run the export after the last message of a session, and commit the result. An export is a snapshot: anything said after it is missing.

Before handing in, check each export for anything personal. Session 1 holds Nabil's email and memory notes on unrelated projects. Nothing secret: a scan on 2026-10-25 for Kaggle, GitHub, Hugging Face, OpenAI and AWS token patterns found none. Redacti

Every new session or subagent run gets a row here the moment it starts. Nabil used no AI chat outside Claude Code for this entry (confirmed 2026-10-25).

## Register

| # | Tool | Session | Started | Purpose | Exported |
|---|---|---|---|---|---|
| 1 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament", app id `local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459`, transcript `5d7d3493-861c-4044-bd86-4e023fd27b66` | 2026-10-24 02:03 BST | Track research, Kaggle playbook, task analysis, r
| 2 | Claude Code subagent of session 1, Claude Opus 5.5 | Research agent for #20 (blend validation). Transcript in session 1's `subagents` folder | 2026-10-24 16:28 BST | Read and wrote the #20 research note on its own branch | Exported with session
| 3 | Claude Code subagent of session 1, Claude Opus 5.5 | Research agent for #23 (Sinkhorn and centring). Transcript in session 1's `subagents` folder | 2026-10-24 16:28 BST | Read, simulated on our OOF predictions and wrote the #23 note | Exported 
| 4 | Claude Code subagent of session 1, Claude Opus 5.5 | Research agent for #24 (what top teams do). Transcript in session 1's `subagents` folder | 2026-10-24 16:28 BST | Read the leaderboard and discussion and wrote the #24 note | Exported with se
1:# Issue #31: Which open witness models fit on two T4s
15:| Qwen | Qwen3-4B-Base, 8B-Base, 14B-Base (`qwen-lm/qwen-3/Transformers/4b-base/1` and up) | Yes. 4B on one T4, 14B over both | Witness: Qwen3-4B-Base, cheap and a base model |
16:| Google | Gemma 4 E2B, E4B, 12B, 26B A4B, 31B (`google/gemma-4/transformers/gemma-4-e4b/1`, base) | E4B yes (about 16 GB with its per-layer embeddings, so over both T4s), 12B yes over both | Witness: Gemma
... (612 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git diff origin/main origin/feat/own-lora -- docs/experiments/LOG.md | git apply && git add docs/experiments/LOG.md && git commit -q -m \"docs(log): bring in probes 2 to 5 and the UltraBERT check from feat/own-lora\n\nrefs #48\" && git log --oneline -1"
}
```

**Tool result**

```
c6be233 docs(log): bring in probes 2 to 5 and the UltraBERT check from feat/own-lora
```

## Claude (2026-10-25 00:43:51 UTC)

Log rows merged; now rewriting the ablation table with the probe deltas.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && python3 - <<'EOF'\np='docs/report/ablation.md'\ns=open(p).read()\nold=s[s.index('## Leaderboard probes'):]\nnew='''## Leaderboard probes\n\nEach probe is meant to change one thing (#20). Δ is against the probe named in \"Against\". Probes 3 and 5 changed two things at once, so their Δ is shared between the two.\n\n| Log row | Probe | Change | Against | Public LB | Δ |\n|---|---|---|---|---|---|\n| 12 | 1 | Nishant's published spine (Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25, centred twins 0.1) through our loader, with our pool SVC (weight 0.5) in place of his, full-strength Sinkhorn | his notebook, 0.620 | 0.626 | +0.006 |\n| 13 | 2 | Sinkhorn at half strength (alpha 0.5) | Probe 1 | 0.629 | +0.003 |\n| 14 | 3 | Add prvsiyan's two Qwen3-8B seeds at a fixed 10%, and the SVC pool weight down to 0.25 | Probe 2 | **0.631** | +0.002 |\n| 15 | 4 | Add Zbigniew Kirchner's in-style Qwen3-8B at 10% | Probe 3 | 0.627 | -0.004 |\n| 17 | 5 | Add UltraBERT at 10%, and ATLAS at 10% on rows whose top two are under 0.3 apart | Probe 3 | 0.631 | 0 |\n| TODO | 6 | Peer correction in place of Sinkhorn | Probe 3 | TODO | TODO |\n| TODO | 7 | Our own Qwen3-8B LoRA as a member | TODO | TODO | TODO |\n| TODO | final | TODO: the two picks for the private LB | | TODO | |\n\nRow 16 is the UltraBERT loader check: in-sample 0.966 on host train, not a score.\n\n## What the table says\n\n- Data did most of the work. The pool gave +0.115 at weight 1 and down-weighting it gave +0.020 more. The SVC itself never changed.\n- More rows beat cleaner labels: the free proxy rows still helped (rows 4 and 5).\n- The pool-weight curve has flattened, so we stopped at 0.25.\n- Leaving suspect labels out hurt more than keeping them (row 8).\n- The likelihood witnesses carry real but small signal, about +0.003 on the SVC.\n- On the LB, the steps that changed how we combine and balance gave the most: half Sinkhorn +0.003. Adding strong published members gave little or hurt: prvsiyan's seeds shared +0.002, Zbigniew's in-style model -0.004, UltraBERT and ATLAS nothing.\n- From probe 3 on, every gain is 0.004 or less. On a public LB of 4,000 passages split 18 ways, that is close to noise, so we only kept changes the CV or a research note also backed.\n'''\ns=s.replace(old,new)\nopen(p,'w').write(s)\nEOF\ngit add docs/report/ablation.md && git commit -q -m \"docs(report): add probes 2 to 5 with deltas and TODO rows for probe 6, our LoRA and the final pick\n\nrefs #48\" && echo ok"
}
```

**Tool result**

```
ok
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && sed -i '' 's/On a public LB of 4,000 passages split 18 ways, that is close to noise/The public LB scores only part of the 4,000 test passages, split 18 ways, so that is close to noise/' docs/report/ablation.md && grep -n \"close to noise\" docs/report/ablation.md && git commit -qam \"docs(report): fix the public LB size wording\" && echo ok"
}
```

**Tool result**

```
55:- From probe 3 on, every gain is 0.004 or less. The public LB scores only part of the 4,000 test passages, split 18 ways, so that is close to noise, so we only kept changes the CV or a research note also backed.
ok
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && sed -i '' 's/so that is close to noise, so we only kept/so that is close to noise. We only kept/' docs/report/ablation.md && cat > docs/report/report-draft.md <<'EOF'\n# Name That LLM: report draft\n\nNabil Shah, solo entry, Imperial Research Arm qualifier. Draft for a two-page PDF (#49). Numbers come from `docs/experiments/LOG.md` and `docs/research/`. Every **TODO** is a number we get on Sunday afternoon (#49, #50).\n\n## Approach\n\nWe wrote our own code from scratch. The first model was a four-view n-gram LinearSVC on host train only, scored on frozen `GroupKFold` folds over `prompt_id`: 0.3485 balanced accuracy, in line with public train-only SVCs (0.343 to 0.370).\n\nWe then worked in two branches (#20). Branch A is a published spine we run through our own loader but never re-fit: Nishant Dahal's two-stage LoRA blend, 0.620 on the public LB. Branch B is what we can check on our own folds, starting with our SVC. Published members have no out-of-fold predictions, so the LB is the only test for them. We probed it one change at a time and only kept a change that our CV or a research note also backed.\n\n## Hypotheses\n\n- **Data beats model size.** Public work showed 8B and 14B backbones tied on the same data, while a pool of 84k passages in place of 35k gave +0.028 on the LB (#24).\n- **Labs differ in surface style, siblings do not.** Our baseline put the right lab first 0.57 of the time, and half its errors stayed inside the right lab. Hardest: gemini-3.7-flash against 3.8-flash, and glm-5.3 read as claude-opus-5.\n- **The prompt is louder than the author.** Passages that answer the same prompt share content, so taking out a topic estimate should leave mostly style (#23).\n- **A lab's own model finds its text more likely.** Likelihood features from a base model in each family should add what n-grams miss (#31 to #33).\n- **The test mix is unequal.** Balancing towards 1/18 should help, but at half strength, since we cannot see the test's skew (#23).\n\n## How the LoRA24 readout works\n\nThe organisers allow published weights if we show what we built on them. Each published LoRA24 checkpoint turns a passage into 18 probabilities like this (#21):\n\n1. Tokenise with no special tokens, cut at 447, and append the end token from the checkpoint's `meta.json` (151643), not the tokenizer's (151645).\n2. Keep only the first 27 of Qwen3-14B's 40 decoder layers, with the final norm switched off.\n3. Merge the adapter into each q, k, v, o, gate, up and down weight: W + B·A × 2.\n4. Take a masked mean of the hidden states at 8 depths, leaving the first token out.\n5. LayerNorm each depth, join them into one 40,960-wide vector and apply a linear head [18, 40,960].\n6. A second, centred head reads the same vector minus its topic cluster's mean.\n\nThe traps are silent. A wrong label order puts claude-fable-5's probabilities under another Claude model with no error. The wrong end token changed 14% of top picks in Nishant's tests.\n\n**What is ours:**\n\n- The loader (`nthl.lora24`), written from scratch from that recipe. It reads Nishant's and prvsiyan's formats into one spec and splits a 14B model over two T4s.\n- Its checks: it asserts the end token and reorders label columns by `meta.json`. On host train the Qwen3-14B checkpoint scored 0.858 in-sample with every Claude label right, so the reorder works (log row 7).\n- The pool: 84,470 passages from 20 open-licence datasets, filtered, de-duplicated and tagged by source (#19, #26).\n- Our SVC and its pool-weight study, 0.3485 to 0.4838 CV.\n- Half-strength Sinkhorn, from our own simulation on the SVC's out-of-fold predictions (#23).\n- Peer correction, re-implemented from avazaku's idea: each topic cluster loses a shrunk share of the other members' mean SVC lean (#46).\n- The witness: Qwen3-4B-Base likelihood features plus tokens per word under each lab's tokenizer (#34).\n- Our own Qwen3-8B LoRA, trained from scratch overnight with our own code on our pool, then host train (#43).\n- The blend experiments: probes 1 to 6 and the rule for judging members we cannot cross-validate (#20).\n\n**Published, used as released:** Nishant Dahal's Qwen3-14B, Qwen2.5-14B and DeBERTa-v3-large with his blend weights, prvsiyan's two Qwen3-8B seeds, and in probes 4 and 5 Zbigniew Kirchner's in-style Qwen3-8B, UltraBERT and ATLAS. All are listed in `CITATIONS.md`.\n\n## Ablation\n\nFull table in `docs/report/ablation.md`.\n\n| Step | CV | Public LB | Δ |\n|---|---|---|---|\n| SVC, host train only | 0.3485 | | |\n| + whole pool, weight 1 | 0.4634 | | +0.115 CV |\n| pool weight 0.5 | 0.4782 | | +0.015 CV |\n| pool weight 0.25 | 0.4838 | | +0.006 CV |\n| + witness features (on the weight 0.5 SVC) | about +0.003 | | +0.003 CV |\n| Probe 1: Nishant's spine with our SVC, full Sinkhorn | | 0.626 | +0.006 on his 0.620 |\n| Probe 2: half Sinkhorn | | 0.629 | +0.003 |\n| Probe 3: + prvsiyan's seeds at 10%, SVC weight 0.25 | | **0.631** | +0.002 |\n| Probe 4: + Zbigniew's in-style model at 10% | | 0.627 | -0.004 on probe 3 |\n| Probe 5: + UltraBERT 10%, ATLAS 10% on unsure rows | | 0.631 | 0 on probe 3 |\n| Probe 6: peer correction in place of Sinkhorn | | TODO | TODO on probe 3 |\n| Probe 7: + our own Qwen3-8B LoRA | TODO | TODO | TODO |\n| Final picks | | TODO | private TODO |\n\n## One thing that didn't work, and why\n\nProbe 4 added Zbigniew Kirchner's in-style Qwen3-8B LoRA at 10%. It scores 0.574 on its own, a strong model, so we expected a small gain. The LB fell from 0.631 to 0.627.\n\nHis model was trained on in-style pool passages much like the ones in our SVC's pool. So on the passages our blend gets wrong, it is likely wrong in the same way. Adding it pushed those errors further rather than fixing them. Probe 5 points the same way: UltraBERT and ATLAS lifted the public 0.620 notebook to 0.636, but added nothing to ours, so our SVC and prvsiyan's seeds already fix the rows they fix.\n\nThe lesson is that a member's own score says little about what it adds. What counts is whether its errors differ from the blend's, and we could not measure that, because published members have no out-of-fold predictions. We dropped Zbigniew's model and stopped adding published members.\n\n## What we'd try next\n\n- Out-of-fold predictions for our own LoRA, so that members can be judged on how much their errors overlap, not by LB probes.\n- A model from outside the Qwen family. Almost all published members are Qwen, and they agree on about 77% of top picks (#18).\n- Test-time pseudo-label heads and Gaussian group heads on the LoRA features: +0.005 on the LB for prvsiyan (#23, #24).\n- Folds cut to one to three passages per prompt, as in the test, to judge every test-time step.\n\nTODO: final pick and why, once probe 6 and our LoRA are scored.\n\n## AI assistance\n\nWe used Claude Code with Claude Opus 5.5 throughout, with agents for research tickets and parallel experiments. Every session is logged in `docs/ai-sessions.md`.\nEOF\npython3 - <<'EOF'\nimport re\ns=open('docs/report/report-draft.md').read()\ntables=sum(len(l.split()) for l in s.splitlines() if l.startswith('|'))\nprint('total',len(s.split()),'tables',tables,'prose',len(s.split())-tables)\nprint('emdash','—' in s)\nEOF"
}
```

**Tool result**

```
total 1197 tables 201 prose 996
emdash False
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && f=docs/report/report-draft.md && sed -i '' -e 's/84,470 passages from 20 open-licence datasets/84,470 passages from open-licence public datasets/' -e \"s/plus tokens per word under each lab's tokenizer (#34)/plus tokens per word under Qwen's and Gemma 4's tokenizers (#34)/\" -e 's/trained from scratch overnight with our own code on our pool, then host train (#43)/trained overnight with our own code, first on our pool, then on host train (#43)/' -e 's/| + witness features (on the weight 0.5 SVC) | about +0.003 | | +0.003 CV |/| + witness features, on the weight 0.5 SVC | 0.1656 alone | | about +0.003 CV |/' $f && grep -n \"open-licence\\|Gemma 4's\\|overnight with\\|witness features\" $f && git add -A docs/report && git commit -q -m \"docs(report): rewrite the draft with probes 1 to 5, the LoRA24 readout and what is ours\n\nPicks probe 4 as the failure and leaves TODO slots for probe 6,\nour own LoRA and the final pick.\n\nrefs #49\" && git push -q -u origin docs/report-final-49 2>&1 | tail -2"
}
```

**Tool result**

```
36:- The pool: 84,470 passages from open-licence public datasets, filtered, de-duplicated and tagged by source (#19, #26).
40:- The witness: Qwen3-4B-Base likelihood features plus tokens per word under Qwen's and Gemma 4's tokenizers (#34).
41:- Our own Qwen3-8B LoRA, trained overnight with our own code, first on our pool, then on host train (#43).
56:| + witness features, on the weight 0.5 SVC | 0.1656 alone | | about +0.003 CV |
remote:      https://github.com/Nabzx/name-that-llm/pull/new/docs/report-final-49        
remote:
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && gh pr create --base main --title \"docs(report): bring the report and ablation table up to probe 5 (#49)\" --body \"$(cat <<'EOF'\n## What this does\n\n- Brings log rows 13 to 17 (probes 2 to 5, UltraBERT check) over from `feat/own-lora`, word for word, so main has the full log.\n- `docs/report/ablation.md`: adds probes 1 to 5 with their LB and deltas, TODO rows for probe 6, our own LoRA and the final pick, and two new lines in \"What the table says\".\n- `docs/report/report-draft.md`: rewrites the draft with a step-by-step section on how the LoRA24 readout works, an exact list of what is ours, one combined CV and LB ablation table, probe 4 (Zbigniew) as the failure, and updated next steps.\n\nrefs #49, refs #48\n\n## Why\n\nThe organisers allow published weights if the report shows we understand them and what we built on top. The draft still had TODO for every LB number. It now needs only Sunday's numbers.\n\n## Result\n\nAbout 1,000 words of prose plus about 200 in tables, so it should fit two pages. TODO left: probe 6, our own LoRA (CV and LB), the final two picks and the private score.\n\n## Checklist\n\n- [x] Numbers match `docs/experiments/LOG.md` rows 1 to 17\n- [x] British English, no em dashes\n- [x] Every published member named is in `CITATIONS.md`\n- [ ] Fill the TODO slots on Sunday afternoon\n\n## Test plan\n\n- Word count: 996 prose words, 201 in tables.\n- `grep '—'` on both files finds nothing.\n- Read the rendered Markdown on GitHub.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\n)\""
}
```

**Tool result**

```
Exit code 1
/bin/bash: eval: line 59: unexpected EOF while looking for matching `''
/bin/bash: eval: line 65: syntax error: unexpected end of file
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && B=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad/pr.md && cat > $B <<'EOF'\n## What this does\n\n- Brings log rows 13 to 17 (probes 2 to 5, UltraBERT check) over from `feat/own-lora`, word for word, so main has the full log.\n- `docs/report/ablation.md`: adds probes 1 to 5 with their LB and deltas, TODO rows for probe 6, our own LoRA and the final pick, and two new lines in \"What the table says\".\n- `docs/report/report-draft.md`: rewrites the draft with a step-by-step section on how the LoRA24 readout works, an exact list of what is ours, one combined CV and LB ablation table, probe 4 (Zbigniew) as the failure, and updated next steps.\n\nrefs #49, refs #48\n\n## Why\n\nThe organisers allow published weights if the report shows we understand them and what we built on top. The draft still had TODO for every LB number. It now needs only Sunday's numbers.\n\n## Result\n\nAbout 1,000 words of prose plus about 200 in tables, so it should fit two pages. TODO left: probe 6, our own LoRA (CV and LB), the final two picks and the private score.\n\n## Checklist\n\n- [x] Numbers match `docs/experiments/LOG.md` rows 1 to 17\n- [x] British English, no em dashes\n- [x] Every published member named is in `CITATIONS.md`\n- [ ] Fill the TODO slots on Sunday afternoon\n\n## Test plan\n\n- Word count: 996 prose words, 201 in tables.\n- A search for em dashes on both files finds nothing.\n- Read the rendered Markdown on GitHub.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\ngh pr create --base main --title \"docs(report): bring the report and ablation table up to probe 5 (#49)\" --body-file $B"
}
```

**Tool result**

```
https://github.com/Nabzx/name-that-llm/pull/94
```

## Claude (2026-10-25 00:45:11 UTC)

PR is open; binding it in the app now.

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
    "number": 94,
    "url": "https://github.com/Nabzx/name-that-llm/pull/94",
    "repo": "Nabzx/name-that-llm",
    "host": "github.com",
    "title": "docs(report): bring the report and ablation table up to probe 5 (#49)",
    "state": "open",
    "draft": false,
    "base": "main",
    "head": "docs/report-final-49"
  },
  "checks": {
    "available": true,
    "passing": 0,
    "failing": 0,
    "pending": 1,
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

I opened [PR #94](https://github.com/Nabzx/name-that-llm/pull/94) on branch `docs/report-final-49`, cut from main in its own worktree at `~/Random/nthl-report`. It brings the report draft and ablation table up to probe 5 and leaves marked TODO slots for Sunday. The one CI check is still running; I didn't bind Auto-fix.

The report is 996 words of prose plus 201 in tables, so it should fit two pages. Neither file has an em dash. There are four commits with no co-author trailers, as `CLAUDE.md` asks.

- **Log:** rows 13 to 17 copied word for word from `feat/own-lora` into `docs/experiments/LOG.md`, so main now has the full log.
- **`docs/report/ablation.md`:** the probe table now has the public LB and deltas for probes 1 to 5. It notes that probes 3 and 5 each changed two things at once. There are TODO rows for probe 6, our own LoRA and the final pick, and two new lines in "What the table says": the LB gains came from how we combine and balance, not from adding members, and gains of 0.004 or less are close to noise.
- **`docs/report/report-draft.md`:**
  - **Readout:** the LoRA24 readout in six concrete steps, with the label-order and end-token traps.
  - **What is ours:** an exact list (loader, its checks, pool, SVC, half Sinkhorn, peer correction, witness, our own LoRA, blend experiments), then what we used as released.
  - **Ablation:** one table with the CV and LB results together.
  - **Failure:** I picked probe 4 (Zbigniew). It's the most instructive because a strong model at 0.574 still cut the LB from 0.631 to 0.627, and probe 5's tie supports the same reading. The lesson: what counts is whether a member's errors differ from the blend's, and we can't measure that without out-of-fold predictions.
  - **Next steps:** updated, leading with out-of-fold predictions for our own LoRA.

**Still TODO on Sunday:**
- Probe 6 LB and its delta against probe 3, in both tables.
- Our own LoRA: its CV, LB and delta (probe 7 row).
- The final two picks, why we chose them, and the private score.
- The draft says "trained overnight" for our LoRA. Check that is still true once training finishes.

**Follow-ups:**
- I didn't add a row for this session to `docs/ai-sessions.md`, which `CLAUDE.md` requires. The parent session should add it with the rest.
- When `feat/own-lora` merges, its log rows 13 to 17 match mine word for word, so they should merge cleanly.

## Nabil (2026-10-25 00:45:26 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (working); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Important: work in your own git worktree, not in ~/Random/research-arm-kaggle itself (the main session switches branches there). First run: git -C ~/Random/research-arm-kaggle fetch -q && git -C ~/Random/research-arm-kaggle worktree add ~/Random/rak-report -b docs/report-probes origin/main, then do everything inside ~/Random/rak-report. Get LOG.md rows 13 to 17 with: git -C ~/Random/rak-report checkout origin/feat/own-lora -- docs/experiments/LOG.md (after the fetch). Never run git checkout or switch branches in ~/Random/research-arm-kaggle.
</cross-session-message>

## Claude (2026-10-25 00:45:32 UTC)

I already did this in a separate worktree, just under different names, so I'm not redoing it. The work is in [PR #94](https://github.com/Nabzx/name-that-llm/pull/94), which is open.

- **Worktree:** `~/Random/nthl-report`, branch `docs/report-final-49`, made from `origin/main` after a fetch. The names differ from your `rak-report` and `docs/report-probes`.
- **Main checkout:** I never ran checkout or switched branches in `~/Random/research-arm-kaggle`. It is still on `feat/probe6-peer`. I only ran `fetch` and `worktree add` there.
- **LOG.md:** I applied the `origin/feat/own-lora` diff to the file, which comes to the same thing as your checkout command: rows 13 to 17 copied word for word, nothing else changed.

If you'd rather have the branch named `docs/report-probes`, I can rename it and repoint the PR.

## Nabil (2026-10-25 10:06:53 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (done, PR #95); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (idle); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (working); hard-pairs = local_189a6d27-6360-4f5e-ae99-73795f412148 (working). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Please update the report draft (PR #94, your worktree ~/Random/nthl-report) with everything since probe 5. Fetch first; LOG.md rows 18 to 25 are on origin/feat/own-lora (git checkout origin/feat/own-lora -- docs/experiments/LOG.md).

New results:
- Probe 6, peer correction in place of half Sinkhorn: 0.622 (-0.009). Good candidate for "one thing that didn't work": it helped avazaku's weaker blend (+0.010) but removes class quotas, and on a strong blend the predictions drift to favoured classes.
- Our own Qwen3-8B LoRA24, trained on Kaggle overnight (4.5 h): stage 1 alone scores 0.519 on host train (honest, it never saw train), final 0.809 in sample. Probe 7 = probe 3 + ours at 10%: 0.631 (tie).
- SVC sweep: C 0.25 0.4775, C 1.0 0.4801, min_df 3 0.4841, so our settings (0.4838) were already best.
- Centred SVC twin (our version of the public stylegram): +0.003 ± 0.013 OOF, noise; replacing the plain SVC on grouped rows loses 0.13.
- Shared-prompt Gaussian group head (our own code after prvsiyan's public method) on the honest stage-1 features: +0.058 OOF at weight 1.0 (sd 0.013) on test-like groups. Biggest measured gain. Probe 8 (probe 7 + this head at 0.4 on grouped rows) is in its test run; leave a TODO for its score.
- Still running, leave TODO slots: 27B full-text classifier check, 122k-pool SVC, ModernBERT member, hard-pair specialists.

The organisers' brief, follow it exactly: Report max 2 pages, PDF. Contents: your starting point and approach; your hypotheses (what you thought would separate the models, and why); an ablation table showing what each change added; one thing that didn't work, and why; what you'd try next. Also keep the short section showing we understand the published LoRA24 weights and what we built on top (the organisers asked for that). Add a short "How we used AI" paragraph (Claude Code, logs in ai-sessions/). Keep Nabil's tone (British English, plain words, no em dashes). Then check it renders to a PDF of at most 2 pages (pandoc or similar locally is fine; say what tool you used) and commit the PDF as docs/report/report.pdf. Push and update the PR. Do not submit anything to Kaggle.
</cross-session-message>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git fetch -q && git status -sb | head -3 && git checkout origin/feat/own-lora -- docs/experiments/LOG.md && git diff --cached --stat && sed -n '/^| 18 /,$p' docs/experiments/LOG.md; which pandoc xelatex pdflatex weasyprint wkhtmltopdf; ls /Applications | grep -i chrome"
}
```

**Tool result**

```
## docs/report-final-49...origin/docs/report-final-49
 docs/experiments/LOG.md | 8 ++++++++
 1 file changed, 8 insertions(+)
| 18 | 2026-10-25 | #46 | **Probe 6.** Probe 3 members with avazaku's peer correction (leave-one-out SVC margin removal in topic clusters) in place of half Sinkhorn. Notebook `nthl-submit` v9 | n/a | n/a | 0.622 | -0.009. Without class quotas the blend drifts towards the classes it already favours. avazaku saw +0.010 on a weaker blend and -0.001 on the full stack, so the gain does not carry to a strong blend. Keep half Sinkhorn. |
| 19 | 2026-10-25 | #43 | Our own Qwen3-8B LoRA24, trained on Kaggle (`nthl-train-lora` v1, 4.5 h): stage 1 on a balanced pool sample, stage 2 two epochs on host train. Checked with `nthl-check-ours` v1 | stage 1 on train **0.519** (honest), final 0.809 in-sample | n/a | not submitted | Stage 1 never saw train, so 0.519 is a fair out-of-sample number, above our SVC (0.484) but below Nishant's 14B after stage 1 (0.572). Weakest on glm-5.3, glm-5.3-flash (0.25) and claude-fable-5 (0.24). End token 151643 as trained. Ready to probe at 10%. |
| 20 | 2026-10-25 | #43 | **Probe 7.** Probe 3 plus our own Qwen3-8B LoRA24 at a fixed 10% (main 80%, prvsiyan 10%, ours 10%), half Sinkhorn. Notebook `nthl-submit` v10 | n/a | n/a | 0.631 | No change against probe 3. Our model runs cleanly and adds no harm, but at 10% its 0.519 view does not move the public score. Same pattern as UltraBERT and ATLAS: extra members at 10% sit inside the noise of a 0.631 blend. |
| 21 | 2026-10-25 | #63 | Row 11 with C 0.25. Notebook `nthl-svc-pool-w025-c025` v1, CPU 60 min | 0.4775 | -0.006 | not submitted | Too much regularisation. |
| 22 | 2026-10-25 | #63 | Row 11 with C 1.0. Notebook `nthl-svc-pool-w025-c1` v1, CPU 70 min | 0.4801 | -0.004 | not submitted | Too little. C 0.5 stays best. |
| 23 | 2026-10-25 | #63 | Row 11 with min_df 3. Notebook `nthl-svc-pool-w025-mindf3` v1, CPU | 0.4841 | +0.0003 | not submitted | Noise. Our SVC settings are at their best. |
| 24 | 2026-10-25 | #46 | Prompt-centred SVC twin (the public stylegram idea, our own code): host train centred within each prompt, test-like groups (1 to 5 passages per prompt, clustered with every test text), added to the row 11 OOF at weights 0.1 to 2. Notebook `nthl-svc-centred-check` v1, CPU 6 min, 10 draws | twin alone 0.327 on grouped rows (plain 0.481) | best +0.003 at weight 1.0, sd 0.0125 | not submitted | Not a clear gain: the spread across draws is four times the best mean gain, and replacing the plain SVC on grouped rows loses 0.13. The twin learns from 2,160 host rows only (the pool has no prompt ids to centre on), so it is much weaker than the pool SVC. Not worth a submission. |
| 25 | 2026-10-25 | #46 | **Shared-prompt Gaussian group head** (our own code after prvsiyan's public method) on the honest stage-1 features of our own LoRA: fitted per frozen fold, test-like groups (1 to 5 passages per prompt, clustered with every test text), added to the plain h
... (597 more characters in the raw transcript)
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git commit -q -m \"docs(log): bring in rows 18 to 25 from feat/own-lora\n\nrefs #48\" && python3 - <<'EOF'\np='docs/report/ablation.md'\ns=open(p).read()\n# SVC settings rows\ns=s.replace(\"\"\"| 11 | Pool weight 0.25 | **0.4838** | +0.001 | **+0.135** |\n\"\"\",\"\"\"| 11 | Pool weight 0.25 | **0.4838** | +0.001 | **+0.135** |\n| 21 | C 0.25 in place of 0.5 | 0.4775 | -0.006 | |\n| 22 | C 1.0 in place of 0.5 | 0.4801 | -0.004 | |\n| 23 | min_df 3 | 0.4841 | +0.0003 | |\n\"\"\")\ns=s.replace(\"Row 2 is a reproduction check: the same probabilities to the last digit.\",\n\"Row 2 is a reproduction check: the same probabilities to the last digit. Rows 21 to 23 are against row 11 and show our settings were already at their best.\")\ns=s.replace(\"\"\"| 7 | Nishant Dahal's Qwen3-14B checkpoint through our loader | n/a | in-sample 0.858 | not a score |\n\"\"\",\"\"\"| 7 | Nishant Dahal's Qwen3-14B checkpoint through our loader | n/a | in-sample 0.858 | not a score |\n| 24 | Prompt-centred SVC twin added to the row 11 OOF, test-like groups of 1 to 5 per prompt, 10 draws | Row 11 | twin alone 0.327 on grouped rows | +0.003 at weight 1.0, sd 0.013: noise |\n| 19 | Our own Qwen3-8B LoRA24, stage 1 only (never saw host train) | Row 11 | **0.519** on host train | +0.035 over our SVC |\n| 25 | Shared-prompt Gaussian group head on our LoRA's stage-1 features, test-like groups, 10 draws | plain head 0.524 on grouped rows | 0.574 alone | **+0.058 at weight 1.0**, sd 0.013 |\n\"\"\")\ns=s.replace(\"Row 7 checks the loader only.\",\"Row 19 is our only honest score for a LoRA: stage 1 trained on the pool alone, so host train is unseen. The final model, after stage 2 on host train, scores 0.809 in-sample, which is not a score. Row 7 checks the loader only.\")\ns=s.replace(\"\"\"| TODO | 6 | Peer correction in place of Sinkhorn | Probe 3 | TODO | TODO |\n| TODO | 7 | Our own Qwen3-8B LoRA as a member | TODO | TODO | TODO |\n\"\"\",\"\"\"| 18 | 6 | Peer correction in place of half Sinkhorn | Probe 3 | 0.622 | -0.009 |\n| 20 | 7 | Add our own Qwen3-8B LoRA24 at 10% (main 80%, prvsiyan 10%) | Probe 3 | 0.631 | 0 |\n| TODO | 8 | Add the Gaussian group head at 0.4 on grouped rows | Probe 7 | TODO | TODO |\n\"\"\")\ns=s.replace(\"Row 16 is the UltraBERT loader check: in-sample 0.966 on host train, not a score.\",\n\"\"\"Row 16 is the UltraBERT loader check: in-sample 0.966 on host train, not a score.\n\n## Still running\n\n| Check | Result |\n|---|---|\n| 27B full-text classifier | TODO |\n| SVC on the 122k pool | TODO |\n| ModernBERT member | TODO |\n| Hard-pair specialists | TODO |\"\"\")\ns=s.replace(\"prvsiyan's seeds shared +0.002, Zbigniew's in-style model -0.004, UltraBERT and ATLAS nothing.\",\n\"prvsiyan's seeds shared +0.002, Zbigniew's in-style model -0.004, UltraBERT, ATLAS and our own LoRA nothing. Dropping the class quotas (peer correction) cost 0.009.\\n- The one large gain on our own folds since the pool is the Gaussian group head, +0.058 on test-like groups.\")\nopen(p,'w').write(s)\nEOF\ngrep -c TODO docs/report/ablation.md; git commit -qam \"docs(report): add rows 18 to 25 and TODO slots for the running checks to the ablation table\n\nrefs #48\" && echo ok"
}
```

**Tool result**

```
6
ok
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && cat > docs/report/report-draft.md <<'EOF'\n# Name That LLM\n\nNabil Shah, solo entry, Imperial Research Arm qualifier. Numbers come from `docs/experiments/LOG.md`. **TODO** marks a number still to come.\n\n## Starting point and approach\n\nWe wrote our own code from scratch. The first model was a four-view n-gram LinearSVC on host train only, scored on frozen `GroupKFold` folds over `prompt_id`: 0.3485 balanced accuracy, in line with public train-only SVCs.\n\nWe then worked in two branches. Branch A is a published spine we run through our own loader: Nishant Dahal's two-stage LoRA blend, 0.620 on the public LB. Branch B is what we can check on our own folds: our SVC, our own LoRA and test-time heads. Published members have no out-of-fold predictions, so for them the LB is the only test. We probed it one change at a time and kept a change only if our CV or a research note also backed it.\n\n## Hypotheses\n\n- **Data beats model size.** Public work showed 8B and 14B backbones tied on the same data, while a bigger pool gave +0.028 on the LB.\n- **Labs differ in surface style, siblings do not.** Our baseline put the right lab first 0.57 of the time, and half its errors stayed inside the right lab. Hardest: gemini-3.7-flash against 3.8-flash, and glm-5.3 read as claude-opus-5.\n- **The prompt is louder than the author.** Passages that answer the same prompt share content, so taking out the prompt should leave mostly style.\n- **The test mix is unequal.** Balancing towards 1/18 should help, but at half strength, since we cannot see the test's skew.\n\n## The published LoRA24 weights and what we built on them\n\nEach LoRA24 checkpoint turns a passage into 18 probabilities like this. Tokenise with no special tokens, cut at 447 and append the end token from the checkpoint's `meta.json` (151643), not the tokenizer's (151645). Keep only the first 27 of Qwen3-14B's 40 decoder layers, with the final norm off. Merge the rank-16 adapter into each attention and MLP weight: W + B·A × 2. Take a masked mean of the hidden states at 8 depths, leaving out the first token. LayerNorm each depth, join them into one 40,960-wide vector and apply a linear head. A second, centred head reads the same vector minus its topic cluster's mean. The traps are silent: a wrong label order puts claude-fable-5's probabilities under another Claude model, and the wrong end token changed 14% of top picks in Nishant's tests.\n\n**What is ours:**\n\n- The loader (`nthl.lora24`), written from that recipe. It asserts the end token and reorders label columns by `meta.json`. Nishant's Qwen3-14B scored 0.858 in-sample through it, with every Claude label right.\n- The pool: 84,470 passages from open-licence public datasets, filtered, de-duplicated and tagged by source.\n- Our SVC and its tuning, 0.3485 to 0.4838 CV.\n- Half-strength Sinkhorn, and peer correction re-implemented from avazaku's idea.\n- A likelihood witness from Qwen3-4B-Base, about +0.003 on the SVC.\n- Our own Qwen3-8B LoRA24, trained on Kaggle with our own code in 4.5 hours: stage 1 on a balanced pool sample, stage 2 on host train.\n- A shared-prompt Gaussian group head, our own code after prvsiyan's public method.\n- Every blend experiment, probes 1 to 8.\n\n**Used as released:** Nishant Dahal's Qwen3-14B, Qwen2.5-14B and DeBERTa-v3-large with his blend weights, and prvsiyan's two Qwen3-8B seeds. All sources are in `CITATIONS.md`.\n\n## Ablation\n\nCV is on our frozen folds. Δ is against the row before, unless named. The full table is in `docs/report/ablation.md`.\n\n| Change | CV | Public LB | Δ |\n|---|---|---|---|\n| SVC, host train only | 0.3485 | | |\n| + whole pool, weight 1 | 0.4634 | | +0.115 |\n| pool weight 0.5 | 0.4782 | | +0.015 |\n| pool weight 0.25 (C 0.25, C 1.0, min_df 3 no better) | 0.4838 | | +0.006 |\n| Prompt-centred SVC twin, test-like groups | | | +0.003, sd 0.013 |\n| Our Qwen3-8B LoRA, stage 1 only, on unseen host train | 0.519 | | +0.035 on the SVC |\n| + Gaussian group head on its features, test-like groups | | | **+0.058**, sd 0.013 |\n| Probe 1: Nishant's spine with our SVC, full Sinkhorn | | 0.626 | +0.006 on his 0.620 |\n| Probe 2: half Sinkhorn | | 0.629 | +0.003 |\n| Probe 3: + prvsiyan's seeds at 10%, SVC weight 0.25 | | **0.631** | +0.002 |\n| Probe 4: + Zbigniew Kirchner's in-style Qwen3-8B at 10% | | 0.627 | -0.004 on 3 |\n| Probe 5: + UltraBERT 10%, ATLAS 10% on unsure rows | | 0.631 | 0 on 3 |\n| Probe 6: peer correction in place of half Sinkhorn | | 0.622 | -0.009 on 3 |\n| Probe 7: + our own LoRA at 10% | | 0.631 | 0 on 3 |\n| Probe 8: + Gaussian head at 0.4 on grouped rows | | TODO | TODO on 7 |\n| Still running: 27B full-text classifier, 122k-pool SVC, ModernBERT, hard-pair specialists | TODO | TODO | TODO |\n| Final picks | | TODO | private TODO |\n\nData did most of the work on CV. On the LB, changing how we balance moved the score most. Adding members at 10% moved it little or made it worse.\n\n## One thing that didn't work, and why\n\nProbe 6 swapped half Sinkhorn for peer correction. Within each topic cluster, every passage loses a shrunk share of the other members' mean SVC lean, so a shared prompt stops pulling them all the same way. avazaku saw +0.010 from it on a weaker blend. On ours the LB fell from 0.631 to 0.622, our worst drop.\n\nPeer correction removes the prompt's lean, but it has no class quotas. Sinkhorn pushes every class towards a fair share of the test. Without it, a strong blend drifts towards the classes it already favours, and balanced accuracy punishes every class it starves. On a weak blend the prompt's lean is the bigger error, so peer correction wins. On a strong one the class drift is bigger. A step that helps one blend is not a free gain on another. We kept half Sinkhorn.\n\n## What we'd try next\n\n- Judge members by how much their errors overlap, using out-of-fold predictions from our own LoRA. Our LoRA, UltraBERT and ATLAS all tied at 10%, and Zbigniew's model hurt: a strong model whose errors match the blend's adds nothing.\n- Train a non-Qwen model. Almost all published members are Qwen, and they agree on about 77% of top picks.\n- Build the folds from one to five passages per prompt, as in the test, to judge every test-time step.\n\nTODO: final pick and why.\n\n## How we used AI\n\nWe used Claude Code (Claude Opus 5.5) throughout: one main session to plan and review, with side sessions and agents for research notes and parallel experiments, each in its own branch. Every session is listed in `docs/ai-sessions.md` and exported to `ai-sessions/`. AI wrote most of the code and notes under our review. Every model fit ran on Kaggle, and no test passage was labelled by hand or by an LLM.\nEOF\npython3 -c \"\ns=open('docs/report/report-draft.md').read();t=sum(len(l.split()) for l in s.splitlines() if l.startswith('|'));print(len(s.split())-t,t,'—' in s)\""
}
```

**Tool result**

```
922 277 False
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && gh pr list --state merged -L 60 --json mergedBy -q '.[].mergedBy.login' | sort | uniq -c"
}
```

**Tool result**

```
  31 Nabzx
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && f=docs/report/report-draft.md && sed -i '' -e \"s/and prvsiyan's two Qwen3-8B seeds. All sources/and prvsiyan's two Qwen3-8B seeds. In probes 4 and 5 only: Zbigniew Kirchner's in-style Qwen3-8B, UltraBERT and ATLAS. All sources/\" -e \"s/AI wrote most of the code and notes under our review./Claude wrote much of the code and the notes. Nabil set the direction, chose every probe and reviewed and merged every pull request./\" $f && grep -c \"probes 4 and 5 only\\|merged every\" $f\ncat > /private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad/pdf.css <<'EOF'\n@page { size: A4; margin: 14mm 15mm; }\nbody { font-family: \"Helvetica Neue\", Arial, sans-serif; font-size: 9.2pt; line-height: 1.3; color: #111; }\nh1 { font-size: 14pt; margin: 0 0 2pt; }\nh2 { font-size: 10.5pt; margin: 7pt 0 2pt; }\np, ul { margin: 2pt 0 3pt; }\nul { padding-left: 14pt; }\nli { margin: 0 0 1pt; }\ntable { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 3pt 0; }\nth, td { border-bottom: 0.5pt solid #bbb; padding: 1pt 4pt; text-align: left; }\ncode { font-size: 8.2pt; }\nEOF\necho ok"
}
```

**Tool result**

```
2
ok
```

## Claude (2026-10-25 10:08:07 UTC)

Report text is rewritten with rows 18 to 25; now rendering it to PDF and checking the page count.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && S=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad && pandoc docs/report/report-draft.md -s --embed-resources -c $S/pdf.css --metadata title=\"Name That LLM report\" -V pagetitle=\"Name That LLM\" -o $S/report.html 2>&1 | grep -v \"title\" ; sed -i '' 's#<header id=\"title-block-header\">.*</header>##' $S/report.html; \"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=docs/report/report.pdf $S/report.html 2>/dev/null; ls -la docs/report/report.pdf; python3 -c \"\nimport re;d=open('docs/report/report.pdf','rb').read();print('pages',len(re.findall(rb'/Type\\s*/Page[^s]',d)))\""
}
```

**Tool result**

```
-rw-r--r--@ 1 nabilshah  staff  113389 Oct 25 10:08 docs/report/report.pdf
pages 2
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && grep -c 'title-block-header' /private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad/report.html; S=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad; qlmanage -t -s 1200 -o $S docs/report/report.pdf >/dev/null 2>&1; ls $S"
}
```

**Tool result**

```
1
pdf.css
pr.md
report.html
report.pdf.png
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && S=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad && pandoc docs/report/report-draft.md -s --embed-resources -c $S/pdf.css -M pagetitle=\"Name That LLM\" -o $S/report.html && grep -c title-block-header $S/report.html; \"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=docs/report/report.pdf $S/report.html 2>/dev/null; python3 -c \"\nimport re;d=open('docs/report/report.pdf','rb').read();print('pages',len(re.findall(rb'/Type\\s*/Page[^s]',d)))\""
}
```

**Tool result**

```
0
pages 2
```

**Tool call: Read**

```json
{
 "file_path": "/Users/nabilshah/Random/nthl-report/docs/report/report.pdf",
 "pages": "1-2"
}
```

**Tool result**

```
PDF pages extracted: 2 page(s) from /Users/nabilshah/Random/nthl-report/docs/report/report.pdf (109.8KB)


```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && S=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad && f=docs/report/report-draft.md && sed -i '' -e 's/| Prompt-centred SVC twin, test-like groups | | | +0.003, sd 0.013 |/| Prompt-centred SVC twin, test-like groups | | | +0.003 on the SVC, sd 0.013 |/' -e 's/| + Gaussian group head on its features, test-like groups | | | \\*\\*+0.058\\*\\*, sd 0.013 |/| + Gaussian group head on its features, test-like groups | | | **+0.058** on its plain head, sd 0.013 |/' $f && grep -c \"on its plain head\\|+0.003 on the SVC, sd\" $f\ncp $S/pdf.css docs/report/report.css && echo 'td:first-child, th:first-child { width: 55%; }' >> docs/report/report.css\ncat > scripts/report_pdf.sh <<'EOF'\n#!/bin/sh\n# Render docs/report/report-draft.md to docs/report/report.pdf (A4, must stay at 2 pages).\n# Needs pandoc and Google Chrome.\nset -e\ncd \"$(dirname \"$0\")/..\"\ntmp=$(mktemp -d)\npandoc docs/report/report-draft.md -s --embed-resources -c docs/report/report.css -M pagetitle=\"Name That LLM\" -o \"$tmp/report.html\"\n\"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\" --headless=new --disable-gpu --no-pdf-header-footer \\\n  --print-to-pdf=docs/report/report.pdf \"$tmp/report.html\" 2>/dev/null\npython3 -c \"import re; d = open('docs/report/report.pdf', 'rb').read(); print('pages:', len(re.findall(rb'/Type\\s*/Page[^s]', d)))\"\nEOF\nchmod +x scripts/report_pdf.sh && ./scripts/report_pdf.sh"
}
```

**Tool result**

```
2
pages: 2
```

**Tool call: Read**

```json
{
 "file_path": "/Users/nabilshah/Random/nthl-report/docs/report/report.pdf",
 "pages": "2"
}
```

**Tool result**

```
PDF pages extracted: 1 page(s) from /Users/nabilshah/Random/nthl-report/docs/report/report.pdf (109.9KB)

```

## Claude (2026-10-25 10:09:05 UTC)

Fixing the table column widths, then committing and pushing.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && sed -i '' 's/td:first-child, th:first-child { width: 55%; }/td:nth-child(1) { width: 50%; } td:nth-child(2), td:nth-child(3) { width: 11%; white-space: nowrap; } td:nth-child(4) { width: 28%; }/' docs/report/report.css && sed -i '' 's/table { border-collapse: collapse;/table { table-layout: fixed; border-collapse: collapse;/' docs/report/report.css && ./scripts/report_pdf.sh"
}
```

**Tool result**

```
pages: 2
```

**Tool call: Read**

```json
{
 "file_path": "/Users/nabilshah/Random/nthl-report/docs/report/report.pdf",
 "pages": "2"
}
```

**Tool result**

```
PDF pages extracted: 1 page(s) from /Users/nabilshah/Random/nthl-report/docs/report/report.pdf (109.9KB)

```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && grep -o '<col[^>]*>' /dev/null; sed -i '' '$d' docs/report/report.css; tail -1 docs/report/report.css; echo 'col:nth-child(1) { width: 50% !important; } col:nth-child(2), col:nth-child(3) { width: 11% !important; } col:nth-child(4) { width: 28% !important; }' >> docs/report/report.css && ./scripts/report_pdf.sh"
}
```

**Tool result**

```
code { font-size: 8.2pt; }
pages: 2
```

**Tool call: Read**

```json
{
 "file_path": "/Users/nabilshah/Random/nthl-report/docs/report/report.pdf",
 "pages": "1-2"
}
```

**Tool result**

```
PDF pages extracted: 2 page(s) from /Users/nabilshah/Random/nthl-report/docs/report/report.pdf (109.8KB)


```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git add docs/report/report-draft.md && git commit -q -m \"docs(report): add probes 6 and 7, our LoRA, the SVC sweep and the group heads, with probe 6 as the failure\n\nrefs #49\" && git add scripts/report_pdf.sh docs/report/report.css && git commit -q -m \"docs(report): add a script that renders the report to an A4 PDF\n\nrefs #49\" && git add docs/report/report.pdf && git commit -q -m \"docs(report): add the rendered two-page PDF\n\nrefs #49\" && git push -q && git log --oneline origin/main..HEAD"
}
```

**Tool result**

```
6b339ea docs(report): add the rendered two-page PDF
5daf47c docs(report): add a script that renders the report to an A4 PDF
0618065 docs(report): add probes 6 and 7, our LoRA, the SVC sweep and the group heads, with probe 6 as the failure
afa7e0c docs(report): add rows 18 to 25 and TODO slots for the running checks to the ablation table
b7030c3 docs(log): bring in rows 18 to 25 from feat/own-lora
4da62d2 docs(report): rewrite the draft with probes 1 to 5, the LoRA24 readout and what is ours
d301624 docs(report): fix the public LB size wording
d2cacee docs(report): add probes 2 to 5 with deltas and TODO rows for probe 6, our LoRA and the final pick
c6be233 docs(log): bring in probes 2 to 5 and the UltraBERT check from feat/own-lora
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && B=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad/pr.md && cat > $B <<'EOF'\n## What this does\n\n- Brings log rows 13 to 25 over from `feat/own-lora`, word for word, so main has the full log.\n- `docs/report/ablation.md`: probes 1 to 7 with their LB and deltas, the SVC sweep (C 0.25, C 1.0, min_df 3), the centred SVC twin, our own LoRA's stage-1 score and the Gaussian group head. TODO rows for probe 8, the final pick and the four checks still running (27B full-text classifier, 122k-pool SVC, ModernBERT, hard-pair specialists).\n- `docs/report/report-draft.md`: follows the organisers' brief in order (starting point and approach, hypotheses, ablation, one failure, next steps), plus the LoRA24 readout and an exact list of what is ours, and a short \"How we used AI\" paragraph. The failure is now probe 6: peer correction helped avazaku's weaker blend but cost us 0.009, because it drops the class quotas.\n- `scripts/report_pdf.sh` and `docs/report/report.css`: render the draft to an A4 PDF with pandoc and headless Chrome and print the page count.\n- `docs/report/report.pdf`: the rendered report.\n\nrefs #49, refs #48\n\n## Why\n\nThe report is half the score and must be at most two pages. It now needs only Sunday's numbers.\n\n## Result\n\n2 pages. Page 2 is about 40% empty, so there is room for the final numbers. About 920 words of prose plus about 280 in the table.\n\n## Checklist\n\n- [x] Numbers match `docs/experiments/LOG.md` rows 1 to 25\n- [x] British English, no em dashes\n- [x] Every published member named is in `CITATIONS.md`\n- [x] PDF is 2 pages\n- [ ] Fill the TODO slots, re-run `scripts/report_pdf.sh` and commit the PDF again\n\n## Test plan\n\n- `./scripts/report_pdf.sh` prints `pages: 2`.\n- Both pages checked by eye.\n- A search for em dashes on both Markdown files finds nothing.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\ngh pr edit 94 --title \"docs(report): bring the report up to probe 7 and render it to a two-page PDF (#49)\" --body-file $B"
}
```

**Tool result**

```
https://github.com/Nabzx/name-that-llm/pull/94
```

## Claude (2026-10-25 10:09:53 UTC)

I've pushed the update to [PR #94](https://github.com/Nabzx/name-that-llm/pull/94): the report now runs through probe 7 and the PDF, `docs/report/report.pdf`, comes out at 2 pages with about 40% of page 2 still free. I rendered it with pandoc and headless Chrome and checked both pages by eye. Nothing was submitted to Kaggle.

- **Log:** rows 18 to 25 come from `origin/feat/own-lora`, using the checkout command you gave.
- **Report text:** about 920 words plus the table, in the brief's order: starting point and approach, hypotheses, ablation, one failure, next steps. It keeps the LoRA24 readout and the list of what is ours, and ends with a "How we used AI" paragraph that points to `docs/ai-sessions.md` and `ai-sessions/`. There are no em dashes.
- **The failure:** probe 6. Peer correction gave avazaku +0.010 on a weaker blend but cost us 0.009, because without class quotas a strong blend drifts towards the classes it already favours.
- **What's new in the numbers:**
  - Our own LoRA scores 0.519 on host train after stage 1, which it never saw, and probe 7 tied at 0.631.
  - The SVC sweep found nothing better than our settings.
  - The centred SVC twin gave +0.003 with an sd of 0.013, which is noise.
  - The Gaussian group head gave +0.058, the biggest gain we measured.
  - Probe 4 (Zbigniew) is now a point under "What we'd try next".
- **`ablation.md`:** has the same new rows, plus a "Still running" table.

The AI paragraph says Nabil "set the direction, chose every probe and reviewed and merged every pull request". All 31 merged PRs were merged by Nabzx, but please confirm the wording before it goes in.

To rebuild the PDF after filling the TODOs, run this from the repo root; it prints the page count. I added the script and its stylesheet, `docs/report/report.css`, in this PR:
```bash
./scripts/report_pdf.sh
```

**Still TODO:**
- Probe 8's LB score and its delta against probe 7.
- The four checks still running: the 27B full-text classifier, the SVC on the 122k pool, ModernBERT and the hard-pair specialists. They share one row in the report table and each has its own row in `ablation.md`.
- The final picks, why we chose them, and the private score.
- No row for this session in `docs/ai-sessions.md` yet; the main session should add it.

## Nabil (2026-10-25 11:57:13 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (done, PR #95); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (done, PR #100); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (done, PR #99); hard-pairs = local_189a6d27-6360-4f5e-ae99-73795f412148 (done, PR #98). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Please update the report (PR #94) with these results, keeping the PDF under 2 pages (re-render and check). Fetch LOG.md from origin/feat/own-lora (rows 23 to 27).
- Probe 8 (probe 7 + shared-prompt Gaussian group head on our stage-1 features, weight 0.4 on grouped rows): 0.631, the fourth 0.631 in a row (probes 3, 5, 7, 8). Honest framing: the head is strong on its own view (+0.058 OOF, row 25) but at 0.4 inside the blend it flips too few rows to move the public slice.
- Probe 9 (probe 8 with Gaussian weight 1.0 + nuridudak's ModernBERT-large, 122k open pool, published OOF 0.569, at 10%): submitted 11:55, score due about 13:00. Leave one clearly marked TODO slot for it and for the final picks.
- Tested today, no submission: centred SVC twin +0.003 ± 0.013 (noise); hard-pair specialists +0.0026 (below the noise bar, PR #98); SVC with the 122k pool 0.4797 vs 0.4838 (worse, row 27); full-text Qwen3.8-27B honest held-out OOF 0.521, about 3 s per passage so 3.7 h on the test, too slow (PR #100); nuridudak encoders in-sample check 0.806 / 0.659 (PR #99).
- Next steps worth naming: a bigger pool needs retraining (122k stage 1 alone about 10 h on 2 T4s); 5-fold members for honest OOF blending; Gaussian decoding across all members jointly; organiser ruling on self-generated data (rule 2.6).
Keep the brief's five sections, the published-weights understanding section, and the AI-use paragraph. Nabil's tone, no em dashes. Push and update the PR.
</cross-session-message>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git fetch -q && git checkout origin/feat/own-lora -- docs/experiments/LOG.md && git diff --cached | grep '^[+-]|' | cut -c1-900"
}
```

**Tool result**

```
+| 26 | 2026-10-25 | #46 | **Probe 8.** Probe 7 plus the Gaussian group head on our stage-1 features at weight 0.4 on grouped test rows, temperature 2. Notebook `nthl-submit` v11 | n/a | n/a | 0.631 | No change against probe 7, the fourth 0.631 in a row. Strong on its own view (row 25), but at 0.4 inside the blend it flips too few rows to move the public slice. |
+| 27 | 2026-10-25 | #63 | Row 11 with nuridudak's 122k public pool (CC BY-SA 4.0) in place of our 84k pool. Notebook `nthl-svc-pool122k` v1, CPU 93 min | 0.4797 | -0.004 | not submitted | Worse for our SVC. More passages from more sources do not help the n-gram view; the audit's +0.021 was for an encoder. Keep our pool. |
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git commit -q -m \"docs(log): bring in rows 26 and 27 from feat/own-lora\n\nrefs #48\" && python3 - <<'EOF'\np='docs/report/ablation.md'\ns=open(p).read()\ndef rep(a,b):\n    global s\n    assert a in s, a\n    s=s.replace(a,b)\nrep(\"\"\"| 23 | min_df 3 | 0.4841 | +0.0003 | |\n\"\"\",\"\"\"| 23 | min_df 3 | 0.4841 | +0.0003 | |\n| 27 | nuridudak's 122k public pool in place of our 84k pool | 0.4797 | -0.004 | |\n\"\"\")\nrep(\"Rows 21 to 23 are against row 11 and show our settings were already at their best.\",\n\"Rows 21 to 23 and 27 are against row 11 and show our settings and our pool were already the best for the SVC.\")\nrep(\"\"\"| TODO | 8 | Add the Gaussian group head at 0.4 on grouped rows | Probe 7 | TODO | TODO |\n\"\"\",\"\"\"| 26 | 8 | Add the Gaussian group head at 0.4 on grouped rows | Probe 7 | 0.631 | 0 |\n| TODO | 9 | Gaussian head at 1.0, and nuridudak's ModernBERT-large (122k pool, published OOF 0.569) at 10% | Probe 8 | TODO | TODO |\n\"\"\")\nrep(\"\"\"## Still running\n\n| Check | Result |\n|---|---|\n| 27B full-text classifier | TODO |\n| SVC on the 122k pool | TODO |\n| ModernBERT member | TODO |\n| Hard-pair specialists | TODO |\"\"\",\"\"\"## Checked, not submitted\n\n| Check | Result |\n|---|---|\n| Hard-pair specialists (#98) | +0.0026 OOF, below the noise bar |\n| Full-text Qwen3.8-27B classifier (#100) | 0.521 honest held-out OOF, but about 3 s per passage, so 3.7 h on the test: too slow |\n| nuridudak's encoders through our loader (#99) | in-sample 0.806 and 0.659: a loader check, not a score |\n| SVC on the 122k pool (row 27) | 0.4797 against 0.4838: worse |\"\"\")\nrep(\"UltraBERT, ATLAS and our own LoRA nothing.\",\"UltraBERT, ATLAS, our own LoRA and the Gaussian head at 0.4 nothing. Probes 3, 5, 7 and 8 all scored 0.631.\")\nopen(p,'w').write(s)\nEOF\ngit commit -qam \"docs(report): add probe 8, a TODO for probe 9 and the checks not submitted to the ablation table\n\nrefs #48\" && echo ok"
}
```

**Tool result**

```
ok
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && python3 - <<'EOF'\np='docs/report/report-draft.md'\ns=open(p).read()\ndef rep(a,b):\n    global s\n    assert a in s, a\n    s=s.replace(a,b)\nrep(\"Every blend experiment, probes 1 to 8.\",\"Every blend experiment, probes 1 to 9.\")\nrep(\"| pool weight 0.25 (C 0.25, C 1.0, min_df 3 no better) | 0.4838 | | +0.006 |\",\n\"| pool weight 0.25 (C 0.25, C 1.0, min_df 3 and the 122k pool no better) | 0.4838 | | +0.006 |\")\nrep(\"\"\"| + Gaussian group head on its features, test-like groups | | | **+0.058** on its plain head, sd 0.013 |\n\"\"\",\"\"\"| + Gaussian group head on its features, test-like groups | | | **+0.058** on its plain head, sd 0.013 |\n| Hard-pair specialists on the SVC | | | +0.0026, below the noise bar |\n| Full-text Qwen3.8-27B classifier, held out | 0.521 | | 3.7 h on the test, too slow |\n\"\"\")\nrep(\"\"\"| Probe 8: + Gaussian head at 0.4 on grouped rows | | TODO | TODO on 7 |\n| Still running: 27B full-text classifier, 122k-pool SVC, ModernBERT, hard-pair specialists | TODO | TODO | TODO |\n\"\"\",\"\"\"| Probe 8: + Gaussian head at 0.4 on grouped rows | | 0.631 | 0 on 7 |\n| **TODO** Probe 9: Gaussian head at 1.0, + nuridudak's ModernBERT-large at 10% | | **TODO** | **TODO** on 8 |\n\"\"\")\nrep(\"| Final picks | | TODO | private TODO |\",\"| **TODO** Final picks | | **TODO** | private **TODO** |\")\nrep(\"Data did most of the work on CV. On the LB, changing how we balance moved the score most. Adding members at 10% moved it little or made it worse.\",\n\"Data did most of the work on CV. On the LB, changing how we balance moved the score most. Probes 3, 5, 7 and 8 all scored 0.631: the Gaussian head is strong on its own view, but at 0.4 inside the blend it flips too few rows to move the public slice, and members at 10% do the same.\")\nold=s[s.index(\"## What we'd try next\"):s.index(\"## How we used AI\")]\nnew=\"\"\"## What we'd try next\n\n- Give every member of our own five-fold out-of-fold predictions, so the blend can be fitted and members judged by how much their errors overlap, not by LB probes. Our LoRA, UltraBERT and ATLAS tied at 10%, and Zbigniew's model hurt.\n- Retrain on a bigger pool. Swapping in the 122k pool did not help the SVC, and our LoRA would need about 10 hours of stage 1 on two T4s to use it.\n- Decode each shared-prompt group jointly across all members, not on one member's features.\n- Ask the organisers whether passages we generate ourselves count as external data under rule 2.6. If they do, we would generate in the host's style for the hardest pairs.\n\n**TODO:** final picks and why.\n\n\"\"\"\ns=s.replace(old,new)\nopen(p,'w').write(s)\nt=sum(len(l.split()) for l in s.splitlines() if l.startswith('|'));print(len(s.split())-t,t,'—' in s)\nEOF\n./scripts/report_pdf.sh"
}
```

**Tool result**

```
1016 325 False
pages: 2
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && sed -i '' 's/ If they do, we would generate in the host.s style for the hardest pairs\\.//' docs/report/report-draft.md && grep -n \"rule 2.6\" docs/report/report-draft.md && ./scripts/report_pdf.sh"
}
```

**Tool call: Read**

```json
{
 "file_path": "/Users/nabilshah/Random/nthl-report/docs/report/report.pdf",
 "pages": "1-2"
}
```

**Tool result**

```
74:- Ask the organisers whether passages we generate ourselves count as external data under rule 2.6.
pages: 2
```

**Tool result**

```
PDF pages extracted: 2 page(s) from /Users/nabilshah/Random/nthl-report/docs/report/report.pdf (113.9KB)


```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && f=docs/report/report-draft.md && sed -i '' -e 's/- Give every member of our own five-fold out-of-fold predictions, so the blend can be fitted and members judged/- Train our own members five-fold, so each has honest out-of-fold predictions, the blend can be fitted on them and members can be judged/' -e \"s/UltraBERT and ATLAS. All sources/UltraBERT and ATLAS. In probe 9: nuridudak's ModernBERT-large. All sources/\" $f && grep -c \"five-fold, so\\|probe 9: nuridudak\" $f && grep -q nuridudak CITATIONS.md && echo cited || echo NOT-CITED; ./scripts/report_pdf.sh"
}
```

**Tool result**

```
2
NOT-CITED
pages: 2
```

## Claude (2026-10-25 11:58:20 UTC)

Report re-rendered at 2 pages; checking whether nuridudak is cited on another branch.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && for b in $(git branch -r --format='%(refname:short)'); do git show $b:CITATIONS.md 2>/dev/null | grep -q nuridudak && echo $b; done"
}
```

**Tool result**

```
Exit code 1
origin/exp/gauss-heads
origin/exp/hard-pairs
origin/feat/encoders-122k
```

## Claude (2026-10-25 11:58:25 UTC)

It's cited on those branches, so the report is fine once they merge. Committing and pushing.

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git add docs/report/report-draft.md && git commit -q -m \"docs(report): add probe 8, a TODO for probe 9, the checks not submitted and new next steps\n\nrefs #49\" && git add docs/report/report.pdf && git commit -q -m \"docs(report): re-render the PDF, still two pages\n\nrefs #49\" && git push -q && B=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad/pr.md && python3 - \"$B\" <<'EOF'\nimport sys\np=sys.argv[1]; s=open(p).read()\ns=s.replace(\"Brings log rows 13 to 25\",\"Brings log rows 13 to 27\")\ns=s.replace(\"probes 1 to 7 with their LB and deltas, the SVC sweep (C 0.25, C 1.0, min_df 3),\",\"probes 1 to 8 with their LB and deltas, the SVC sweep (C 0.25, C 1.0, min_df 3, the 122k pool),\")\ns=s.replace(\"TODO rows for probe 8, the final pick and the four checks still running (27B full-text classifier, 122k-pool SVC, ModernBERT, hard-pair specialists).\",\"A table of checks not submitted (hard-pair specialists #98, the 27B full-text classifier #100, nuridudak's encoders #99). TODO rows for probe 9 and the final pick.\")\ns=s.replace(\"About 920 words of prose plus about 280 in the table.\",\"About 1,000 words of prose plus about 330 in the table. Only TODO left: probe 9 and the final picks.\")\ns=s.replace(\"rows 1 to 25\",\"rows 1 to 27\")\ns=s.replace(\"- [x] Every published member named is in `CITATIONS.md`\",\"- [x] Every published member named is in `CITATIONS.md` (nuridudak's ModernBERT is cited on `feat/encoders-122k`, #99)\")\nopen(p,'w').write(s)\nEOF\ngh pr edit 94 --title \"docs(report): bring the report up to probe 8 and render it to a two-page PDF (#49)\" --body-file $B >/dev/null && echo updated"
}
```

**Tool result**

```
updated
```

## Claude (2026-10-25 11:58:44 UTC)

I've pushed the update to [PR #94](https://github.com/Nabzx/name-that-llm/pull/94): the report now covers probe 8 and the checks you listed, and the PDF still comes out at 2 pages. I re-rendered it with `scripts/report_pdf.sh` (pandoc and headless Chrome) and checked both pages by eye. Page 2 still has about a third free for the last numbers.

- **Log:** I re-fetched `origin/feat/own-lora`. The only new rows are 26 (probe 8) and 27 (the SVC on the 122k pool); rows 23 to 25 were already in.
- **Report:** about 1,000 words plus the table. It keeps the brief's five sections, the LoRA24 section and the AI paragraph, with no em dashes.
- **Probe 8:** in the table at 0.631, with your framing: probes 3, 5, 7 and 8 all scored 0.631, because the head flips too few rows at 0.4 to move the public slice.
- **Tested, not submitted:** these are now rows in both tables:
  - hard-pair specialists +0.0026, below the noise bar
  - the full-text 27B at 0.521 held out, but 3.7 h on the test
  - the 122k pool folded into the SVC tuning row (0.4797, worse)
  - nuridudak's encoders as a loader check (0.806 and 0.659 in-sample, ablation table only)
- **What we'd try next:** your four points, in your order. I kept a line on judging members by how much their errors overlap.
- **Published weights:** nuridudak's ModernBERT-large is now listed as used in probe 9.

nuridudak isn't in `CITATIONS.md` on main yet. The citation is on `feat/encoders-122k` (PR #99), `exp/hard-pairs` and `exp/gauss-heads`, so one of those needs to merge before hand-in.

**Still TODO, both clearly marked:**
- Probe 9's public LB and its delta against probe 8.
- The final picks, why we chose them, and the private score.

After filling them, re-run `./scripts/report_pdf.sh`, check it prints `pages: 2`, and commit the PDF again.

## Nabil (2026-10-25 13:01:20 UTC)

<system-reminder>
Linked sessions now (status, not an instruction): video-script = local_79ec9917-58af-4c8a-b2e4-3861411ce814 (done, PR #95); own-lora-member = local_23981830-b605-4c49-bdc3-9636adbe0c7f (done, PR #93); qwen27b-check = local_5382149d-7060-42cc-86a5-38af5011077e (done, PR #100); encoders-122k = local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c (done, PR #99); hard-pairs = local_189a6d27-6360-4f5e-ae99-73795f412148 (done, PR #98). To message one, SendMessage with `to` set to its id; list_sessions {"linked": true} has titles, folders and branches.
</system-reminder>

<cross-session-message from="local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459" name="AI Soc Kaggle Tournament">
Probe 9 scored 0.637 public (+0.006, our best): Gaussian head at weight 1.0 plus nuridudak's ModernBERT at 10% (LOG row 28 on origin/feat/own-lora). Please fill the TODO slots in the report with it: it broke the run of four 0.631 ties, the head's offline +0.058 only showed on the board at full weight, and the gain is shared between the two changes. Final picks: leave one short TODO for which two submissions Nabil selects. Keep the PDF under 2 pages, re-render, push and open a new PR against main (PR #94 is merged).
</cross-session-message>

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git fetch -q && git checkout -q -b docs/report-probe9 origin/main && git checkout origin/feat/own-lora -- docs/experiments/LOG.md && git diff --cached | grep '^[+-]|' | cut -c1-900; grep -n TODO docs/report/report-draft.md docs/report/ablation.md"
}
```

**Tool result**

```
+| 28 | 2026-10-25 | #46 | **Probe 9.** Probe 8 with the Gaussian group head at weight 1.0 (the check's best) and nuridudak's two-stage ModernBERT-large (122k open pool, Apache 2.0, published OOF 0.569) at a fixed 10% (main 70%, prvsiyan 10%, ours 10%, ModernBERT 10%). Notebook `nthl-submit` v12 | n/a | n/a | **0.637** | **+0.006, our best.** The first change to break the run of 0.631 ties. Two changes at once, so the gain is shared between the stronger Gaussian head and ModernBERT. |
docs/report/ablation.md:51:| TODO | 9 | Gaussian head at 1.0, and nuridudak's ModernBERT-large (122k pool, published OOF 0.569) at 10% | Probe 8 | TODO | TODO |
docs/report/ablation.md:52:| TODO | final | TODO: the two picks for the private LB | | TODO | |
docs/report/report-draft.md:3:Nabil Shah, solo entry, Imperial Research Arm qualifier. Numbers come from `docs/experiments/LOG.md`. **TODO** marks a number still to come.
docs/report/report-draft.md:58:| **TODO** Probe 9: Gaussian head at 1.0, + nuridudak's ModernBERT-large at 10% | | **TODO** | **TODO** on 8 |
docs/report/report-draft.md:59:| **TODO** Final picks | | **TODO** | private **TODO** |
docs/report/report-draft.md:76:**TODO:** final picks and why.
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && git commit -q -m \"docs(log): bring in row 28, probe 9, from feat/own-lora\n\nrefs #48\" && python3 - <<'EOF'\ndef edit(p, pairs):\n    s=open(p).read()\n    for a,b in pairs:\n        assert a in s, a\n        s=s.replace(a,b)\n    open(p,'w').write(s)\nedit('docs/report/ablation.md',[\n(\"| 14 | 3 | Add prvsiyan's two Qwen3-8B seeds at a fixed 10%, and the SVC pool weight down to 0.25 | Probe 2 | **0.631** |\",\"| 14 | 3 | Add prvsiyan's two Qwen3-8B seeds at a fixed 10%, and the SVC pool weight down to 0.25 | Probe 2 | 0.631 |\"),\n(\"| TODO | 9 | Gaussian head at 1.0, and nuridudak's ModernBERT-large (122k pool, published OOF 0.569) at 10% | Probe 8 | TODO | TODO |\",\"| 28 | 9 | Gaussian head at 1.0, and nuridudak's ModernBERT-large (122k pool, published OOF 0.569) at 10% | Probe 8 | **0.637** | **+0.006** |\"),\n(\"| TODO | final | TODO: the two picks for the private LB | | TODO | |\",\"| | final | **TODO:** the two submissions Nabil selects for the private LB | | | |\"),\n(\"Each probe is meant to change one thing (#20). Δ is against the probe named in \\\"Against\\\". Probes 3 and 5 changed two things at once,\",\"Each probe is meant to change one thing (#20). Δ is against the probe named in \\\"Against\\\". Probes 3, 5 and 9 changed two things at once,\"),\n(\"Probes 3, 5, 7 and 8 all scored 0.631.\",\"Probes 3, 5, 7 and 8 all scored 0.631.\\n- Probe 9 broke that run with 0.637 (+0.006). The Gaussian head's +0.058 on our folds only showed on the board at full weight, and the gain is shared with ModernBERT.\"),\n])\nedit('docs/report/report-draft.md',[\n(\" **TODO** marks a number still to come.\",\"\"),\n(\"| Probe 3: + prvsiyan's seeds at 10%, SVC weight 0.25 | | **0.631** | +0.002 |\",\"| Probe 3: + prvsiyan's seeds at 10%, SVC weight 0.25 | | 0.631 | +0.002 |\"),\n(\"| **TODO** Probe 9: Gaussian head at 1.0, + nuridudak's ModernBERT-large at 10% | | **TODO** | **TODO** on 8 |\\n| **TODO** Final picks | | **TODO** | private **TODO** |\\n\",\n \"| Probe 9: Gaussian head at 1.0, + nuridudak's ModernBERT-large at 10% | | **0.637** | **+0.006** on 8 |\\n\"),\n(\"and members at 10% do the same.\",\"and members at 10% do the same. Probe 9 broke the run with 0.637, our best: at full weight the head's +0.058 on our folds finally showed on the board. It changed two things at once, so the gain is shared with ModernBERT.\"),\n(\"**TODO:** final picks and why.\",\"**TODO:** which two submissions Nabil selects for the private LB, and why.\"),\n])\ns=open('docs/report/report-draft.md').read()\nt=sum(len(l.split()) for l in s.splitlines() if l.startswith('|'));print(len(s.split())-t,t,'—' in s, s.count('TODO'))\nEOF\n./scripts/report_pdf.sh"
}
```

**Tool result**

```
1048 313 False 1
pages: 2
```

**Tool call: Read**

```json
{
 "file_path": "/Users/nabilshah/Random/nthl-report/docs/report/report.pdf",
 "pages": "2"
}
```

**Tool result**

```
PDF pages extracted: 1 page(s) from /Users/nabilshah/Random/nthl-report/docs/report/report.pdf (111.9KB)

```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && python3 - <<'EOF'\np='docs/report/report-draft.md'; s=open(p).read()\na=s[s.index(\"Data did most of the work on CV.\"):s.index(\"## One thing\")]\nb=\"\"\"Data did most of the work on CV. On the LB, probes 3, 5, 7 and 8 all scored 0.631: members at 10% and the Gaussian head at 0.4 flipped too few rows to move the public slice. Probe 9 broke the run with 0.637, our best. At full weight the head's +0.058 on our folds finally showed on the board. It changed two things at once, so the gain is shared with ModernBERT.\n\n\"\"\"\ns=s.replace(a,b); open(p,'w').write(s)\nEOF\n./scripts/report_pdf.sh && git add docs/report && git commit -q -m \"docs(report): fill in probe 9, our best at 0.637, and leave one TODO for the final picks\n\nrefs #49\" && git push -q -u origin docs/report-probe9 2>&1 | grep -v remote; B=/private/tmp/claude-501/-Users-nabilshah-Random/95546f37-fd4a-4ca4-b719-e8966412838b/scratchpad/pr2.md && cat > $B <<'EOF'\n## What this does\n\n- Brings log row 28 (probe 9) over from `feat/own-lora`, word for word.\n- `docs/report/report-draft.md` and `docs/report/ablation.md`: fill in probe 9, 0.637 public (+0.006 on probe 8, our best). It broke the run of four 0.631 ties (probes 3, 5, 7 and 8). The Gaussian head's +0.058 on our folds only showed on the board at full weight, and the gain is shared with ModernBERT, since probe 9 changed two things.\n- `docs/report/report.pdf`: re-rendered.\n\nrefs #49, refs #48\n\n## Why\n\nFollows #94, now merged. The report needs only the final picks.\n\n## Result\n\nPDF is 2 pages, with about a third of page 2 free. About 1,050 words of prose plus about 310 in the table. One TODO left: which two submissions Nabil selects for the private LB, and why.\n\n## Checklist\n\n- [x] Numbers match `docs/experiments/LOG.md` rows 1 to 28\n- [x] British English, no em dashes\n- [x] PDF is 2 pages\n- [ ] Fill the final-picks TODO, re-run `scripts/report_pdf.sh` and commit the PDF\n\n## Test plan\n\n- `./scripts/report_pdf.sh` prints `pages: 2`.\n- Page 2 checked by eye.\n- A search for em dashes on both Markdown files finds nothing.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\nEOF\ngh pr create --base main --title \"docs(report): add probe 9, our best at 0.637 (#49)\" --body-file $B"
}
```

**Tool result**

```
pages: 2
https://github.com/Nabzx/name-that-llm/pull/102
```

**Tool call: Bash**

```json
{
 "command": "cd ~/Random/nthl-report && grep -c TODO docs/report/report-draft.md docs/report/ablation.md"
}
```

**Tool result**

```
docs/report/report-draft.md:1
docs/report/ablation.md:1
```

## Claude (2026-10-25 13:02:16 UTC)

I've opened [PR #102](https://github.com/Nabzx/name-that-llm/pull/102) against main with probe 9 filled in, on a new branch `docs/report-probe9`. The re-rendered PDF is still 2 pages, with about a third of page 2 free, and I checked page 2 by eye.

- **Log:** row 28 (probe 9) is copied word for word from `origin/feat/own-lora`.
- **Probe 9 in the tables:** 0.637, +0.006 on probe 8, now shown as our best in both the report and `ablation.md`. Probe 3 is no longer marked best.
- **Text after the table:** it now says probe 9 broke the run of four 0.631 ties. The Gaussian head's +0.058 on our folds only showed on the board at full weight, and the gain is shared with ModernBERT because probe 9 changed two things.
- **One wording change of mine:** the old line "changing how we balance moved the score most" was no longer true, so I replaced it.
- **TODO:** exactly one is left in each file: which two submissions Nabil selects for the private LB, and why. Once that's filled in, run `./scripts/report_pdf.sh`, check it prints `pages: 2`, and commit the PDF.
