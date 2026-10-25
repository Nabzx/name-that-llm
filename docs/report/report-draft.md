# Name That LLM

Nabil Shah, solo entry, Imperial Research Arm qualifier. Best public LB **0.637**. Every number is a row in `docs/experiments/LOG.md`.

## Starting point: research before code

I wrote no code for the first two hours. I read the rules, 22 public notebooks, 18 discussion threads and the papers behind them, and wrote 15 short research notes, each ending in a recommendation (`docs/research/`). Three findings set my plan. Data mattered more than model size: the same Qwen3-14B recipe went from 0.569 to 0.611 when its public pool grew from 35k to 84k passages. The best public blend (0.620, Nishant Dahal) published its LoRA weights, which the organisers confirmed I could use if I showed I understood them. And no public checkpoint had out-of-fold predictions, so the leaderboard would be my only test for those members.

So my approach was: validate first, build on the published spine with my own loader, and add members I could check on my own folds. I froze five folds grouped by `prompt_id` before fitting anything, so validation looked like the hidden test. My first model, a four-view n-gram SVC on host data, scored 0.348.

## Literature review

- **Two-stage LoRA on a public pool** (Nishant Dahal, 0.592 then 0.620): truncate a Qwen decoder, fine-tune on public in-style passages, then on host train; most of the gain came from the pool, not the backbone.
- **Shared-prompt Gaussian heads** (prvsiyan, *The Prompt Is Not the Author*): model features as author mean plus prompt effect plus noise and decode prompt groups jointly; about +0.005 each time it was added publicly.
- **Likelihood attribution** (ATLAS, Ko et al., arXiv 2610.19583; Fast-DetectGPT, Binoculars): source-conditioned likelihoods beat single-score detectors at naming a model, but add little on top of fine-tuned classifiers.
- **Test-time balancing** (forum study by s0chn1y): full Sinkhorn is a bet that the test is even; a partial balance keeps most of the gain and caps the loss.
- **Data audits** (rmsgrk, nuridudak): a bigger merged pool helps encoders; filtering it rarely helps.

## How I made decisions

One change per leaderboard probe, and I kept a change only if my cross-validation or a research note backed it too. When a probe and my CV disagreed, I trusted the CV: the public slice is small, and four probes in a row tied at 0.631. I dropped members that hurt (Zbigniew Kirchner's in-style LoRA, −0.004) and ideas that stayed in the noise, and wrote both down.

## Hypotheses

- **Style lives in small habits.** Function words, punctuation and word shapes change little with topic, so n-grams should separate labs. They did: the SVC put the right lab first 0.57 of the time. Siblings stayed hard: glm-5.3 against its flash model, gemini-3.7 against 3.8.
- **Data beats model size.** 84,470 open-licence public passages took the SVC from 0.348 to 0.484.
- **The prompt hides the author.** Test passages come in groups answering one prompt. Remove what a group shares and style is left. A shared-prompt Gaussian group head on my own model's honest features added **+0.058** on test-like groups.
- **The test mix is uneven.** Full balancing bets on an even test, so I balanced halfway: +0.003 on the LB.

## The published weights, and what is mine

A LoRA24 checkpoint tokenises with no special tokens, cuts at 447 and appends the end token from its `meta.json` (151643, not the tokenizer's 151645). It keeps the first 27 of Qwen3-14B's 40 layers with the final norm off, merges the rank-16 adapter (W + B·A × 2), mean-pools 8 depths without the first token, LayerNorms each and applies a linear head. Both traps are silent: the wrong end token changed 14% of top picks, and a wrong label order puts one Claude model's probabilities under another's name.

**Mine:** the loader, which asserts both traps (0.858 in-sample on Nishant's 14B); the 84k pool; the SVC and its tuning; half-strength balancing; my own Qwen3-8B LoRA, trained on Kaggle in 4.5 hours with my own code; the Gaussian group head, my own code after prvsiyan's public method; and every blend experiment. **Used as released:** Nishant Dahal's Qwen3-14B, Qwen2.5-14B and DeBERTa, prvsiyan's two Qwen3-8B seeds and nuridudak's ModernBERT. Sources are in `CITATIONS.md`.

## Ablation

| Change | CV | Public LB | Δ |
|---|---|---|---|
| SVC, host data only | 0.348 | | |
| + 84k public pool, weight 0.25 | 0.484 | | +0.136 |
| My Qwen3-8B LoRA, stage 1, on unseen host train | 0.519 | | |
| + Gaussian group head, test-like groups | | | **+0.058** |
| Centred SVC, pair specialists, 122k pool | | | noise or worse |
| Probe 1: published spine + my SVC | | 0.626 | +0.006 on 0.620 |
| Probe 2: half-strength balancing | | 0.629 | +0.003 |
| Probe 3: + prvsiyan's seeds at 10% | | 0.631 | +0.002 |
| Probe 4: + in-style Qwen3-8B | | 0.627 | −0.004 |
| Probes 5, 7, 8: + UltraBERT and ATLAS / my LoRA / Gaussian head at 0.4 | | 0.631 | 0 |
| Probe 6: peer correction | | 0.622 | −0.009 |
| **Probe 9: Gaussian head at 1.0 + ModernBERT** | | **0.637** | **+0.006** |
| Probe 10: + Gaussian heads on the 14B LoRAs | | 0.630 | −0.007 |

Final picks: probe 9 (best) and probe 7 (0.631, a simpler blend that carries my own model).

## One thing that didn't work, and why

Peer correction, a public idea, replaces balancing: inside each topic cluster every passage loses a share of its neighbours' shared SVC lean, and class quotas go. Its author gained +0.010 on a weaker blend. On mine the LB fell from 0.631 to 0.622. Without quotas a strong blend drifts towards the classes it already favours, and balanced accuracy punishes every class it starves. A trick that rescues a weak blend can hurt a strong one. Probe 10 taught a related lesson: the Gaussian head helped on my own model's honest features but hurt on the 14B models, which had seen all of train, so their train features were too clean.

## What I'd try next

Retrain on the open 122k pool (stage 1 alone is about 10 hours on two T4s). Give every member five folds, so the blend is fitted on honest scores rather than probes. Decode each prompt group jointly across all members. Ask the organisers whether self-generated passages count as external data, then target the sibling pairs.

## Research practice

I treated the weekend as a small research project, not a leaderboard chase. Every idea started as a hypothesis with a reason, and every result, good or bad, became a row in the log: 29 rows, about a dozen of them failures or ties. I used held-out evidence I could trust: frozen prompt-grouped folds, test-like groups of 1 to 5 passages clustered with the real test texts, and features from a model that never saw the rows it was scored on. Leaderboard probes changed one thing at a time, so each score has a cause. I cited every source and licence, and I chose the honest result over the flattering one when they disagreed: the 27B model's held-out 0.521, not its in-sample score.

## How to reproduce

```
git clone https://github.com/Nabzx/name-that-llm && cd name-that-llm
python -m venv .venv && .venv/bin/pip install -e ".[dev]" kaggle
.venv/bin/pytest -q                  # 88 unit tests, CPU only
.venv/bin/kaggle auth login           # your own Kaggle account
scripts/push_src.sh                   # src/nthl as the code dataset
.venv/bin/kaggle kernels push -p notebooks/train-lora --accelerator NvidiaTeslaT4   # my LoRA
.venv/bin/kaggle kernels push -p notebooks/submit --accelerator NvidiaTeslaT4       # probe 9
```

Folds are frozen in `src/nthl/folds.csv` and every seed is 42. Each log row names its notebook and version, so any number can be rerun on its own; the submission's settings sit at the top of `notebooks/submit/run.py`.

## How I used AI

I used Claude Code (Claude Opus 5.5) from the first hour, set up like my other repositories, and that structure is why one person could run this many experiments. Before any model, the repo had an `AGENTS.md` and `CLAUDE.md` with the competition rules as invariants (no hand-labelling, train only on Kaggle, cite everything), phases with small tickets, a research note before every decision and one log row per change. Folds were frozen, tests came before code and every model ran in a Kaggle notebook, so any result can be rerun. With that in place, Claude researched, wrote code and ran checks, and up to six side sessions worked in parallel on their own branches: the report, the video, the encoders, the 27B, the pair specialists and my LoRA's wiring. I set the direction, chose every probe, made every submission and merge call, and asked for evidence before keeping anything. Every session is exported in `ai-sessions/` and listed in `docs/ai-sessions.md`.
