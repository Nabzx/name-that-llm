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

In every pool run gemini-3.7-flash recall fell from 0.41 to 0.19 while 3.8-flash rose to 0.72. We read that as mislabelled pool rows: some of the pool's 3.7 passages might really be 3.8. So we left the pool's 3.7-flash and 3.8-flash rows out (log row 8).

CV fell by 0.030, to 0.4482. Recall on 3.7 dropped to 0.01 and on 3.8 to 0.03, and their passages went to gemini-3.6-flash instead (0.90). The pool rows were noisy, but they were also the only examples deep enough to tell the three flash models apart. Host train alone could not. Noisy labels for a class beat no labels for it. We kept the rows.

## What we built on top of published work

The organisers allow published weights if we show what we built on them.

**Ours:**

- The pool build: 84,470 passages from open-licence public datasets, filtered, de-duplicated and tagged by source and route (#19, #26).
- The four-view n-gram SVC and its pool-weight study (log rows 1 to 11).
- Our own LoRA loader (`nthl.lora24`). It reads two checkpoint formats, Nishant's and prvsiyan's, into one spec, merges the adapter, pools the hidden states and reorders the label columns. It asserts the label order and the training end token.
- The blend and balancing analysis: how to validate a blend whose members have no out-of-fold predictions, and why half-strength balancing (#20, #23).
- The likelihood witnesses (#34).

**Published, used as released:**

- Nishant Dahal's Qwen3-14B and Qwen2.5-14B LoRA checkpoints and his DeBERTa-v3-large, with his blend weights (CC BY-SA 4.0).
- prvsiyan's two Qwen3-8B seeds (CC BY-SA 4.0).

Every dataset, model and notebook we read is listed in `CITATIONS.md`.

## What we'd try next

- A model from a non-Qwen family. Almost all published members are Qwen, and they agree on about 77% of top picks (#18).
- Test-time pseudo-label heads and Gaussian group heads on the 14B LoRA features. No public notebook has the first, and the second added about +0.005 on the LB for prvsiyan (#23, #24).
- Judge every test-time step on folds cut to one to three passages per prompt, as in the test, not train's 18.

## AI assistance

We used Claude Code with Claude Opus 5.5 throughout, with agents for research tickets. Every session is logged in `docs/ai-sessions.md`.
