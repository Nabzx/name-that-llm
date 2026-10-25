# Name That LLM

Nabil Shah, solo entry, Imperial Research Arm qualifier. Numbers come from `docs/experiments/LOG.md`.

## Starting point and approach

We wrote our own code from scratch. The first model was a four-view n-gram LinearSVC on host train only, scored on frozen `GroupKFold` folds over `prompt_id`: 0.3485 balanced accuracy, in line with public train-only SVCs.

We then worked in two branches. Branch A is a published spine we run through our own loader: Nishant Dahal's two-stage LoRA blend, 0.620 on the public LB. Branch B is what we can check on our own folds: our SVC, our own LoRA and test-time heads. Published members have no out-of-fold predictions, so for them the LB is the only test. We probed it one change at a time and kept a change only if our CV or a research note also backed it.

## Hypotheses

- **Data beats model size.** Public work showed 8B and 14B backbones tied on the same data, while a bigger pool gave +0.028 on the LB.
- **Labs differ in surface style, siblings do not.** Our baseline put the right lab first 0.57 of the time, and half its errors stayed inside the right lab. Hardest: gemini-3.7-flash against 3.8-flash, and glm-5.3 read as claude-opus-5.
- **The prompt is louder than the author.** Passages that answer the same prompt share content, so taking out the prompt should leave mostly style.
- **The test mix is unequal.** Balancing towards 1/18 should help, but at half strength, since we cannot see the test's skew.

## The published LoRA24 weights and what we built on them

Each LoRA24 checkpoint turns a passage into 18 probabilities like this. Tokenise with no special tokens, cut at 447 and append the end token from the checkpoint's `meta.json` (151643), not the tokenizer's (151645). Keep only the first 27 of Qwen3-14B's 40 decoder layers, with the final norm off. Merge the rank-16 adapter into each attention and MLP weight: W + B·A × 2. Take a masked mean of the hidden states at 8 depths, leaving out the first token. LayerNorm each depth, join them into one 40,960-wide vector and apply a linear head. A second, centred head reads the same vector minus its topic cluster's mean. The traps are silent: a wrong label order puts claude-fable-5's probabilities under another Claude model, and the wrong end token changed 14% of top picks in Nishant's tests.

**What is ours:**

- The loader (`nthl.lora24`), written from that recipe. It asserts the end token and reorders label columns by `meta.json`. Nishant's Qwen3-14B scored 0.858 in-sample through it, with every Claude label right.
- The pool: 84,470 passages from open-licence public datasets, filtered, de-duplicated and tagged by source.
- Our SVC and its tuning, 0.3485 to 0.4838 CV.
- Half-strength Sinkhorn, and peer correction re-implemented from avazaku's idea.
- A likelihood witness from Qwen3-4B-Base, about +0.003 on the SVC.
- Our own Qwen3-8B LoRA24, trained on Kaggle with our own code in 4.5 hours: stage 1 on a balanced pool sample, stage 2 on host train.
- A shared-prompt Gaussian group head, our own code after prvsiyan's public method.
- Every blend experiment, probes 1 to 9.

**Used as released:** Nishant Dahal's Qwen3-14B, Qwen2.5-14B and DeBERTa-v3-large with his blend weights, and prvsiyan's two Qwen3-8B seeds. In probes 4 and 5 only: Zbigniew Kirchner's in-style Qwen3-8B, UltraBERT and ATLAS. In probe 9: nuridudak's ModernBERT-large. All sources are in `CITATIONS.md`.

## Ablation

CV is on our frozen folds. Δ is against the row before, unless named. The full table is in `docs/report/ablation.md`.

| Change | CV | Public LB | Δ |
|---|---|---|---|
| SVC, host train only | 0.3485 | | |
| + whole pool, weight 1 | 0.4634 | | +0.115 |
| pool weight 0.5 | 0.4782 | | +0.015 |
| pool weight 0.25 (C 0.25, C 1.0, min_df 3 and the 122k pool no better) | 0.4838 | | +0.006 |
| Prompt-centred SVC twin, test-like groups | | | +0.003 on the SVC, sd 0.013 |
| Our Qwen3-8B LoRA, stage 1 only, on unseen host train | 0.519 | | +0.035 on the SVC |
| + Gaussian group head on its features, test-like groups | | | **+0.058** on its plain head, sd 0.013 |
| Hard-pair specialists on the SVC | | | +0.0026, below the noise bar |
| Full-text Qwen3.8-27B classifier, held out | 0.521 | | 3.7 h on the test, too slow |
| Probe 1: Nishant's spine with our SVC, full Sinkhorn | | 0.626 | +0.006 on his 0.620 |
| Probe 2: half Sinkhorn | | 0.629 | +0.003 |
| Probe 3: + prvsiyan's seeds at 10%, SVC weight 0.25 | | 0.631 | +0.002 |
| Probe 4: + Zbigniew Kirchner's in-style Qwen3-8B at 10% | | 0.627 | -0.004 on 3 |
| Probe 5: + UltraBERT 10%, ATLAS 10% on unsure rows | | 0.631 | 0 on 3 |
| Probe 6: peer correction in place of half Sinkhorn | | 0.622 | -0.009 on 3 |
| Probe 7: + our own LoRA at 10% | | 0.631 | 0 on 3 |
| Probe 8: + Gaussian head at 0.4 on grouped rows | | 0.631 | 0 on 7 |
| Probe 9: Gaussian head at 1.0, + nuridudak's ModernBERT-large at 10% | | **0.637** | **+0.006** on 8 |

Data did most of the work on CV. On the LB, probes 3, 5, 7 and 8 all scored 0.631: members at 10% and the Gaussian head at 0.4 flipped too few rows to move the public slice. Probe 9 broke the run with 0.637, our best. At full weight the head's +0.058 on our folds finally showed on the board. It changed two things at once, so the gain is shared with ModernBERT.

## One thing that didn't work, and why

Probe 6 swapped half Sinkhorn for peer correction. Within each topic cluster, every passage loses a shrunk share of the other members' mean SVC lean, so a shared prompt stops pulling them all the same way. avazaku saw +0.010 from it on a weaker blend. On ours the LB fell from 0.631 to 0.622, our worst drop.

Peer correction removes the prompt's lean, but it has no class quotas. Sinkhorn pushes every class towards a fair share of the test. Without it, a strong blend drifts towards the classes it already favours, and balanced accuracy punishes every class it starves. On a weak blend the prompt's lean is the bigger error, so peer correction wins. On a strong one the class drift is bigger. A step that helps one blend is not a free gain on another. We kept half Sinkhorn.

## What we'd try next

- Train our own members five-fold, so each has honest out-of-fold predictions, the blend can be fitted on them and members can be judged by how much their errors overlap, not by LB probes. Our LoRA, UltraBERT and ATLAS tied at 10%, and Zbigniew's model hurt.
- Retrain on a bigger pool. Swapping in the 122k pool did not help the SVC, and our LoRA would need about 10 hours of stage 1 on two T4s to use it.
- Decode each shared-prompt group jointly across all members, not on one member's features.
- Ask the organisers whether passages we generate ourselves count as external data under rule 2.6.

**TODO:** which two submissions Nabil selects for the private LB, and why.

## How we used AI

We used Claude Code (Claude Opus 5.5) throughout: one main session to plan and review, with side sessions and agents for research notes and parallel experiments, each in its own branch. Every session is listed in `docs/ai-sessions.md` and exported to `ai-sessions/`. Claude wrote much of the code and the notes. Nabil set the direction, chose every probe and reviewed and merged every pull request. Every model fit ran on Kaggle, and no test passage was labelled by hand or by an LLM.
