# Name That LLM: report draft

Nabil Shah, solo entry, Imperial Research Arm qualifier. Draft for a two-page PDF (#49). Numbers come from `docs/experiments/LOG.md` and `docs/research/`. Every **TODO** is a number we get on Sunday afternoon (#49, #50).

## Approach

We wrote our own code from scratch. The first model was a four-view n-gram LinearSVC on host train only, scored on frozen `GroupKFold` folds over `prompt_id`: 0.3485 balanced accuracy, in line with public train-only SVCs (0.343 to 0.370).

We then worked in two branches (#20). Branch A is a published spine we run through our own loader but never re-fit: Nishant Dahal's two-stage LoRA blend, 0.620 on the public LB. Branch B is what we can check on our own folds, starting with our SVC. Published members have no out-of-fold predictions, so the LB is the only test for them. We probed it one change at a time and only kept a change that our CV or a research note also backed.

## Hypotheses

- **Data beats model size.** Public work showed 8B and 14B backbones tied on the same data, while a pool of 84k passages in place of 35k gave +0.028 on the LB (#24).
- **Labs differ in surface style, siblings do not.** Our baseline put the right lab first 0.57 of the time, and half its errors stayed inside the right lab. Hardest: gemini-3.7-flash against 3.8-flash, and glm-5.3 read as claude-opus-5.
- **The prompt is louder than the author.** Passages that answer the same prompt share content, so taking out a topic estimate should leave mostly style (#23).
- **A lab's own model finds its text more likely.** Likelihood features from a base model in each family should add what n-grams miss (#31 to #33).
- **The test mix is unequal.** Balancing towards 1/18 should help, but at half strength, since we cannot see the test's skew (#23).

## How the LoRA24 readout works

The organisers allow published weights if we show what we built on them. Each published LoRA24 checkpoint turns a passage into 18 probabilities like this (#21):

1. Tokenise with no special tokens, cut at 447, and append the end token from the checkpoint's `meta.json` (151643), not the tokenizer's (151645).
2. Keep only the first 27 of Qwen3-14B's 40 decoder layers, with the final norm switched off.
3. Merge the adapter into each q, k, v, o, gate, up and down weight: W + B·A × 2.
4. Take a masked mean of the hidden states at 8 depths, leaving the first token out.
5. LayerNorm each depth, join them into one 40,960-wide vector and apply a linear head [18, 40,960].
6. A second, centred head reads the same vector minus its topic cluster's mean.

The traps are silent. A wrong label order puts claude-fable-5's probabilities under another Claude model with no error. The wrong end token changed 14% of top picks in Nishant's tests.

**What is ours:**

- The loader (`nthl.lora24`), written from scratch from that recipe. It reads Nishant's and prvsiyan's formats into one spec and splits a 14B model over two T4s.
- Its checks: it asserts the end token and reorders label columns by `meta.json`. On host train the Qwen3-14B checkpoint scored 0.858 in-sample with every Claude label right, so the reorder works (log row 7).
- The pool: 84,470 passages from open-licence public datasets, filtered, de-duplicated and tagged by source (#19, #26).
- Our SVC and its pool-weight study, 0.3485 to 0.4838 CV.
- Half-strength Sinkhorn, from our own simulation on the SVC's out-of-fold predictions (#23).
- Peer correction, re-implemented from avazaku's idea: each topic cluster loses a shrunk share of the other members' mean SVC lean (#46).
- The witness: Qwen3-4B-Base likelihood features plus tokens per word under Qwen's and Gemma 4's tokenizers (#34).
- Our own Qwen3-8B LoRA, trained overnight with our own code, first on our pool, then on host train (#43).
- The blend experiments: probes 1 to 6 and the rule for judging members we cannot cross-validate (#20).

**Published, used as released:** Nishant Dahal's Qwen3-14B, Qwen2.5-14B and DeBERTa-v3-large with his blend weights, prvsiyan's two Qwen3-8B seeds, and in probes 4 and 5 Zbigniew Kirchner's in-style Qwen3-8B, UltraBERT and ATLAS. All are listed in `CITATIONS.md`.

## Ablation

Full table in `docs/report/ablation.md`.

| Step | CV | Public LB | Δ |
|---|---|---|---|
| SVC, host train only | 0.3485 | | |
| + whole pool, weight 1 | 0.4634 | | +0.115 CV |
| pool weight 0.5 | 0.4782 | | +0.015 CV |
| pool weight 0.25 | 0.4838 | | +0.006 CV |
| + witness features, on the weight 0.5 SVC | 0.1656 alone | | about +0.003 CV |
| Probe 1: Nishant's spine with our SVC, full Sinkhorn | | 0.626 | +0.006 on his 0.620 |
| Probe 2: half Sinkhorn | | 0.629 | +0.003 |
| Probe 3: + prvsiyan's seeds at 10%, SVC weight 0.25 | | **0.631** | +0.002 |
| Probe 4: + Zbigniew's in-style model at 10% | | 0.627 | -0.004 on probe 3 |
| Probe 5: + UltraBERT 10%, ATLAS 10% on unsure rows | | 0.631 | 0 on probe 3 |
| Probe 6: peer correction in place of Sinkhorn | | TODO | TODO on probe 3 |
| Probe 7: + our own Qwen3-8B LoRA | TODO | TODO | TODO |
| Final picks | | TODO | private TODO |

## One thing that didn't work, and why

Probe 4 added Zbigniew Kirchner's in-style Qwen3-8B LoRA at 10%. It scores 0.574 on its own, a strong model, so we expected a small gain. The LB fell from 0.631 to 0.627.

His model was trained on in-style pool passages much like the ones in our SVC's pool. So on the passages our blend gets wrong, it is likely wrong in the same way. Adding it pushed those errors further rather than fixing them. Probe 5 points the same way: UltraBERT and ATLAS lifted the public 0.620 notebook to 0.636, but added nothing to ours, so our SVC and prvsiyan's seeds already fix the rows they fix.

The lesson is that a member's own score says little about what it adds. What counts is whether its errors differ from the blend's, and we could not measure that, because published members have no out-of-fold predictions. We dropped Zbigniew's model and stopped adding published members.

## What we'd try next

- Out-of-fold predictions for our own LoRA, so that members can be judged on how much their errors overlap, not by LB probes.
- A model from outside the Qwen family. Almost all published members are Qwen, and they agree on about 77% of top picks (#18).
- Test-time pseudo-label heads and Gaussian group heads on the LoRA features: +0.005 on the LB for prvsiyan (#23, #24).
- Folds cut to one to three passages per prompt, as in the test, to judge every test-time step.

TODO: final pick and why, once probe 6 and our LoRA are scored.

## AI assistance

We used Claude Code with Claude Opus 5.5 throughout, with agents for research tickets and parallel experiments. Every session is logged in `docs/ai-sessions.md`.
