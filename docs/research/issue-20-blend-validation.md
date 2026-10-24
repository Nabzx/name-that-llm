# Issue #20: Validating a blend when members have seen every train passage

## Question

Most published checkpoints we will blend were trained on all 2,699 host train passages. Their predictions on train are in sample, so they cannot set blend weights. What can we trust instead: published OOF files, fold checkpoints, our own OOF members or the public LB?

## What we looked at

- The issue 18 note on which checkpoints come with out-of-fold data.
- GoPain's single-member submissions in discussion 750390, and the follow-ups in 751941 and 751370.
- How nine public notebooks set their weights: Nishant Dahal's two, Lucas Boesen's adaptive centering, prvsiyan's, Кирилл Тымчишин's two, Joseph Adamski's ATLAS mix, lightningv08's 0.636 and hanifnoerrofiq's.
- Our SVC's OOF and summary (log row 6).
- Background on stacking and on reusing a holdout: Breiman (1996), Blum and Hardt (2015), Roelofs et al. (2019) and Whitehill (2016, 2017).

Nothing was run or fitted.

## Findings

### Train predictions from a leaked member are no use for weights

A member that read a passage in training is near certain on it. Fit weights on those predictions and the member that memorised most takes all the weight. Stacking has always needed out-of-sample inputs: Breiman fits non-negative weights on cross-validated predictions for this reason. No public notebook breaks this.

In-sample predictions are still good for one thing: a load check. A checkpoint loaded correctly reproduces its own training fit. prvsiyan asserts the features of three training passages before trusting a checkpoint. Nishant picks the end token that best reproduces the fit on 64 training rows, because the wrong one changed 14% of the 14B's top picks with no error.

### Where public notebooks get their weights

| Weights come from | Who | What they report |
|---|---|---|
| The author's own 5-fold OOF, then rounded | Nishant, part two: SVC 1.0, Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25 | Part one fitted weights to three decimals. Part two rounds them, because finer fitting on 2,699 passages mostly fits noise. A stacker did no better. |
| L-BFGS on prompt-grouped OOF, checked on held-out halves | Lucas Boesen | Fits on half the prompts and scores the other half, over 20 splits. Each new member is reported with the share of splits it wins, for example +0.006 in 20 of 20. Half-split score 0.4928 against 0.4927 in sample, so a handful of weights barely overfit. |
| A fold checkpoint stands in for the full-data one | Lucas (lightningv08's LoRA24: fold OOF for the weight, `final` at test); lightningv08 (outer-0 checkpoint as evidence for the full-data Qwen3.8-27B) | The usual way to weight a model refit on all of train. |
| A share fixed in advance, then one LB test | Кирилл's 85/15 branch mix; ATLAS at 10%; fleongg's complement at 15% | Кирилл calls 85/15 a cautious guess fixed in advance, not a CV optimum. ATLAS at 10% gained +0.002 to +0.007 on seven different bases, on the LB only. |
| Inherited unchanged | prvsiyan and the forks of the 0.563 notebook | Its OOF-fitted weights are reused with no re-tuning. |

hanifnoerrofiq goes furthest: five outer folds score the whole selection procedure, which runs again inside each fold.

### Few members come with OOF

From issue 18: UltraBERT ships 5 fold models (OOF 0.408), the lightningv08 bundle has an OOF cache, Zbigniew reports fold 0 only (539 rows) and prvsiyan reports a number (0.568) with no file. Nishant's part two, our spine, ships no OOF, but he set its weights on his own OOF. Our SVC has honest OOF on our frozen folds (0.4782).

Mixing OOF from different sources is fair for a linear blend: each prediction is out of fold for its own member, even if the folds differ. Join on `id` and check no row is missing. A fold model is a little weaker than the full-data model or the 5-fold average used at test, so OOF undersells it slightly.

### Where OOF misleads

- **Pool-trained members read high.** In GoPain's submissions a train-only SVM tracked the LB (CV minus LB −0.009), while pool-trained members read 0.027 to 0.046 above it. Nishant's part two: 0.642 CV, 0.620 LB. OOF-fitted weights will favour a pool-trained member over a train-only one (UltraBERT, ATLAS) by about that offset.
- **Test-time steps are missing.** Sinkhorn balancing gained +0.011 on the LB but under 0.001 on CV (GoPain). The centred twin needs test clusters, which plain CV lacks. So OOF weights should be scored after balancing, and twin weights stay as their authors set them.
- **Scale.** In a log-probability blend a weight means something only on its member's own scale. Nishant's SVC enters as a softmax of the raw SVC scores. Ours is divided by a temperature of 3.55 first, so at the same weight it counts about a third as much. Our SVC is also close to the SVC already in the spine (0.478 against 0.473, same recipe), so it adds little to it.

### What the LB can and cannot tell us

The event scores the public LB, so a public score is the target itself, not an estimate of it. Kaggle shows a team's best public score, so a probe that scores lower costs only its slot. The limits are elsewhere.

- **Budget.** Five a day, reset at 00:00 UTC. Saturday's five run out at 01:00 UK on Sunday morning and Sunday's five at the 18:00 deadline. Each probe is a full GPU run on the hidden test, one to two hours. So at most 10, and one probe gives one number for one weight vector. The LB can set one or two shares, not a weight per member.
- **Noise beyond these rows.** Against the full test the public LB's standard error is about 0.014. Between two submissions the paired error is smaller, about √d / 1,200, where d is the number of rows on which exactly one of them is right. A 10% share that flips 60 rows gives about 0.006 at most. Every probe scores the same rows, so repeating probes does not average that out (GoPain). Gains of +0.002 to +0.007 are real on the public rows but say little about the private board. Blum and Hardt's Ladder only accepts a new best above such a margin. Roelofs et al. found little overfitting to the public board across 120 Kaggle contests, so ten probes put the private score at small risk.

Submitting to learn test labels is a different matter. Whitehill shows a score oracle leaks labels when submissions are built for it. That is labelling test data by another route, and our invariants forbid it.

## Recommendation

Validate the blend in two branches and mix them once.

**Branch A, the spine: no OOF, weights frozen.** Nishant's part two as published: SVC 1.0, Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25, twin 0.1, his Sinkhorn. We never re-fit these. Two checks before anything else:

1. A load check on train: run each checkpoint on a sample of train passages with the end token in its `meta.json`. It should score far above its OOF. If it does not, the load is wrong.
2. A reproduction probe: our run must land within 0.003 of his public 0.620. A bigger gap is a bug, and no other probe goes in until it is fixed.

Leaked additions join branch A one at a time as a fixed 10% share of the pre-Sinkhorn probabilities, the share the public tests used. One addition per probe; keep it if the LB goes up.

**Branch B: honest OOF only.** Our SVC, UltraBERT's fold models (each scored on its own held-out fold) and every member we train later on the frozen folds.

- Fit weights on the joined OOF after Sinkhorn, non-negative, in steps of 0.05.
- Check them on 20 random half splits by prompt: fit on one half, score the other. A member stays only if it helps in at least 15 of the 20.
- Weigh pool-trained against train-only members with the 0.03 to 0.04 offset in mind.

**The mix.** `P = (1 − α) · P_A + α · P_B` before balancing, then one Sinkhorn, as in Кирилл's 85/15 mix. α is the only weight the LB sets. Start at 0.15.

| Member | Honest OOF | Weight from |
|---|---|---|
| Nishant's part two (four members and twin) | No file; the author fitted on his | Frozen as published |
| ATLAS | No | Fixed 10% in A, one probe |
| prvsiyan's two Qwen3-8B seeds | Reported number only | Fixed 10% in A, one probe |
| Zbigniew's in-style Qwen3-8B | Fold 0 number only | Fixed 10% in A, one probe |
| Our SVC | Yes, frozen folds | OOF fit within B |
| UltraBERT, 5 folds | Yes, if each fold's held-out rows are known | OOF fit within B |
| Our later members | Yes, frozen folds | OOF fit within B |
| Branch B as a whole | Only within B | α from the LB |

The Qwen3.8-27B stays out for now (#18). If it returns, check which folds its OOF cache covers before treating it as a branch B member.

**Probes, one change each, each with Nabil's go-ahead.**

| # | Submission | Keep if |
|---|---|---|
| 1 | Spine as published, run in our notebook | Within 0.003 of 0.620; else debug |
| 2 | + ATLAS at 10% | LB up |
| 3 | + prvsiyan's seeds at 10% | LB up |
| 4 | + Zbigniew's LoRA at 10% | LB up |
| 5 | + branch B at α 0.15 | LB up |
| 6 to 8 | α 0.1 or 0.25 depending on 5; new branch B members once their OOF earns a place | LB up |
| 9 to 10 | Reserve for a failed run and the final pick | |

Spend Saturday's slots on branch A, since they are lost at 01:00 UK. Submit the last probe by 15:00 UK on Sunday so it scores in time. Log each probe in `docs/experiments/LOG.md` with d against the previous best, so we know which gains clear the paired noise. Of the two final picks for the private board, take the best public score and the best blend whose changes all had OOF or published support.

**What we never do.**

- Fit weights, temperatures or a stacker on train predictions of a member that saw those passages.
- Re-fit the weights inside a published block without its OOF.
- Fit weights finer than 0.05 on 2,699 passages.
- Change two things in one probe.
- Build a submission to learn test labels from its score: single rows, one class, crafted probabilities. Every submission is a full blend from code.
- Catch a failing member and carry on. Nishant lost points to a handler that silently dropped one; the probe then measures a different blend.

#29 builds the blend this way. #45 turns the probe table into the day plan.

## Sources

- `docs/research/issue-18-published-checkpoints.md`
- `docs/experiments/LOG.md`, row 6, and `outputs/svc-pool-all-w05/summary.json`
- GoPain, comment in [discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390), and in [751941](https://www.kaggle.com/competitions/name-that-llm/discussion/751941) and [751370](https://www.kaggle.com/competitions/name-that-llm/discussion/751370)
- Nishant Dahal, [Two-stage LoRA, part two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two) and [Two-stage LoRA on the public pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- Lucas Boesen, [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- prvsiyan, [The Prompt Is Not the Author](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)
- Кирилл Тымчишин, [0.600 Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt) and [0.639 Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature)
- Joseph Adamski, [Teach the Model + 10% ATLAS](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas)
- Vladislav Lavrentev, [0.636 LoRA Llama Qwen3.8-27B](https://www.kaggle.com/code/lightningv08/0-636-lora-llama-qwen3-8-27b)
- hanifnoerrofiq, [SVM frozen Qwen](https://www.kaggle.com/code/hanifnoerrofiq/svm-frozen-qwen)
- L. Breiman, "Stacked Regressions", Machine Learning 24 (1996), [tech report](https://stat.berkeley.edu/sites/default/files/tech-reports/367.pdf)
- A. Blum and M. Hardt, "The Ladder: A Reliable Leaderboard for Machine Learning Competitions", ICML 2015, [arXiv:1502.04585](https://arxiv.org/abs/1502.04585)
- R. Roelofs et al., "A Meta-Analysis of Overfitting in Machine Learning", NeurIPS 2019, [paper](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)
- J. Whitehill, "Exploiting an Oracle that Reports AUC Scores in Machine Learning Contests", AAAI 2016, [arXiv:1506.01339](https://arxiv.org/abs/1506.01339), and "Climbing the Kaggle Leaderboard by Exploiting the Log-Loss Oracle", 2017, [arXiv:1707.01825](https://arxiv.org/abs/1707.01825)
