# Ablation table

Built from `docs/experiments/LOG.md` (#48). CV is balanced accuracy on the frozen `GroupKFold` folds over `prompt_id`. "Step" is the gain over the row before it in the chain. "Total" is the gain over the train-only baseline.

## Our SVC, one change at a time

| Log row | Change | CV | Step | Total |
|---|---|---|---|---|
| 1 | Four-view n-gram LinearSVC, host train only (baseline) | 0.3485 | | |
| 2 | Same run through the shared runner `nthl.svc.run` | 0.3485 | 0 | 0 |
| 3 | Add the pool's official API and OpenRouter rows (36,794), weight 1 | 0.4189 | +0.070 | +0.070 |
| 4 | Add every pool row except free proxies (69,603), weight 1 | 0.4534 | +0.035 | +0.105 |
| 5 | Add the whole pool (84,470), weight 1 | 0.4634 | +0.010 | +0.115 |
| 6 | Pool weight 0.5 | 0.4782 | +0.015 | +0.130 |
| 10 | Pool weight 0.35 | 0.4827 | +0.005 | +0.134 |
| 11 | Pool weight 0.25 | **0.4838** | +0.001 | **+0.135** |
| 21 | C 0.25 in place of 0.5 | 0.4775 | -0.006 | |
| 22 | C 1.0 in place of 0.5 | 0.4801 | -0.004 | |
| 23 | min_df 3 | 0.4841 | +0.0003 | |

Row 2 is a reproduction check: the same probabilities to the last digit. Rows 21 to 23 are against row 11 and show our settings were already at their best.

## Changes off the main chain

| Log row | Change | Against | CV | Δ |
|---|---|---|---|---|
| 8 | Leave the pool's gemini-3.7-flash and 3.8-flash rows out (75,943 pool rows) | Row 6 | 0.4482 | -0.030 |
| 9 | Witness features (Qwen3-4B-Base likelihoods, tokens per word) blended with the row 6 SVC OOF at weight 0.35 | Row 6 | 0.1656 alone | +0.005 at the peak, about +0.003 expected |
| 7 | Nishant Dahal's Qwen3-14B checkpoint through our loader | n/a | in-sample 0.858 | not a score |
| 24 | Prompt-centred SVC twin added to the row 11 OOF, test-like groups of 1 to 5 per prompt, 10 draws | Row 11 | twin alone 0.327 on grouped rows | +0.003 at weight 1.0, sd 0.013: noise |
| 19 | Our own Qwen3-8B LoRA24, stage 1 only (never saw host train) | Row 11 | **0.519** on host train | +0.035 over our SVC |
| 25 | Shared-prompt Gaussian group head on our LoRA's stage-1 features, test-like groups, 10 draws | plain head 0.524 on grouped rows | 0.574 alone | **+0.058 at weight 1.0**, sd 0.013 |

Row 19 is our only honest score for a LoRA: stage 1 trained on the pool alone, so host train is unseen. The final model, after stage 2 on host train, scores 0.809 in-sample, which is not a score. Row 7 checks the loader only. The checkpoint saw every train passage, so its train accuracy is not CV. Its own published CV is 0.611.

## Leaderboard probes

Each probe is meant to change one thing (#20). Δ is against the probe named in "Against". Probes 3 and 5 changed two things at once, so their Δ is shared between the two.

| Log row | Probe | Change | Against | Public LB | Δ |
|---|---|---|---|---|---|
| 12 | 1 | Nishant's published spine (Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25, centred twins 0.1) through our loader, with our pool SVC (weight 0.5) in place of his, full-strength Sinkhorn | his notebook, 0.620 | 0.626 | +0.006 |
| 13 | 2 | Sinkhorn at half strength (alpha 0.5) | Probe 1 | 0.629 | +0.003 |
| 14 | 3 | Add prvsiyan's two Qwen3-8B seeds at a fixed 10%, and the SVC pool weight down to 0.25 | Probe 2 | **0.631** | +0.002 |
| 15 | 4 | Add Zbigniew Kirchner's in-style Qwen3-8B at 10% | Probe 3 | 0.627 | -0.004 |
| 17 | 5 | Add UltraBERT at 10%, and ATLAS at 10% on rows whose top two are under 0.3 apart | Probe 3 | 0.631 | 0 |
| 18 | 6 | Peer correction in place of half Sinkhorn | Probe 3 | 0.622 | -0.009 |
| 20 | 7 | Add our own Qwen3-8B LoRA24 at 10% (main 80%, prvsiyan 10%) | Probe 3 | 0.631 | 0 |
| TODO | 8 | Add the Gaussian group head at 0.4 on grouped rows | Probe 7 | TODO | TODO |
| TODO | final | TODO: the two picks for the private LB | | TODO | |

Row 16 is the UltraBERT loader check: in-sample 0.966 on host train, not a score.

## Still running

| Check | Result |
|---|---|
| 27B full-text classifier | TODO |
| SVC on the 122k pool | TODO |
| ModernBERT member | TODO |
| Hard-pair specialists | TODO |

## What the table says

- Data did most of the work. The pool gave +0.115 at weight 1 and down-weighting it gave +0.020 more. The SVC itself never changed.
- More rows beat cleaner labels: the free proxy rows still helped (rows 4 and 5).
- The pool-weight curve has flattened, so we stopped at 0.25.
- Leaving suspect labels out hurt more than keeping them (row 8).
- The likelihood witnesses carry real but small signal, about +0.003 on the SVC.
- On the LB, the steps that changed how we combine and balance gave the most: half Sinkhorn +0.003. Adding strong published members gave little or hurt: prvsiyan's seeds shared +0.002, Zbigniew's in-style model -0.004, UltraBERT, ATLAS and our own LoRA nothing. Dropping the class quotas (peer correction) cost 0.009.
- The one large gain on our own folds since the pool is the Gaussian group head, +0.058 on test-like groups.
- From probe 3 on, every gain is 0.004 or less. The public LB scores only part of the 4,000 test passages, split 18 ways, so that is close to noise. We only kept changes the CV or a research note also backed.
