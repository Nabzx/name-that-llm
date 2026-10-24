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

Row 2 is a reproduction check: the same probabilities to the last digit.

## Changes off the main chain

| Log row | Change | Against | CV | Δ |
|---|---|---|---|---|
| 8 | Leave the pool's gemini-3.7-flash and 3.8-flash rows out (75,943 pool rows) | Row 6 | 0.4482 | -0.030 |
| 9 | Witness features (Qwen3-4B-Base likelihoods, tokens per word) blended with the row 6 SVC OOF at weight 0.35 | Row 6 | 0.1656 alone | +0.005 at the peak, about +0.003 expected |
| 7 | Nishant Dahal's Qwen3-14B checkpoint through our loader | n/a | in-sample 0.858 | not a score |

Row 7 checks the loader only. The checkpoint saw every train passage, so its train accuracy is not CV. Its own published CV is 0.611.

## Leaderboard probes

Each probe changes one thing against the one before it (#20).

| Probe | Change | Public LB | Δ |
|---|---|---|---|
| 1 | Nishant's published spine (Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25, centred twins 0.1) with our pool SVC in place of his, full-strength Sinkhorn | TODO | TODO (target: within 0.003 of his 0.620) |
| 2 | Half-strength Sinkhorn (alpha 0.5) | TODO | TODO |
| 3 | Add prvsiyan's two Qwen3-8B seeds at a fixed 10% | TODO | TODO |

## What the table says

- Data did most of the work. The pool gave +0.115 at weight 1 and down-weighting it gave +0.020 more. The SVC itself never changed.
- More rows beat cleaner labels: the free proxy rows still helped (rows 4 and 5).
- The pool-weight curve has flattened, so we stopped at 0.25.
- Leaving suspect labels out hurt more than keeping them (row 8).
- The likelihood witnesses carry real but small signal, about +0.003 on the SVC.
