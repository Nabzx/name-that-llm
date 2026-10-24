# Issue #22: Kaggle pitfalls when running published checkpoints

## Question

What breaks, slows down or silently goes wrong when we run published checkpoints on Kaggle, and what must our notebook do about it?

## What we looked at

Our own smoke runs (#17), the checkpoint files (#21), Nishant Dahal's part two notebook and discussion thread 751268 on runs hitting the 9 hour limit.

## Findings

**The environment.** Python 3.13, torch 2.11 with CUDA 12.8, transformers 5.16, peft 0.20. The checkpoints were saved with peft 0.21, so we merge adapters by hand and need no peft at all. bitsandbytes is missing, so no 4 or 8 bit loading without an offline wheel.

**Loading speed is the big risk.** In thread 751268 someone measured one T4 session:

| Load | Time |
|---|---|
| Qwen3.5-9B straight from a Dataset | over 600 s, stopped (22 s in a good session) |
| Same folder copied to /tmp first | 420 s copy, then 33 s load |
| Qwen3-8B attached as a Kaggle Model | 19 s |
| Qwen3-14B attached as a Kaggle Model | 37 s |

Reading large weights from a Dataset can crawl in some sessions, and one scoring run was 1.9 times slower than its commit run. Kaggle Models load fast. Nishant also found DeBERTa took 16 minutes from the Dataset mount and 25 seconds after a copy to local disk.

**Silent failures.**

- The end token: Kaggle's Qwen3-14B-Base tokenizer gives 151645, but training used 151643. Wrong here changes 14% of top picks.
- The label order: the checkpoints put claude-fable-5 after claude-sonnet-5.
- The final norm must stay off.
- A try/except around a member hides an out-of-memory error. The run still scores, just lower, and you only learn from the log.

**Numerics and memory on T4s.** No fast bf16, so run in fp16 with fp32 sums (`allow_fp16_reduced_precision_reduction = False`), and redo any batch that overflows in bf16. A 14B trunk is about 19 GB, so split it over both T4s. Sort passages by length to cut padding, and on out-of-memory redo the batch one passage at a time.

**The 3-row test.** In a commit run `test.csv` has 3 rows. In the scoring run it has about 4,000. Anything that needs many rows (Sinkhorn, topic clusters) must skip itself on tiny input, or it misbehaves.

**Offline.** The scoring run has no internet. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so nothing tries to reach the Hub.

## Recommendation

Every notebook that runs a checkpoint passes this checklist:

1. Base models come from Kaggle Models (`qwen-lm/qwen-3/Transformers/14b-base/1`, `qwen-lm/qwen2.5/Transformers/14b/1`). Anything large in a Dataset is copied to `/tmp` before loading.
2. End token and label order come from each checkpoint's `meta.json`, with assertions.
3. Final norm off. fp16 with fp32 sums, bf16 retry on overflow, one-at-a-time retry on out-of-memory.
4. No try/except around a member. A member that breaks stops the run.
5. Sinkhorn and clustering skip themselves below 50 test rows.
6. Hub offline flags set at the top.
7. Elapsed time logged after every member. Before the final submission, time a commit run on 2,699 passages (train) as a stand-in for the 4,000 in the scoring run, and keep the total well under 9 hours, aiming for 4 or less, to absorb a slow session.
