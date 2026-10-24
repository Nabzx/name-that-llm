# Issue #33: Which likelihood features to extract

## Question

Which per-passage features should each witness give the head, and what does it cost to compute them for about 6,700 passages (train plus test) on two T4s? Keep the list short.

## What we looked at

- Our #32 note: what ATLAS, Fast-DetectGPT, Binoculars, GLTR, Sniffer and the statistics ensemble in the ATLAS paper used, and how they did as namers.
- Our #31 note: the witnesses (Qwen3-4B-Base, Gemma 4 E4B base), tokenizers only for OpenAI and Moonshot, no bitsandbytes on the image.
- `abh001/ntl-frozen-lm-features-train`: six log-likelihood statistics per train passage from Qwen3-1.7B-Base and Qwen3-4B-Base, 512 tokens at most, fp16 on a T4.
- Public notebooks in this competition. Only the ATLAS adapter uses likelihood, and none of the read notebooks use hand-made log-likelihood statistics.
- Train passage lengths: mean 115 words, median 101, 10th percentile 48, 90th 206, longest 274.

## Findings

- **Several statistics beat one score.** In the ATLAS appendix one scalar per text names about 45% of five classes, eight averaged statistics from one model reach 74%, and four witnesses from different families reach 76% and hold up best under shift.
- **Short passages need steady statistics.** Quantiles, log-rank and top-1 share move less on 50 tokens than the mean log-prob. Most of what the public dataset computes is on our list already.
- **Compare witnesses per character.** Each witness has its own tokenizer, so only a per-character or per-word figure lines up across them. A linear head on both witnesses' features side by side learns the difference itself, so no separate difference features.
- **Movement through the passage is noise here.** Half of a median passage is about 70 tokens. A first-half against second-half drift on that is mostly noise, and the token log-prob spread (std, quantiles) already holds what GPT-who's information density features found. Dropped.
- **Tokenizers are free.** Tokens per word under each lab's tokenizer costs seconds on CPU and reaches OpenAI (o200k, from gpt-oss) and Moonshot (Kimi), which have no witness. No paper tests it as an attribution feature, so it is a cheap bet.
- **The test has no prompt text.** Score the passage alone, after the beginning-of-text token, and skip the first token's log-prob.

## Recommendation

### The feature list

Per likelihood witness, all from one forward pass and its logits, over every token after the first:

| # | Feature | What it is |
|---|---|---|
| 1 | `lp_mean` | mean token log-prob |
| 2 | `lp_std` | std of token log-prob |
| 3 | `lp_q10` | 10th percentile of token log-prob |
| 4 | `lp_q50` | median token log-prob |
| 5 | `ent_mean` | mean entropy of the next-token distribution |
| 6 | `fdg` | Fast-DetectGPT curvature, `sum(lp + H) / sqrt(sum Var)`, with Var the variance of log-prob under the model's own distribution |
| 7 | `logrank_mean` | mean of log(1 + rank of the true token) |
| 8 | `top1` | share of tokens that are the model's top pick |
| 9 | `bpc` | minus the summed log-prob over the number of characters, the one figure that lines up across witnesses |

That is 9 features for each of Qwen3-4B-Base and Gemma 4 E4B base, 18 in all. Then:

- **Tokenizers (5).** `tpw_*`: tokens per word under o200k (OpenAI, gpt-oss), Gemma 4 (Google), Qwen3 (Qwen), GLM-4 (ZAI) and Kimi (Moonshot). Anthropic has no public tokenizer for its recent models.
- **Length (1).** `log_words`, since every statistic above shifts with length.
- **Optional (1).** `bino`: Binoculars score with Qwen3-4B-Base as observer and the post-trained Qwen3-4B as performer, which share a tokenizer. Add it in a second round if the first shows signal.

24 features, 25 with Binoculars. Standardise them within each fold and fit a logistic regression on host train with the frozen folds. Judge it by the gain when stacked on the SVC's out-of-fold predictions.

Left out: perplexity and the log-rank ratio (both follow from features 1 and 7), GLTR rank buckets (covered by 7 and 8), drift through the passage, DetectGPT rewrites.

### How to compute them

- One code path for train and test, inside the scoring notebook. The public train dataset cannot feed the final head: it has no test twin and its settings may differ from ours.
- Length-sorted batches, right padding, fp16 weights. Logits to fp32 in chunks of a few thousand tokens: a full batch of 262k-wide logits does not fit. From each chunk: log-softmax, the true token's log-prob, entropy, the variance term (`sum p * lp^2 - H^2`) and rank (`(logits > true logit).sum() + 1`).
- Assert every value is finite. Gemma overflows fp16 more often than Qwen. Redo a failing batch in fp32 on its own, as the public notebooks do.
- Copy weights to local disk before loading (#22).

### The cost

About 6,700 passages of 170 tokens each is 1.1M tokens. A forward pass costs about 2 × parameters × tokens. We assume a T4 reaches 15 to 25 of its 65 fp16 TFLOPS on length-sorted batches. These are estimates, not timings.

| Step | Work | Where | Estimate |
|---|---|---|---|
| Qwen3-4B-Base | 8.8 PFLOP | one copy on each T4 | 3 to 5 min, plus 1 to 2 min to load 8 GB |
| Gemma 4 E4B base | about 11 PFLOP with its 262k output head | split over both T4s | 8 to 12 min, plus 2 min to load; several times longer if many batches need fp32 |
| Statistics over the vocabulary | about 3 × 10^11 values, a few passes | GPU, memory bound | under 1 min per witness |
| Five tokenizers | 6,700 passages | CPU | under 1 min |
| Binoculars performer, optional | another 8.8 PFLOP, alongside the observer | one model per T4 | adds 3 to 5 min |

All in, 15 to 25 minutes of GPU for train plus test, about 30 with Binoculars and loading. The 4,000 test passages are 60% of that inside the scoring run, well within the 9-hour limit and #22's 4-hour aim. #34 should time one commit run on train to replace these estimates.

Before any GPU run, #34 can test the idea on CPU in seconds: stack the six Qwen3-4B-Base statistics from the public train dataset on our SVC out-of-fold predictions, on Kaggle and on the frozen folds. If they add nothing, the witnesses are unlikely to.

## Sources

- Our #32 (likelihood attribution) and #31 (witness models).
- Ko et al., *ATLAS*, arXiv 2610.19583, Appendix B (baseline features). https://arxiv.org/abs/2610.19583
- Bao et al., *Fast-DetectGPT*, arXiv 2310.05130. https://arxiv.org/abs/2310.05130
- Hans et al., *Binoculars*, arXiv 2401.12070. https://arxiv.org/abs/2401.12070
- Gehrmann et al., *GLTR*, arXiv 1906.04043. https://arxiv.org/abs/1906.04043
- Su et al., *DetectLLM*, arXiv 2306.05540. https://arxiv.org/abs/2306.05540
- Li et al., *Sniffer*, arXiv 2304.14072. https://arxiv.org/abs/2304.14072
- abh001, *Name That LLM frozen LM features (train)*. https://www.kaggle.com/datasets/abh001/ntl-frozen-lm-features-train
- NVIDIA T4 data sheet: 65 TFLOPS fp16 tensor, 320 GB/s, 16 GB. https://www.nvidia.com/en-gb/data-center/tesla-t4/
