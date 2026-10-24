# Issue #32: What ATLAS and likelihood-based detectors tell us

## Question

Phase 2 scores each passage with one open model per lab and feeds likelihood statistics to a small head. What does the literature say about using likelihood to name the model, not just to spot AI text? What carries over to passages of 40 to 250 words (train median 101 words, 10% under 48)?

## What we looked at

- ATLAS (Ko et al., arXiv 2610.19583): method, main table, ablations and Appendix B on how the baselines were built.
- The detectors it compares or cites: DetectGPT, Fast-DetectGPT, Binoculars, GPT-who, Sniffer, a statistics ensemble, DNA-GPT. Also Ghostbuster, LLMDet and Mireshghallah et al. on which proxy models detect best.
- In this competition: Joseph Adamski's *Teach the Model + 10% ATLAS* and his public ATLAS adapter, kirilltym's 0.639 notebook, and the `abh001/ntl-frozen-lm-features-train` dataset.
- Our #31 note on which witnesses fit on two T4s.

## Findings

### ATLAS

- **Method.** One frozen base model (Qwen2.5-7B, also Qwen3-8B) with one condition per source: a learnt prefix plus a per-source LoRA up-projection on a shared down-projection. Each condition is trained only on the plain next-token loss of its own source's texts. No classification loss.
- **Scoring.** Score the text under every condition. Per token, compare each source's log-likelihood with every rival and clip: `r_kt = exp(-(1/T) * sum_j max(l_jt - l_kt, 0))`, so `r_kt = 1` when source k explains that token best. Average over tokens and over T in {1, 2, 4, 8}, pick the largest. No single surprising token can swing the vote.
- **Results.** Five families, passages of 90 to 220 words, close to ours. In distribution everything trained scores about 99.5%. The gap shows under shift: 74.3% against 65.3% for the best encoder when both domain and model version change. Clipping is worth +4.6 to +5.1 points under shift, the per-source LoRA only +0.8 to +2.2. Errors under shift run towards Claude: up to 52% of Gemini texts and 24% of GPT texts.
- **Distillation.** It finds the teacher of every distilled model tried. Shared training data leaves a likelihood trace, which is the bet behind lab witnesses.

### Zero-shot likelihood statistics are weak namers

ATLAS's appendix runs the classic detectors as five-class attributors (chance 20%), all on Qwen2.5-7B:

| Method | Features | In distribution | Domain and version shift |
|---|---|---|---|
| Fast-DetectGPT | one curvature scalar, logistic regression | 44.4 | 31.9 |
| Binoculars | one perplexity / cross-perplexity ratio | 45.6 | 32.6 |
| GPT-who | surprisal spread statistics | 58.6 | 39.9 |
| StatEnsemble | 8 averaged stats (log-prob, rank, log-rank, entropy, LRR, the two above) from base and instruct Qwen, MLP | 74.4 | 32.3 |
| Sniffer | per-word NLL from four different model families, pairwise contrasts, MLP | 76.0 | 47.6 |
| Fine-tuned encoders | text | 99.5 | 63 to 65 |

- One scalar per text carries about two classes of signal. Several averaged statistics from one model reach 74%. Four witnesses from different families do best and hold up best under shift.
- Detectors answer "is this text likely under some model?". Naming needs the contrast between candidates. Sniffer and ATLAS both win by comparing models against each other, not by reading one model's level.

### The detectors, and what each lends us

- **DetectGPT** compares the log-likelihood with that of about 100 rewrites. Far too slow for us.
- **Fast-DetectGPT** gets the same curvature from one pass: how far the text's log-likelihood sits above what the model expects from its own samples, divided by the spread, `(sum logp + sum H) / sqrt(sum Var)`. Free once we have the logits.
- **Binoculars** divides log-perplexity by cross-perplexity between two models that share a tokenizer. It cancels how surprising the topic is. Its authors note the edge grows on short texts. It needs two models with one tokenizer.
- **GLTR, DetectLLM.** Token ranks and log-ranks are steadier than raw log-probabilities and cheap.
- **Ghostbuster** feeds token probabilities from weak models into a structured search over feature combinations, then logistic regression. The search would overfit 2,699 rows. The idea of several weak models does carry over.
- **LLMDet** traces the source with proxy perplexity per candidate model. Same idea as Sniffer, with cached n-gram probabilities instead of the models.
- **DNA-GPT** regenerates the end of a text with the candidate model and compares. It needs the closed models, so it is out.
- **Mireshghallah et al.** Small, partly trained models make the best universal detectors, and sharing training data with the generator is "not critically important". So an own-lab witness is a hypothesis to test, not a given.

### In this competition

- Joseph Adamski's public ATLAS adapter (Qwen3-8B-Base, 18 conditions, official train only, five epochs) adds +0.002 to +0.007 on the public LB at 10% weight, on seven different notebooks. It sits inside the 0.636 and 0.639 stacks. Learnt likelihood adds a little on top of LoRA classifiers, it does not replace them.
- Scoring 18 conditions on an 8B model is the costly part. kirilltym runs it under a time budget, shortest passages first.
- `abh001/ntl-frozen-lm-features-train` already holds six log-likelihood statistics per train passage for Qwen3-1.7B-Base and Qwen3-4B-Base: mean, std and 10th percentile of log-prob, mean entropy, top-1 share, mean of log-prob plus entropy. Train only, so test features must still be computed in our notebook.

### What carries over to 40 to 250 words

- **Noise.** A mean over 65 tokens is about 1.6 times as noisy as one over 170. Quantiles, ranks and clipped scores hold up better than the mean log-prob on the shortest passages.
- **Topic swamps author.** A passage's likelihood depends more on its prompt than on who wrote it. Ratios that cancel the topic (Binoculars, Fast-DetectGPT's z-score, differences between witnesses) matter more than levels.
- **Tokenizers differ.** Witness statistics must be compared per word or per character, not per token, as Sniffer aligns to words.
- **Expect a small gain.** Even the strong form (ATLAS) adds under 0.01 on the LB. Our zero-shot witnesses are the weak form. They earn a place only if they add to the SVC's out-of-fold predictions.

## Recommendation

1. Keep the two witnesses from #31 (Qwen3-4B-Base, Gemma 4 E4B base) and the free tokenizer counts. Add a third witness only if #34 shows the first two add signal.
2. Per witness, take a few steady statistics, not one score: log-prob quantiles, entropy, the Fast-DetectGPT curvature, log-rank, top-1 share and bits per character. Put both witnesses' features side by side so a linear head can learn the contrast. Use Binoculars only inside one tokenizer family. #33 fixes the list and the cost.
3. Fit a logistic regression on standardised features, on the frozen folds, and judge it by the gain when stacked on the SVC's out-of-fold predictions.
4. Before spending GPU, have #34 check the signal cheaply: the six Qwen statistics in the public train dataset, stacked on our SVC OOF, on Kaggle. If they add nothing, drop the witness track early.
5. Skip DetectGPT, DNA-GPT and Ghostbuster's feature search.
6. The ATLAS adapter itself is the stronger likelihood member, worth +0.002 to +0.007 at 10%. That is a spine question, not a witness one. Raise it with #29 if the time budget allows 18 passes of an 8B model over the test.

## Sources

- Ko et al., *Who's Behind the Text? Robust Model Authorship Attribution* (ATLAS), arXiv 2610.19583. https://arxiv.org/abs/2610.19583
- Mitchell et al., *DetectGPT*, arXiv 2301.11305. https://arxiv.org/abs/2301.11305
- Bao et al., *Fast-DetectGPT*, arXiv 2310.05130. https://arxiv.org/abs/2310.05130
- Hans et al., *Spotting LLMs With Binoculars*, arXiv 2401.12070. https://arxiv.org/abs/2401.12070
- Verma et al., *Ghostbuster*, arXiv 2305.15047. https://arxiv.org/abs/2305.15047
- Yang et al., *DNA-GPT*, arXiv 2305.17359. https://arxiv.org/abs/2305.17359
- Venkatraman et al., *GPT-who*, arXiv 2310.06202. https://arxiv.org/abs/2310.06202
- Li et al., *Origin Tracing and Detecting of LLMs* (Sniffer), arXiv 2304.14072. https://arxiv.org/abs/2304.14072
- Wu et al., *LLMDet*, arXiv 2305.15004. https://arxiv.org/abs/2305.15004
- Mireshghallah et al., *Smaller Language Models are Better Zero-shot Machine-Generated Text Detectors*, arXiv 2305.09859. https://arxiv.org/abs/2305.09859
- Gehrmann et al., *GLTR*, arXiv 1906.04043. https://arxiv.org/abs/1906.04043
- Su et al., *DetectLLM* (log-rank ratio), arXiv 2306.05540. https://arxiv.org/abs/2306.05540
- Joseph Adamski, *Teach the Model + 10% ATLAS*. https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas
- Joseph Adamski, *NTL ATLAS Qwen3 8B v1* (adapter). https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1
- kirilltym, *0.639 | Teach More, Keep the Signature*. https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature
- abh001, *Name That LLM frozen LM features (train)*. https://www.kaggle.com/datasets/abh001/ntl-frozen-lm-features-train
- Our #31 (witness models that fit on two T4s).
