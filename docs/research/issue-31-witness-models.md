# Issue #31: Which open witness models fit on two T4s

## Question

For each lab, is there an open model on Kaggle Models that can score our passages on two T4s (15 GB each, fp16)?

## What we looked at

Kaggle Models listings and model pages, the model cards, and what #17 and #22 taught us about the Kaggle image (no bitsandbytes, so no 4 or 8 bit).

## Findings

| Lab | Open model on Kaggle | Fits on two T4s in fp16? | Verdict |
|---|---|---|---|
| Qwen | Qwen3-4B-Base, 8B-Base, 14B-Base (`qwen-lm/qwen-3/Transformers/4b-base/1` and up) | Yes. 4B on one T4, 14B over both | Witness: Qwen3-4B-Base, cheap and a base model |
| Google | Gemma 4 E2B, E4B, 12B, 26B A4B, 31B (`google/gemma-4/transformers/gemma-4-e4b/1`, base) | E4B yes (about 16 GB with its per-layer embeddings, so over both T4s), 12B yes over both | Witness: Gemma 4 E4B base |
| OpenAI | gpt-oss-20b (Unsloth's copy) | No. 21B parameters is about 42 GB in fp16, and its 4 bit format needs kernels the T4 lacks | Tokenizer only: o200k, the tokenizer of OpenAI's recent models |
| Moonshot | Moonlight-16B-A3B (third-party upload) | No. 16B is about 32 GB in fp16 | Tokenizer only |
| ZAI | GLM-4-9B (third-party uploads) | 9B is about 18 GB, so over both T4s, but the uploads are old chat formats that need remote code | Later, if time allows |
| Anthropic | none | | No witness |
| Neutral | Llama 3.1 8B | Yes, over both | Optional reference for cross-perplexity |

Cost is small. A 4B witness reads about 6,700 passages (train plus test) of about 170 tokens in a few minutes on a T4. Scoring the 84k pool as well would take about an hour per witness.

## Recommendation

Prototype (#34) with two likelihood witnesses: Qwen3-4B-Base and Gemma 4 E4B base. Add tokens per word under each lab's tokenizer, which costs nothing and reaches OpenAI and Moonshot too. Fit the head on host train only, on the frozen folds, and judge it by whether it adds to our SVC's out-of-fold predictions. GLM and Llama come later if the prototype shows signal.
