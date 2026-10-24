# Issue #19: Which public passage pools exist and what is in them

## Question

Other Kagglers have published passages from the 18 models. Which pools exist, how were they made, what licence do they carry, and what should we train on?

## How we looked

Kaggle search plus the credit lists in the public notebooks gave 24 datasets. We read each one's metadata, then ran a census notebook on Kaggle (`notebooks/pool-census/`) that attached 21 of them and counted. Nothing was fitted and nothing was downloaded to the laptop.

A row counts as clean when it is 30 to 300 words, one paragraph, has no markdown, is not a copy of a train passage, has content TF-IDF cosine below 0.3 to every train passage, and does not use punctuation its model never uses in the host train set (curly quotes for Claude and GLM, em dashes for Gemini).

## The pools

How they were made matters. The free Kaggle Benchmarks proxy reaches 12 of the 18 models, but it does not show which model really answered, so its labels are less sure.

| Pool | Licence | How it was made | Models | Clean rows |
|---|---|---|---|---|
| nishantdahal/ntl-handwriting-blend (pool v4) | CC BY-SA 4.0 | Merge of 25 public datasets, already cleaned | 18 | 74,795 |
| prvsiyan/ntl-public-in-style-pool-v1 | CC BY-SA 4.0 | Merge of 15 public datasets | 18 | 38,197 |
| willrice/name-that-llm-openrouter-harvest | CC BY 4.0 | OpenRouter, pinned to each model's own provider | 13 | 12,986 |
| prvsiyan/prvsiyans-extended-name-that-llm-dataset | CC BY 4.0 | Official APIs, 510 prompts, logged | 18 | 12,524 |
| maniceet/name-that-llm-openrouter-openai-generations | CC BY 4.0 | Paid OpenRouter and OpenAI | 10 | 7,665 |
| seyitkaangunes/ntl-extra-api-passages | CC BY 4.0 | Official Anthropic API | 5 | 4,996 |
| michaelsemenoff/ntl-own-extra-v1 | CC BY 4.0 | Merge plus OpenRouter and proxy | 18 | 4,878 |
| lxdlxdlxd/ntl-synth-generations-v1 | CC BY 4.0 | Official APIs | 13 | 4,391 |
| lucasboesen/ntl-free-proxy-generations | CC BY 4.0 | Free proxy | 8 | 3,919 |
| alperen5252525/ntl-vertex-gemini-proxy-gpt-generations | CC BY 4.0 | Vertex for Gemini, proxy for GPT | 9 | 3,622 |
| zbigniewkirchner/ntl-extra-gpt-passages | CC BY 4.0 | Not stated | 7 | 1,951 |
| prvsiyan/ntl-own-prompt-generations-v1 | CC BY-SA 4.0 | Free proxy | 12 | 1,429 |
| darragh1walsh/ntl-kaggle-ai-gen-run-1 | CC0 | Kaggle's daily AI quota | 12 | 1,338 |
| ziyuzhou1207, abdullah1afify, keshavatearth, dophuchung, haileycheng, kaplinskiy | CC BY 4.0 | Free proxy | 5 to 12 | 892 to 1,003 each |
| olegzholobov/name-that-llm-response-corpus | other | External responses, wide format | 18 | 78,561 |
| ankiteshtiwari/ntl-curated-pool-routed | unknown | Merge, mostly of the olegzholobov corpus | 18 | 73,494 |

Clean rows are counted per pool, so the merges overlap with their sources. A third of clean texts sit in two or more pools.

## What stood out

- **Two pools are trouble.** The olegzholobov corpus has a median of 34 words, 40% of rows with line breaks and 21% with markdown. It ships a "quarantine" file for gpt-5.6-luna, and its licence is "other". The ankiteshtiwari pool is built on top of it and has no licence at all. All 8,473 texts in the census that carry two different labels come from these two.
- **The open-licence pools agree with each other.** Once those two are out, not one text carries two labels.
- **The open-licence union is 90,721 passages**, from 2,948 (gemini-3.7-flash) to 7,738 (gpt-5.6-luna) per model. That is 20 to 50 times the host data for each model.
- **The thin spots** are gemini-3.7-flash (2,948), claude-opus-5 (3,124), qwen3.8-max (3,301) and gemini-3.1-pro-preview (3,333). The free proxy cannot reach 3.7-flash at all.
- **The host style is easy to match.** Most open pools are already one paragraph, 90 to 140 words and markdown-free. prvsiyan's extended set is the exception: a median of 162 words and 14% of rows too close to a train passage.
- **The pool helps on the LB, but less than CV says.** In discussion 750390 a train-only SVM went from 0.361 to 0.423 on the LB with a pool, while grouped CV claimed a bigger jump. Expect CV to overstate pool-trained members by 0.02 to 0.04.

## Correction (#26)

The census overcounted. It turned every column into a string before picking the text, so it also read the raw API responses in files like prvsiyan's `provenance.jsonl` as extra passages. Those are near copies of the clean passages. The pool build in #26 reads only real string columns and lands at **84,470 passages**, 2,666 to 7,210 per model. Use that number, not 90,721.

## Recommendation

Build one pool from the open-licence datasets only:

1. Take every pool in the table except olegzholobov and ankiteshtiwari.
2. Apply the clean filter above, drop any text with two labels and de-duplicate on whitespace-normalised text. That gives about 90,700 passages.
3. Keep the source pool and how it was made as columns, so #26 can test official-API rows on their own against all rows. That settles whether the proxy labels hurt.
4. Train with balanced class weights. The pool is 2.6 times as deep for luna as for 3.7-flash.
5. Cite every pool used. Most are CC BY 4.0. Three are CC BY-SA 4.0 (Nishant's pool v4 and both prvsiyan pools), which matters only if we publish a derived pool, and we won't.

Treat any CV gain from the pool with care, and confirm the first pool-trained member on the LB before stacking more on it. That takes a submission, so it waits for Nabil's go-ahead.
