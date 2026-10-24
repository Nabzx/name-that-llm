# Issue #18: Which published checkpoints exist and what each one scores

## Question

Our direction builds on checkpoints other Kagglers have published. Which exist, what do they score alone, what did they train on, how big are they, and can we use them?

## How we looked

The inputs attached to the 12 public notebooks we read (#2), Kaggle dataset search, and each dataset's own description. Nothing was run. Scores are as reported by their authors: CV is 5-fold split by prompt unless noted, and "fold 0" means one held-out fold of 539 rows.

## The checkpoints

| Checkpoint | Who | Backbone | Trained on | Reported alone | Licence |
|---|---|---|---|---|---|
| `lora_qwen3_14b` in nishantdahal/ntl-handwriting-blend | Nishant Dahal | Qwen3-14B-Base, first 27 of 40 layers, LoRA r16 | Pool of 84k, then train | CV 0.611 | CC BY-SA 4.0 |
| `lora_qwen25_14b`, same dataset | Nishant Dahal | Qwen2.5-14B, first 32 of 48 layers | Pool of 35k, then train | CV 0.569 | CC BY-SA 4.0 |
| `deberta_v3_large`, same dataset | Nishant Dahal | DeBERTa-v3-large, full fine-tune | Pool of 61k, then train | CV 0.546 | CC BY-SA 4.0 |
| `lora8b_v1`, `lora14b_v1` in nishantdahal/ntl-two-stage-lora | Nishant Dahal | Qwen3-8B and 14B, truncated | Pool of 35k, then train | CV 0.569 each | CC BY-SA 4.0 |
| prvsiyan/ntl-two-stage-lora-adapters-v1 | prvsiyan | Qwen3-8B, first 24 layers, two seeds | Pool of 39k, then train | Grouped OOF 0.568 | CC BY-SA 4.0 |
| zbigniewkirchner/ntl-qwen3-8b-lora-instyle | Zbigniew Kirchner | Qwen3-8B, first 24 layers | Train plus 9.8k in-style passages | Fold 0: 0.516 | Apache 2.0 |
| zbigniewkirchner/ntl-qwen3-8b-lora-claude-head | Zbigniew Kirchner | Qwen3-8B, first 24 layers, plus a 5-way Claude head | Train, Claude head also on Claude API passages | Fold 0: 0.484, 0.493 with the Claude head | Apache 2.0 |
| josephadamski91/ntl-atlas-qwen3-8b-v1 | Joseph Adamski | ATLAS: 18 source adapters on frozen Qwen3-8B-Base | Train only, 5 epochs | +0.002 to +0.007 on the LB at 10% weight | CC BY-SA 4.0 weights, Apache 2.0 code |
| pjmathematician/ultrabert-380m-ntl | pjmathematician | UltraBERT-380M, 5 fold models | Train only | OOF 0.408 | Apache 2.0 |
| dhruvtibarewal/nlm-l24-adapters | Dhruv Tibarewal | Qwen LoRA24 adapters | Public sources plus train | Notebook LB 0.611 | MIT |
| Qwen3.8-27B NF4 LoRA in lightningv08/ntl-olmo2-native24-final-inference-assets | Vladislav Lavrentev | Qwen3.8-27B in 4 bit, last 8 layers adapted, full text | 28k external, then train | 10% of a 0.636 blend | Apache 2.0 for this part; the bundle mixes licences |
| Notebook outputs of lucasboesen (`train-lora24-s7`, `tokattn`, `tokattn2`, `glm4-9b-trunc25`, `extract-encoders2`, ELECTRA and Desklib weights) | Lucas Boesen | Qwen3-8B LoRA24, token attention heads, frozen GLM-4-9B and encoders | Train, some with free-proxy passages | Members of a 0.549 to 0.551 blend | Kaggle notebook outputs |
| kirilltym/ntl-spectral-style-heads-paired-4404 | Кирилл Тымчишин | Readout heads on the public LoRA24 features | Train plus 4.4k private passages | Part of the 0.639 notebook | unknown |
| fleongg/ntl-legacy-qwen24-recovery | fle3n | Older Qwen LoRA24 | Not stated | Part of the 0.636 notebook | other |

The best public notebooks are stacks of most of these. The 0.639 notebook attaches about 30 inputs between checkpoints, pools and base models.

## What stood out

- **One bundle carries the stack.** Nishant's part two holds the strongest single model (Qwen3-14B, 0.611) plus two weaker but different members. On their own those three blended to 0.642 CV and 0.620 on the LB. The bundle runs the public test in about 15 minutes on two T4s and documents its end token and label order.
- **Almost nothing is out of fold.** Nearly every checkpoint was trained on all 2,699 train passages, so its predictions on train are not honest. The exceptions: UltraBERT ships 5 fold models, the lightningv08 bundle includes an out-of-fold validation cache, and Zbigniew reports fold-0 numbers. This is the problem #20 has to solve.
- **Backbones are almost all Qwen.** Qwen3-8B and 14B, Qwen2.5-7B and 14B. Members agree on about 77% of top picks, so a different family (Gemma, Llama, GLM) would add the most diversity. That feeds #41.
- **Bigger stacks add little.** Going from Nishant's 0.620 to 0.639 took a 15% stack of older members, ATLAS at 10% and extra readout heads, for +0.019 in all.
- **Licences are mostly clear.** CC BY-SA 4.0, Apache 2.0 and MIT are all free and public. kirilltym's heads have no stated licence and fleongg's say "other". The lightningv08 bundle includes a component under PriorLabs non-commercial terms. We would leave those three out.

## Recommendation

Shortlist for the spine, in the order to add them (#27, #28):

1. **Nishant's part two** (Qwen3-14B LoRA, Qwen2.5-14B LoRA, DeBERTa-v3-large). Load it first. It is the whole core of the 0.620 notebook.
2. **prvsiyan's two Qwen3-8B seeds.** Same recipe, different trainer and pool, cheap to run.
3. **Zbigniew's in-style Qwen3-8B.** Trained differently (train plus in-style passages, no two-stage), and it added +0.005 on the LB as one more member.
4. **ATLAS**, if its runtime fits: 18 adapter passes per passage. Worth +0.007 on the LB at 10%.
5. **UltraBERT's 5 folds**, mainly for its real out-of-fold predictions, which help #20.

Leave out the unknown and "other" licences, and the non-commercial component. Treat the Qwen3.8-27B classifier as a later option: it needs bitsandbytes, which Kaggle's image lacks (#17).

Cite every checkpoint we load in `CITATIONS.md`, with its author and licence.
