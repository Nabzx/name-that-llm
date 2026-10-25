# Citations

Every public notebook, discussion, dataset and pretrained model used or read for this entry.

## Notebooks and discussions

- avazaku, *Name That LLM, Peer Correction: 0.631*. https://www.kaggle.com/code/avazaku/name-that-llm-fingerprint-ensemble. Read for leave-one-out peer correction in place of Sinkhorn.
- Roman Tamrazov, *Name That LLM, Who am I?*. https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i. Read for UltraBERT use and consensus-gated reranking.
- Vladislav Lavrentev, *0.642 LoRA + Llama + Qwen3.8-27B*. https://www.kaggle.com/code/lightningv08/0-642-lora-llama-qwen3-8-27b. Read, not copied, for how the full-text Qwen3.8-27B classifier is run (#46).
- Vladislav Lavrentev, *0.637 LoRA Llama Qwen3.8-27B*. https://www.kaggle.com/code/lightningv08/0-637-lora-llama-qwen3-8-27b. Read to see what the 0.637 to 0.639 stacks contain.
- gougou1, *ntllm v9 final*. https://www.kaggle.com/code/gougou1/ntllm-v9-final. Read for in-style LoRA plus Gaussian group heads.
- s0chn1y, *Sinkhorn balancing on an uneven test*. https://www.kaggle.com/code/jusill/sinkhorn-balancing-on-an-uneven-test. Read for the risk of full balancing.
- zhuyu2, *ntl 607 l24x w10*, and arlearonreidma's 0.636 to 0.639 forks. https://www.kaggle.com/code/zhuyu2/ntl-607-l24x-w10. Read to confirm the public stacks share one lineage.
- Discussions 752365 (Sinkhorn on an uneven test), 752357 (what 134 public versions show), 751871 (what transferred to the LB) and 752197 (the GLM and Fable corner). https://www.kaggle.com/competitions/name-that-llm/discussion/752357. Read for the plan to 0.64.

- Lucas Boesen, *LoRA x Adaptive Centering*. https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering. Read for half-strength balancing and centring (#23).
- prvsiyan, *The Prompt Is Not the Author*. https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author. Read for the Gaussian group head and test-time grouping (#23).
- Zbigniew Kirchner, *0.574 In-Style Qwen3-8B LoRA (fully public)*. https://www.kaggle.com/code/zbigniewkirchner/0-574-in-style-qwen3-8b-lora-fully-public. Read for how his in-style checkpoint is loaded (#28).

- HALX2158 and replies, *Forked a 0.600+ notebook and hit the 9-hour limit? Check the Qwen3.5-9B load time*, Name That LLM discussion 751268. https://www.kaggle.com/competitions/name-that-llm/discussion/751268. Read for load times from Datasets against Kaggle Models (#22).

- Nishant Dahal, *Two-stage LoRA, part two* (LB 0.620). https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two. Read for the LoRA24 readout (#21) and the blend weights.
- Nishant Dahal, *Two-stage LoRA on the public pool* (LB 0.592). https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool. Read for the training recipe.

- Adisak Sukul and replies (GoPain), *New here? Grouped CV that tracks the LB, CPU baseline code (LB 0.353), pitfalls, 2xT4 tips*, Name That LLM discussion 750390. https://www.kaggle.com/competitions/name-that-llm/discussion/750390. Read for train-only baseline scores and the CV to LB gap.

## Datasets

- Name That LLM: extra GPT and Gemini generations (abdullah1afify/ntl-generations), CC BY 4.0. https://www.kaggle.com/datasets/abdullah1afify/ntl-generations. Read in the pool census (#19).
- NTL Vertex Gemini + proxy GPT generations (alperen5252525/ntl-vertex-gemini-proxy-gpt-generations), CC BY 4.0. https://www.kaggle.com/datasets/alperen5252525/ntl-vertex-gemini-proxy-gpt-generations. Read in the pool census (#19).
- NTL Curated Pool (86.5k, route-filtered) (ankiteshtiwari/ntl-curated-pool-routed), unknown. https://www.kaggle.com/datasets/ankiteshtiwari/ntl-curated-pool-routed. Read in the pool census (#19).
- ntl-kaggle-ai-gen (darragh1walsh/ntl-kaggle-ai-gen-run-1), CC0-1.0. https://www.kaggle.com/datasets/darragh1walsh/ntl-kaggle-ai-gen-run-1. Read in the pool census (#19).
- Name That LLM: free-proxy generations (dophuchung/ntl-proxy-generations), CC BY 4.0. https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations. Read in the pool census (#19).
- NTL Quality Selected Public SVC Pool (fleongg/ntl-quality-selected-public-svc-pool), other. https://www.kaggle.com/datasets/fleongg/ntl-quality-selected-public-svc-pool. Metadata read for #19.
- Name That LLM: Same-Prompt Passages (Free Quota) (haileycheng/name-that-llm-same-prompt-passages), CC BY 4.0. https://www.kaggle.com/datasets/haileycheng/name-that-llm-same-prompt-passages. Read in the pool census (#19).
- NTL: extra GPT passages via free Kaggle proxy (kaplinskiy/ntl-gpt-proxy-passages), CC BY 4.0. https://www.kaggle.com/datasets/kaplinskiy/ntl-gpt-proxy-passages. Read in the pool census (#19).
- New GPT Responses on 100 Independent Prompts (keshavatearth/ntl-gpt-new-prompts-v1), CC BY 4.0. https://www.kaggle.com/datasets/keshavatearth/ntl-gpt-new-prompts-v1. Read in the pool census (#19).
- NTL free-proxy GPT and Gemini generations (lucasboesen/ntl-free-proxy-generations), CC BY 4.0. https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations. Read in the pool census (#19).
- NTL synthetic generations v1 (13 labels) (lxdlxdlxd/ntl-synth-generations-v1), CC BY 4.0. https://www.kaggle.com/datasets/lxdlxdlxd/ntl-synth-generations-v1. Read in the pool census (#19).
- Name That LLM: OpenRouter + OpenAI Generations (maniceet/name-that-llm-openrouter-openai-generations), CC BY 4.0. https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations. Read in the pool census (#19).
- NTL own extra v1 (michaelsemenoff/ntl-own-extra-v1), CC BY 4.0. https://www.kaggle.com/datasets/michaelsemenoff/ntl-own-extra-v1. Read in the pool census (#19).
- NTL two-stage LoRA part two (nishantdahal/ntl-handwriting-blend), CC-BY-SA-4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend. Read in the pool census (#19).
- NTL two-stage LoRA checkpoints and public pool (nishantdahal/ntl-two-stage-lora), CC-BY-SA-4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora. Metadata read for #19.
- Name That LLM: Response Corpus (olegzholobov/name-that-llm-response-corpus), other. https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus. Read in the pool census (#19).
- Name That LLM: own-prompt generations v1 (prvsiyan/ntl-own-prompt-generations-v1), CC-BY-SA-4.0. https://www.kaggle.com/datasets/prvsiyan/ntl-own-prompt-generations-v1. Read in the pool census (#19).
- NTL public in-style pool v1 (prvsiyan/ntl-public-in-style-pool-v1), CC-BY-SA-4.0. https://www.kaggle.com/datasets/prvsiyan/ntl-public-in-style-pool-v1. Read in the pool census (#19).
- prvsiyan's Extended Name That LLM Dataset (prvsiyan/prvsiyans-extended-name-that-llm-dataset), CC BY 4.0. https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset. Read in the pool census (#19).
- Name That LLM: Extra API Passages (seyitkaangunes/ntl-extra-api-passages), CC BY 4.0. https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages. Read in the pool census (#19).
- Name That LLM OpenRouter Harvest (willrice/name-that-llm-openrouter-harvest), CC BY 4.0. https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest. Read in the pool census (#19).
- Name That LLM: Extra GPT + Gemini Passages (zbigniewkirchner/ntl-extra-gpt-passages), CC BY 4.0. https://www.kaggle.com/datasets/zbigniewkirchner/ntl-extra-gpt-passages. Read in the pool census (#19).
- NTL free-proxy generations (GPT + Gemini) (ziyuzhou1207/ntl-zz-proxy-generations), CC BY 4.0. https://www.kaggle.com/datasets/ziyuzhou1207/ntl-zz-proxy-generations. Read in the pool census (#19).

## Pretrained models

- Nishant Dahal, *NTL two-stage LoRA part two* (nishantdahal/ntl-handwriting-blend): Qwen3-14B and Qwen2.5-14B LoRAs, DeBERTa-v3-large. CC BY-SA 4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend. Read for #18.
- Nishant Dahal, *NTL two-stage LoRA checkpoints and public pool* (nishantdahal/ntl-two-stage-lora). CC BY-SA 4.0. https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora. Read for #18.
- prvsiyan, *NTL two-stage LoRA adapters v1* (prvsiyan/ntl-two-stage-lora-adapters-v1). CC BY-SA 4.0. https://www.kaggle.com/datasets/prvsiyan/ntl-two-stage-lora-adapters-v1. Read for #18.
- Zbigniew Kirchner, *In-Style Qwen3-8B LoRA* (zbigniewkirchner/ntl-qwen3-8b-lora-instyle). Apache 2.0. https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-instyle. Read for #18.
- Zbigniew Kirchner, *Qwen3-8B LoRA + Claude Head* (zbigniewkirchner/ntl-qwen3-8b-lora-claude-head). Apache 2.0. https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head. Read for #18.
- Joseph Adamski, *NTL ATLAS Qwen3 8B v1* (josephadamski91/ntl-atlas-qwen3-8b-v1). Weights CC BY-SA 4.0, code Apache 2.0. ATLAS method by Ko et al., arXiv 2610.19583. https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1. Read for #18.
- pjmathematician, *UltraBERT-380M Name That LLM 5-fold* (pjmathematician/ultrabert-380m-ntl). Apache 2.0. https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl. Read for #18, run as a member in #28.
- nuridudak, *NTL Two-Stage Encoders: ModernBERT + DeBERTa* (nuridudak/ntl-two-stage-encoders). Apache 2.0. https://www.kaggle.com/datasets/nuridudak/ntl-two-stage-encoders. ModernBERT-large (Answer.AI and LightOn, Apache 2.0) and desklib/ai-text-detector-v1.01 (MIT, on Microsoft's DeBERTa-v3-large), trained on nuridudak/ntl-130k-clean-passages (CC BY-SA 4.0, merged from 45 public datasets, mostly CC BY 4.0). Run as members through our own loader in #28; light_members.npz not used.
- nuridudak, *0.624 | When LLM Decoders Meet BERT Encoders*. https://www.kaggle.com/code/nuridudak/0-624-when-llm-decoders-meet-bert-encoders. Read, not copied, for how the encoders are blended (#28).
- Dhruv Tibarewal, *nlm-l24-adapters* (dhruvtibarewal/nlm-l24-adapters). MIT. https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters. Read for #18.
- Vladislav Lavrentev, *Name That LLM Artifacts* (lightningv08/ntl-olmo2-native24-final-inference-assets), including the Qwen3.8-27B NF4 LoRA. Mixed licences per component. https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets. Read for #18.
  - Used in #46 through our own loader (`src/nthl/qwen27b.py`): the full-text Qwen3.8-27B classifier only, that is the NF4 base `name-that-llm-artifacts/qwen38-27b-nf4` (greghavens/Qwen3.8-27B-bnb-4bit revision 26157380225e427827263c34df23018a1e28dff5, of Qwen/Qwen3.8-27B; its own Apache 2.0 LICENSE file), the task adapters `ntl627-own-fulltext/{final,outer0}.safetensors` and the offline wheels in `name-that-llm-artifacts/qwen38-offline-linux`. Licence of the 27B component, in the collection's own words: "Apache 2.0 terms apply to that model component." The adapter folder has no LICENSE file of its own. Wheels keep their packaged licences, printed by the check notebook. Nothing from the PriorLabs tabicl/tabular or Llama 3.1 parts is used.
- Lucas Boesen, notebook outputs `name-that-llm-train-lora24-s7`, `-train-tokattn`, `-train-tokattn2`, `-glm4-9b-trunc25`, `-extract-encoders2`, `-electra-large-weights`, `-desklib-weights`. https://www.kaggle.com/lucasboesen. Read for #18.
- Кирилл Тымчишин, *NTL Spectral Style Heads, Paired 4404* (kirilltym/ntl-spectral-style-heads-paired-4404). Licence unknown, so not used. Read for #18.
- fle3n, *NTL Legacy Qwen24 Recovery* (fleongg/ntl-legacy-qwen24-recovery). Licence "other", so not used. Read for #18.
- Qwen Team, base models from Kaggle Models, all Apache 2.0: Qwen3-14B-Base (`qwen-lm/qwen-3/Transformers/14b-base`), Qwen2.5-14B (`qwen-lm/qwen2.5/Transformers/14b`), Qwen3-8B-Base (`qwen-lm/qwen-3/Transformers/8b-base`, also the base of our own LoRA) and Qwen3-4B-Base (`qwen-lm/qwen-3/Transformers/4b-base`, the witness). https://www.kaggle.com/models/qwen-lm. Used as the trunks the published and our own adapters sit on.
- Google, Gemma 4 E2B (`google/gemma-4/transformers/gemma-4-e2b`). Gemma terms. Only its tokenizer is used, for the tokens-per-word witness feature.
- dddmdd, *Name That LLM fine-tuned BERT-family encoders* (dddmdd/ntl-finetuned-encoders). CC BY-SA 4.0. https://www.kaggle.com/datasets/dddmdd/ntl-finetuned-encoders. Read, not used.
- Travis Nguyen, notebook output `vnton37nguyn/ntl-train-lora-qwen25-instyle`. https://www.kaggle.com/code/vnton37nguyn/ntl-train-lora-qwen25-instyle. Read, not used.

## Read for the pool and today's checks

- rmsgrk, *NTL Pool Audit: Keep the Whole Pool* (rmsgrk/ntl-pool-audit). CC BY-SA 4.0. https://www.kaggle.com/datasets/rmsgrk/ntl-pool-audit. Read for the 122k pool test (LOG row 27).

## Our own artefacts

Our code (this repository) is Apache 2.0. Our trained weights (`nabilshah/nthl-train-lora`) and the code dataset `nabilshah/nthl-src` stay private and are shared with the organisers on request. Anything we publish that is derived from a CC BY-SA 4.0 dataset would carry CC BY-SA 4.0.
