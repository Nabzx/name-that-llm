# Issue #21: How the LoRA24 readout works

## Question

How do the published LoRA checkpoints turn a passage into 18 probabilities? We need to write our own loader (#27), so we need every step and every shape, without copying anyone's code.

## What we looked at

Nishant Dahal's *Two-stage LoRA, part two* notebook (read only) and the `meta.json` and `adapter_config.json` files of his checkpoints in `nishantdahal/ntl-handwriting-blend`. prvsiyan's and Zbigniew's checkpoints follow the same recipe on Qwen3-8B, with their own file layouts.

## The recipe, step by step

Shapes are for `lora_qwen3_14b`. H is the hidden size, 5,120 for both 14B models.

| Step | What happens | Shape |
|---|---|---|
| 1. Tokenise | No special tokens, cut at `max_len` 447, then append the end token from `meta.json` (`eos_id` 151643, `<|endoftext|>`) | up to 448 ids per passage |
| 2. Trunk | Only the first `keep` decoder layers: 27 of 40 for Qwen3-14B, 32 of 48 for Qwen2.5-14B. The final norm is switched off (identity), because training ran without it | |
| 3. Merge the adapter | For every adapted linear layer (q, k, v, o, gate, up, down): W ← W + B·A × alpha/r. Here r is 16 and alpha 32, so the scale is 2 | A is [16, in], B is [out, 16] |
| 4. Run | Hidden states of every layer. Index 0 is the embeddings, index l is after decoder layer l | [batch, tokens, H] per layer |
| 5. Pool | At 8 depths from `meta.json` (2, 4, 6, 9, 13, 17, 22, 27 for Qwen3-14B), a masked mean over tokens. The first token is left out of the mean, because it soaks up attention whatever it is | [batch, H] per depth |
| 6. Normalise | One LayerNorm per depth, in fp32, with weights stored in `heads.safetensors` (`norms.*`) | [batch, H] per depth |
| 7. Concatenate | The 8 normalised vectors side by side | [batch, 40,960] |
| 8. Plain head | Linear: Z·Wᵀ + b, then softmax | W is [18, 40,960], output [batch, 18] |
| 9. Centred head (optional) | For test passages in a topic cluster of 2 to 18, subtract the cluster's mean Z, then a second linear head trained on such differences | same shapes |

DeBERTa-v3-large is simpler: the full encoder, a masked mean of the last layer (first token kept), one linear head [18, 1,024], `max_len` 384, run in fp32.

The 0.620 notebook blends in log space: SVC 1.0, Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25, each centred twin 0.1, then Sinkhorn balancing.

## Traps

- **Label order.** The checkpoints list `claude-fable-5` after `claude-sonnet-5`. Ours lists it first among the Claude labels. Columns must be reordered by the `labels` field of each `meta.json`. Skip that and claude-fable-5's probabilities land under another Claude model's name with no error.
- **End token.** Kaggle's Qwen3-14B-Base ships the chat model's tokenizer, whose end token is `<|im_end|>` (151645). Training used `<|endoftext|>` (151643). Use `eos_id` from `meta.json`, never the tokenizer's. Getting this wrong changed 14% of top picks in Nishant's tests.
- **Final norm.** Leave it switched off. With it on, the features no longer match the heads.
- **Memory.** 27 layers of a 14B model plus its embeddings is about 19 GB in fp16, more than one 15 GB T4. Split it over both T4s.
- **fp16 overflow.** T4s have no fast bf16, so run in fp16, and redo any batch that overflows in bf16.
- **No out-of-fold check.** The checkpoints saw every train passage, so a run on train is a smoke test, not a score. A correct loader should get almost every train passage right; a wrong label order or end token shows up as a big drop.

## Recommendation

Write our own small loader in `src/nthl/` for #27: load the first `keep` layers, merge the adapter, pool the 8 depths with the first token left out, apply the LayerNorms and the plain head, and reorder columns by `meta.json`. Test it against these traps, with an assertion on label order and end token. Smoke-test it by running the Qwen3-14B checkpoint on host train on two T4s: expect accuracy well above our 0.48 CV, because the model has seen these passages. Use the centred head later, once #23 says whether centring is worth it.
