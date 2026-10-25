# Video script: three minutes

For the hand-in video (#50). One speaker, one screen, recorded solo. About 440 spoken words, which is close to three minutes at a calm pace. Numbers come from `docs/experiments/LOG.md` (rows 1 to 27) and `docs/research/`. TODO marks a number we do not have yet.

## Before you record

Open these in browser tabs, left to right, so each section is one tab switch:

1. The GitHub repo, `Nabzx/name-that-llm`, on the README.
2. `docs/research/issue-38-glm-pair.md` on GitHub, scrolled to the style table.
3. `docs/experiments/LOG.md` on GitHub (rendered table).
4. The Kaggle notebook `nthl-train-lora`, on the cell with `EOS = 151643` and the stage 1 and stage 2 loop.
5. `docs/report/pipeline.svg`, opened full screen.
6. The competition leaderboard, filtered to our team, plus our submissions page.

Record each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take. Read the TODO lines last, once probe 9 and the final score are in.

Say "glm-5.3" as "G L M five point three".

## Script

### 1. The task (0:00 to 0:17), about 45 words

> Hi, I'm Nabil, and this is my solo entry to Name That LLM. Each passage was written by one of eighteen models, scored on balanced accuracy. Labs are easy to tell apart. Siblings are not: glm-5.3 against its flash model, and Gemini 3.7-flash against 3.8-flash.

**On screen:** tab 1, the repo README, then tab 2, the GLM style table. Point at the semicolon row: the only difference between the two GLMs.

### 2. CV first and our SVC (0:17 to 0:40), about 50 words

> Before any model, I froze the folds: GroupKFold on the prompt, since all eighteen models answer the same prompts. Every experiment gets a row in this log. My first model, an n-gram SVM on host data, scored 0.348. Adding eighty-four thousand public passages, at a quarter weight, took it to 0.484.

**On screen:** tab 3, `LOG.md`. Highlight 0.3485 in row 1, then 0.4838 in row 11.

### 3. Published LoRAs and our own (0:40 to 1:05), about 60 words

> The strongest public members are LoRA fine-tunes of Qwen. I wrote my own loader for them, and it caught two traps: the end token the model was trained with is not the tokenizer's default, and each checkpoint orders the labels its own way. Then I trained my own Qwen3-8B LoRA overnight on Kaggle. Its honest score is 0.519, above my SVC.

**On screen:** tab 4, the Kaggle notebook. Scroll slowly from `EOS` to the two training stages. Then tab 3, row 19, highlight 0.519.

### 4. The probes (1:05 to 1:35), about 65 words

> Then leaderboard probes, one idea each. The published spine with my SVC scored 0.626. Balancing at half strength: 0.629. Adding prvsiyan's two seeds: 0.631. Then four probes in a row landed on exactly 0.631, with new members that each changed real rows. That tells me the public slice is too small to show gains of this size, so I trust cross-validation more than this board.

**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 8. Cut to tab 3, rows 12 to 26, if the submissions page is hard to read.

### 5. The biggest gain (1:35 to 2:05), about 60 words

> Here is the pipeline. Members blend in log space, then take fixed shares, then balancing. The blue boxes are ours. The biggest measured gain is a shared-prompt Gaussian group head, our own code after prvsiyan's public method. Passages that answer the same prompt are decoded together, with the prompt's effect taken out. On its own honest view it added 0.058.

**On screen:** tab 5, `pipeline.svg`. Point at each box as you name it, ending on the Gaussian group head. Then tab 3, row 25, highlight +0.058.

### 6. One honest failure (2:05 to 2:25), about 45 words

> One failure. Peer correction, a public trick, swaps balancing for removing the lean that passages on one topic share. It helped weaker blends. On ours it scored 0.622, our worst probe. Without class quotas the blend drifts towards the models it already favours.

**On screen:** tab 3, row 18. Highlight 0.622.

### 7. What came back flat (2:25 to 2:40), about 40 words

> Some things came back flat, and that is useful too. A prompt-centred SVC, pair specialists for the two hardest pairs, and a bigger public pool all stayed within noise. A 27-billion model took three seconds per passage, too slow.

**On screen:** tab 3, rows 24 and 27, then the PR list on GitHub showing the hard-pairs and Qwen 27B pull requests.

### 8. Next, and the result (2:40 to 3:00), about 55 words

> TODO probe 9: "The last probe put the Gaussian head at full weight and added ModernBERT at ten per cent. It scored ___."
>
> Next I would add a model from outside the Qwen family, and test each step on folds that look like the test. TODO final: "The final submission scored ___. Thanks for watching."

**On screen:** tab 6, the leaderboard with our row highlighted.

## Timing check

| Section | Start | Words |
|---|---|---|
| 1. The task | 0:00 | 45 |
| 2. CV and SVC | 0:17 | 50 |
| 3. LoRAs | 0:40 | 60 |
| 4. Probes | 1:05 | 65 |
| 5. Gaussian head | 1:35 | 60 |
| 6. Failure | 2:05 | 45 |
| 7. Flat tests | 2:25 | 40 |
| 8. Next | 2:40 | 55 with the TODOs |

If a run goes over three minutes, cut the 27B line in section 7 first, then the two loader traps in section 3.
