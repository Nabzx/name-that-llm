# Video script: three minutes

For the hand-in video (#50). One speaker, one screen, recorded solo. About 440 spoken words, which is close to three minutes at a calm pace. Numbers come from `docs/experiments/LOG.md` and `docs/research/`. TODO marks a number we do not have yet.

## Before you record

Open these in browser tabs, left to right, so each section is one tab switch:

1. The GitHub repo, `Nabzx/name-that-llm`, on the README.
2. `docs/research/issue-38-glm-pair.md` on GitHub, scrolled to the style table.
3. `docs/experiments/LOG.md` on GitHub (rendered table).
4. The Kaggle notebook `nthl-spine-qwen3-14b`, scrolled to the loader cell with `EOS` and the label reorder.
5. `docs/report/pipeline.svg`, opened full screen.
6. The competition leaderboard, filtered to our team, plus our submissions page.

Record each section as its own clip with QuickTime or OBS, then join them. A fumbled line costs one clip, not the whole take. Read the TODO lines last, once probe 6 and the final score are in.

Say "glm-5.3" as "G L M five point three".

## Script

### 1. The task (0:00 to 0:20), about 50 words

> Hi, I'm Nabil, and this is my solo entry to Name That LLM. Each passage was written by one of eighteen models, and the score is balanced accuracy. Labs are fairly easy to tell apart. Siblings are not. The two hardest pairs are glm-5.3 against glm-5.3-flash, and Gemini 3.7-flash against 3.8-flash.

**On screen:** tab 1, the repo README, then tab 2, the GLM style table. Point at the semicolon row: the only difference between the two GLMs.

### 2. CV first (0:20 to 0:35), about 40 words

> Before any model, I froze the folds. All eighteen models answer the same prompts, so a random split leaks the prompt. I used GroupKFold on the prompt, and every experiment since uses the same folds and gets a row in this log.

**On screen:** tab 3, `LOG.md`, top of the table. Hover over the CV column header.

### 3. The pool and our SVC (0:35 to 0:55), about 50 words

> The first model was an n-gram linear SVM on the host data alone: 0.348. Then I built a pool of eighty-four thousand public passages, filtered and de-duplicated. More rows beat cleaner labels, and counting pool rows at a quarter weight took cross-validation to 0.484.

**On screen:** tab 3, rows 1 to 11. Highlight 0.3485 in row 1 and 0.4838 in row 11.

### 4. Published LoRAs and our own loader (0:55 to 1:25), about 75 words

> The strongest public members are LoRA fine-tunes of Qwen. The rules allow published weights if you show what you built, so I wrote my own loader for two checkpoint formats. Two traps nearly caught me. The end token: training used 151643, but the tokenizer says 151645, and the wrong one quietly shifts every feature. And label order: each checkpoint lists the models its own way, so the loader reorders the columns and checks it.

**On screen:** tab 4, the Kaggle notebook. Scroll slowly from the `EOS` constant to the label reorder and its assert. Then tab 3, row 7.

### 5. Five probes (1:25 to 2:05), about 95 words

> Then leaderboard probes, one idea each. Probe one, the published spine with my SVC: 0.626, just above the original's 0.620. Probe two, balancing at half strength: 0.629. Full balancing over-corrects when the test mix is uneven. Probe three added prvsiyan's two Qwen3-8B seeds at a fixed ten per cent: 0.631. Probe four added another in-style LoRA and fell to 0.627. It was trained on data much like ours, so it repeated our mistakes. Probe five added two members that lifted public notebooks a long way, and stayed at 0.631. We already covered those rows.

**On screen:** tab 6, our submissions page, scrolling from probe 1 to probe 5. Cut to tab 3, rows 12 to 17, if the submissions page is hard to read.

### 6. What we added (2:05 to 2:30), about 60 words

> Here is the whole pipeline. Members blend in log space, then take fixed shares, then a balancing step. What I added: half-strength Sinkhorn balancing; peer correction, which removes the lean that passages on the same topic share; small likelihood features from base models; and my own two-stage LoRA on Qwen3-8B, trained overnight on Kaggle.
>
> TODO probe 6: "Probe six swapped Sinkhorn for peer correction and scored ___. That tells us ___."

**On screen:** tab 5, `pipeline.svg`. Point at each box in turn, left to right, as you name it. Blue boxes are ours.

### 7. One honest failure (2:30 to 2:45), about 45 words

> One failure. In every pool run Gemini 3.7-flash recall collapsed, so I guessed the pool's 3.7 rows were mislabelled and dropped them. Cross-validation fell by 0.03, and both flash models went to 3.6-flash instead. Noisy labels for a class beat no labels at all.

**On screen:** tab 3, row 8. Highlight 0.4482 and the -0.030.

### 8. Next, and the result (2:45 to 3:00), about 35 words

> Next I would add a model from outside the Qwen family, since the published ones agree on most picks, and test each step on folds that look like the test.
>
> TODO final: "The final submission scored ___ public. Thanks for watching."

**On screen:** tab 6, the leaderboard with our row highlighted.

## Timing check

| Section | Start | Words |
|---|---|---|
| 1. The task | 0:00 | 50 |
| 2. CV first | 0:20 | 40 |
| 3. Pool and SVC | 0:35 | 45 |
| 4. Loader | 0:55 | 75 |
| 5. Probes | 1:25 | 95 |
| 6. What we added | 2:05 | 60 plus probe 6 |
| 7. Failure | 2:30 | 45 |
| 8. Next | 2:45 | 30 plus final |

If a run goes over three minutes, cut probe four's reason in section 5 first, then the end-token detail in section 4.
