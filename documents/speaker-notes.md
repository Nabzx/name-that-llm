# Speaker notes: Name That LLM

For the 3-minute video, read alongside `documents/presentation.pdf` (or `presentation.html` in a browser, full screen). About 450 spoken words, roughly 18 seconds a slide. Say "glm-5.3" as "G L M five point three".

## 1. Cover (0:00 to 0:08)

> Hi, I'm Nabil. This is my solo entry to the Research Arm qualifier, Name That LLM.

## 2. The task (0:08 to 0:25)

> Each passage, forty to two hundred and fifty words, was written by one of eighteen models from six labs. We have 2,699 training passages and about four thousand hidden test passages on prompts we never see. The score is balanced accuracy. Labs are easy to tell apart. Siblings are not: the two GLM models, and Gemini 3.7 against 3.8 flash.

## 3. Starting point (0:25 to 0:45)

> Before any model, I froze the folds: grouped by prompt, so validation looks like the hidden test. My first model, an n-gram SVC on host data only, scored 0.348. The best public notebook scored 0.620. Rather than fork it, I wrote my own loader for its published weights and built my own members on top, all trained on Kaggle and all cited.

## 4. Approach (0:45 to 1:05)

> Here's the pipeline. Each member gives log-probabilities, which are blended as a weighted sum. A few extra members join at fixed ten per cent shares. Then a half-strength Sinkhorn step pulls the class totals halfway to even. The filled boxes are ours: the SVC, the Gaussian group head and my own Qwen3-8B LoRA, trained overnight.

## 5. Hypotheses (1:05 to 1:30)

> Four hypotheses. One: style lives in small habits like function words and punctuation, so n-grams should work. Two: data beats model size, and eighty-four thousand public passages took the SVC from 0.348 to 0.484. Three: the prompt hides the author. Passages that share a prompt also share content, so taking that out should leave style. Four: the test mix is uneven, so balance only halfway.

## 6. Ablation, leaderboard (1:30 to 1:55)

> Each probe changed one thing. The published spine with my SVC: 0.626. Half-strength balancing: 0.629. Two extra seeds: 0.631. Then a run of ties at 0.631 as I added members, including my own LoRA. Probe nine put the Gaussian head at full weight and added ModernBERT, and the score jumped to 0.637.

## 7. Ablation, cross-validation (1:55 to 2:15)

> Off the leaderboard, every number comes from folds the model never saw. The pool lifted the SVC. My own LoRA's first stage reached 0.519, level with a 27-billion model. The biggest single gain was the Gaussian group head: plus 0.058 on its own view. Other ideas, like pair specialists and a bigger pool for the SVC, stayed within noise.

## 8. What didn't work (2:15 to 2:32)

> One failure. Peer correction, a public trick, removes the lean shared within a topic and drops class quotas. It helped a weaker blend. On ours it fell to 0.622, our worst probe, because without quotas a strong blend drifts towards the models it already favours.

## 9. What the leaderboard told us (2:32 to 2:45)

> The ties taught me the public slice is small, so small gains round away. I steered by cross-validation instead, and the head that won offline was the one that finally moved the board.

## 10. What I'd try next (2:45 to 3:00)

> Next: retrain on the bigger open pool, five folds for every member so the blend is fitted on honest scores, one joint Gaussian model across all members, and targeted data for the GLM and Gemini siblings. Every AI session I used is in the repo. Thanks for watching.

## Timing

| Slide | Start | Words |
|---|---|---|
| 1 Cover | 0:00 | 20 |
| 2 Task | 0:08 | 65 |
| 3 Starting point | 0:25 | 65 |
| 4 Approach | 0:45 | 60 |
| 5 Hypotheses | 1:05 | 70 |
| 6 Leaderboard | 1:30 | 60 |
| 7 Cross-validation | 1:55 | 60 |
| 8 Failure | 2:15 | 50 |
| 9 What it told us | 2:32 | 40 |
| 10 Next | 2:45 | 50 |

If a take runs long, cut the second sentence of slide 3 first, then the 27B mention on slide 7.
