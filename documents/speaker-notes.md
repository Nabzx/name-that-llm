# Speaker notes: Name That LLM

Read these while showing `documents/presentation.pdf`, one block per slide. About 480 words, so three minutes at a brisk, steady pace. Say "glm-5.3" as "G L M five point three".

## 1. Cover (0:00)

> Hi, I'm Nabil. This is my solo entry to Name That LLM.

## 2. The task (0:04)

> Each passage was written by one of eighteen models. I train on 2,699 passages and am scored on about four thousand, on prompts I never see. Labs are easy to tell apart. Siblings are not: the two GLM models, and Gemini 3.7 against 3.8.

## 3. Literature review (0:17)

> I wrote no code for the first two hours. I read twenty-two public notebooks and eighteen discussion threads, and wrote fifteen research notes, each ending in a decision. Three findings set the plan: data beats model size, the best blend's weights were public, and no public model had honest out-of-fold scores, so I had to validate on my own folds.

## 4. Starting point (0:37)

> So I froze the folds first, grouped by prompt, like the hidden test. A plain n-gram SVC scored 0.348. Then I wrote my own loader for the published 0.620 weights and built on top.

## 5. Approach (0:50)

> Each member gives log-probabilities. They are blended, a few extra members join at fixed shares, and half-strength balancing evens out the classes. The filled boxes are mine: the SVC, a Gaussian group head and my own Qwen3-8B LoRA, trained overnight on Kaggle.

## 6. Hypotheses (1:06)

> Four ideas. Style lives in small habits, so n-grams should work. Data beats size: public passages took the SVC to 0.484. The prompt hides the author: take out what passages on one prompt share, and style is left. And the test mix is uneven, so balance only halfway.

## 7. Ablation, leaderboard (1:24)

> One change per probe. The spine with my SVC: 0.626. Half-strength balancing: 0.629. Two public seeds: 0.631. Then four ties at 0.631. Probe nine, the Gaussian head at full weight plus ModernBERT, reached 0.637. My picks are probe nine, and probe seven, which carries my own model.

## 8. Ablation, cross-validation (1:45)

> These scores come from folds each model never saw. My own LoRA reached 0.519, level with a 27-billion model. The Gaussian group head added 0.058, my biggest single gain. Other ideas stayed within noise.

## 9. What didn't work (2:00)

> Peer correction removes a topic's shared lean and drops class quotas. It helped its author's weaker blend. On mine it fell to 0.622: without quotas, a strong blend drifts towards the models it already favours.

## 10. What the leaderboard told me (2:14)

> Small gains round away on the public slice, so I steered by cross-validation. And features must be honest: the head helped on my own model, but on the 14B models, which saw all of train, it dropped to 0.630.

## 11. How I used AI (2:27)

> I used Claude Code from the first hour, set up like my other repos: an AGENTS file with the rules as invariants, small tickets, a research note before each decision and a log row for every change. Frozen folds, tests first and every model on Kaggle make it reproducible. That structure let six side sessions work in parallel while I made every call.

## 12. What I'd try next (2:47)

> Next: a bigger open pool, five folds per member for honest blending, joint decoding across members, and targeted data for the sibling pairs. Thanks for watching.
