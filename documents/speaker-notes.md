# Speaker notes: Name That LLM

Read these while showing `documents/presentation.pdf`, one block per slide. About 400 words, so three minutes at a calm pace. Say "glm-5.3" as "G L M five point three".

## 1. Cover (0:00)

> Hi, I'm Nabil. This is my solo entry to Name That LLM.

## 2. The task (0:05)

> Each passage was written by one of eighteen models from six labs. We train on 2,699 passages and are scored on about four thousand, on prompts we never see, with balanced accuracy. Labs are easy to tell apart. Siblings are not: the two GLM models, and Gemini 3.7 against 3.8.

## 3. Starting point (0:22)

> First I froze the folds, grouped by prompt, so validation looks like the hidden test. A plain n-gram SVC scored 0.348. The best public notebook scored 0.620, so I wrote my own loader for its published weights and built on top.

## 4. Approach (0:40)

> Each member gives log-probabilities. They are blended, a few extra members join at fixed ten per cent shares, and half-strength balancing evens out the classes. The filled boxes are ours: the SVC, a Gaussian group head and my own Qwen3-8B LoRA, trained overnight on Kaggle.

## 5. Hypotheses (1:00)

> Four ideas. Style lives in small habits, so n-grams should work. Data beats model size: public passages took the SVC to 0.484. The prompt hides the author: take out what passages on one prompt share, and style is left. And the test mix is uneven, so only balance halfway.

## 6. Ablation, leaderboard (1:22)

> One change per probe. The spine with my SVC: 0.626. Half-strength balancing: 0.629. Two public seeds: 0.631. Then four probes tied at 0.631. Probe nine, the Gaussian head at full weight plus ModernBERT, reached 0.637.

## 7. Ablation, cross-validation (1:45)

> These scores come from folds each model never saw. My own LoRA reached 0.519, level with a 27-billion model. The Gaussian group head added 0.058, our biggest single gain. Several other ideas stayed within noise.

## 8. What didn't work (2:05)

> Peer correction removes a topic's shared lean and drops class quotas. It helped its author's weaker blend. On ours it fell to 0.622: without quotas, a strong blend drifts towards the models it already favours.

## 9. What the leaderboard told us (2:22)

> The ties showed that small gains round away on the public slice, so I steered by cross-validation. The head that won offline moved the board once it ran at full weight.

## 10. What I'd try next (2:38)

> Next: retrain on the bigger open pool, give every member five folds so the blend is fitted on honest scores, decode all members jointly, and target the sibling pairs. Every AI session is in the repo. Thanks for watching.
