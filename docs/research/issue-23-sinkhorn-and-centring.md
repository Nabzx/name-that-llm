# Issue #23: Sinkhorn balancing and prompt centring, when they help

## Question

Every strong public notebook runs two steps at test time: Sinkhorn balancing and prompt centring. What does each do and what does it assume about the hidden test? What did others measure? The competition page says the test does not hold the same number of passages per model, and the public and private LB hold different model mixes. When could balancing hurt?

## What we looked at

- Nishant Dahal's two-stage LoRA notebooks, parts one and two: Sinkhorn after the blend, and the centred twin.
- prvsiyan's *The Prompt Is Not the Author*: the Gaussian group head, with test groups from TF-IDF plus GLM embeddings.
- Lucas Boesen's *LoRA x Adaptive Centering*: half-strength balancing, and a centring simulation for each version.
- GoPain's comment in discussion 750390: single-member LB submissions with and without balancing.
- Our #4: how well TF-IDF clustering finds passages that share a prompt.
- A simulation on the out-of-fold predictions of our best SVC (log row 6, CV 0.4782), in `scripts/sinkhorn_sim.py`. It reads saved predictions only. Nothing is fitted.

## Findings

### Balancing

**What it does.** After the blend each passage has 18 probabilities. Add them up per model over the whole test set. If one model collects far more than 1/18 of the total, the blend leans towards it. Sinkhorn finds one scale per model that brings every total to 1/18, renormalising each passage as it goes. It uses no labels.

**What it assumes.** That the test holds about the same number of passages per model. Then any lean it sees is the blend's fault and worth removing. If the test is not balanced, it takes the real mix for a lean as well and pushes predictions towards the rarer models.

Balanced accuracy already weighs every model equally, so an unequal test does not call for a correction by itself. A blend with no lean needs no balancing whatever the mix. Balancing pays by removing the lean, and it measures the lean by assuming the test is balanced.

**What public notebooks measured.**

- **Nishant**, part one: about +0.005 locally. He warns it would hurt on a test far from balanced. Part two keeps it, at 300 steps.
- **Lucas**: half strength, with the target halfway between the current totals and 1/18. Never negative in his CV, +0.002 to +0.007, also with unequal class counts. When a genre shift collapsed the predictions onto one class, balancing lifted balanced accuracy from 0.12 to 0.19.
- **GoPain**: the same pool-trained LoRA went from 0.538 to 0.549 on the public LB with balancing, while its CV moved by less than 0.001. Pool-trained members lean on the test in a way the OOF does not show. That +0.011 was measured on the real, unequal public LB.
- **prvsiyan**: one full balancing after mixing two branches. No number on its own.

**Our simulation.** Each model keeps a random share of its OOF rows, between a floor and 100%, so a floor of 30% lets one model have up to 3.3 times the rows of another. Each row of the table is 200 draws. "Lean" multiplies each model's column by a random factor (standard deviation 0.3 in log terms) to mimic a pool-trained member on the test. That lean costs 0.013 on the balanced set, and full balancing then gains +0.017, the same order as GoPain's +0.011. "Full" is Sinkhorn to 1/18, "half" is Lucas's half strength. Each cell is the mean change in balanced accuracy, with the share of draws where it hurt in brackets.

| Rows kept per model | No lean, full | No lean, half | Lean, full | Lean, half |
|---|---|---|---|---|
| 100% (balanced) | +0.004 | +0.004 | +0.017 | +0.013 |
| 70 to 100% | +0.004 (5%) | +0.006 (0%) | +0.015 (1%) | +0.012 (1%) |
| 50 to 100% | +0.002 (25%) | +0.004 (7%) | +0.014 (3%) | +0.012 (3%) |
| 30 to 100% | +0.000 (50%) | +0.003 (25%) | +0.013 (9%) | +0.012 (2%) |
| 10 to 100% | -0.005 (71%) | +0.001 (47%) | +0.007 (29%) | +0.010 (9%) |

- **With no lean, full balancing stops helping at about a threefold skew** and hurts beyond it. Half balancing stays positive on average all the way down.
- **With a lean, balancing helps even on a badly skewed test.** Full beats half up to about a threefold skew, half wins beyond it.
- **The public and private split does not change this.** Balancing runs once over the whole test, so what counts is the mix of the whole test, not of either slice. We balanced the whole OOF set once, then scored a "public" part holding 20 to 60% of each model's rows and the rest as "private". The gain reached both parts: +0.003 and +0.004 with no lean, +0.016 and +0.017 with one.
- **Our SVC already leans on its own OOF.** It predicts gemini-3.8-flash 270 times for 150 true passages and gemini-3.7-flash 78 times. Balancing lifts it from 0.4782 to 0.4819, in line with Nishant's and Lucas's local gains.

### Centring

**What it does.** Every passage answers a prompt, and the prompt marks the text more loudly than the author does. If several test passages answer the same prompt, subtract their average features and what is left is mostly how each one was written. The test has no prompt ids, so the groups come from clustering the texts by content. A second, centred head, trained on prompt-centred train features, reads what is left.

**What it assumes.** That several test passages share a prompt and that each cluster holds one prompt. Also that a group's passages come from different models, which holds in train, where no prompt has the same model twice. It does not care how many passages each model has, so the public and private mixes do not matter to it. Group size does. The test holds about 1 to 5 passages per prompt, and centring a pair subtracts half of each passage's own author signal along with the prompt.

**What public notebooks measured.** Grouped CV has no test clusters, so these numbers come from simulations that draw a few passages per prompt from held-out train prompts, or from the LB.

- **Add, don't replace.** Lucas's first version centred passages that had no group and failed. Replacing the plain head with the centred one on groups of 4 to 18 gained nothing at 1 to 2 passages per prompt and +0.019 at 1 to 5. Keeping the plain head and adding the centred head as extra evidence from groups of 2 gained +0.024 and +0.042 on his frozen-model stack. His simulated gains matched his LB gains to within 0.002 to 0.004 per version.
- **Small groups need less subtraction.** prvsiyan found that on groups of two a centred head alone scores 0.05 to 0.10 below the plain head, because the pair's mean is half of each passage. His Gaussian group head subtracts a shrunk estimate of the prompt instead of the full mean, and decodes small groups jointly. It gained +0.010 on groups of 2 to 3 and +0.04 on whole prompts, and beat the centred head in all 68 cells he tried. On the LB it moved one notebook from 0.518 to 0.533, and another from 0.599 to 0.604 when added to every two-stage LoRA (with one grouping change).
- **Strong members gain less.** Nishant's two-stage LoRAs were trained with a centred head on same-prompt triplets. Their twin adds about +0.005 at 1 to 2 passages per prompt, and more with larger groups. The members already learn to look past the prompt.
- **Grouping is the weak link.** Content TF-IDF clustering (stop words out, average linkage, cut 0.85) finds about half the same-prompt pairs with almost no wrong merges: precision 0.97 to 1.0, recall 0.52 to 0.65 (#4). Lucas reports GLM-4-9B embeddings on top lift recall from 0.78 to 0.89 at precision 0.97. Clusters larger than 18 mix prompts and are left alone.
- **Hard assignment loses on the LB.** Giving each model at most one passage per cluster wins in simulation and loses on the LB (Nishant part two, #4). Hungarian matching inside clusters did not help either (part one). The test has too few passages per prompt.

## Recommendation

### Balancing: keep it, at half strength

Balance once, after the final blend, over the whole test set, with the target halfway between the blend's totals and 1/18 (Lucas's alpha 0.5, 300 steps). Apply it to the blended probabilities, not to each member. Skip it when the test has fewer than 50 rows, as on the 3-row public file.

Our members learn mostly from a public pool written with other settings, 2.7 times as deep for some models as for others. They will lean on the test, and then balancing is worth about +0.01 even on a skewed test. Full strength gains at most 0.004 more on a near-balanced test but can lose on a skewed one, and we do not know how skewed the test is. Half strength stayed positive on average in every case we tried. The LB cannot settle full against half: the gap is a few thousandths and the public LB's standard error is about 0.014.

Do not judge balancing by CV. CV shows +0.004, while the test gain comes from a lean the OOF cannot show.

### Centring: keep the centred twin as published, add the Gaussian head if time allows

Use centring only in the form the published checkpoints were trained for: each LoRA's centred head, added to its plain head and never replacing it. Cluster the test by content TF-IDF at 0.85, use clusters of 2 to 18, leave singletons and larger clusters alone, and weight each twin at 0.1 as in Nishant's part two. That comes with the spine (#27, #28) for every checkpoint that ships a centred head, and needs no tuning. No hard per-prompt assignment. Leave the SVC plain.

The Gaussian group head is the next step if time allows. It beat the centred twin everywhere prvsiyan measured and added about +0.005 on the LB. It needs one pass over the 2,699 train passages per LoRA to fit its covariances, on Kaggle, so weigh it against the deadline in #29.

## Sources

- Competition page: per-model test counts are unequal and the model mix differs between the public and private LB. https://www.kaggle.com/competitions/name-that-llm
- Nishant Dahal, *Two-stage LoRA on the public pool*. https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool
- Nishant Dahal, *Two-stage LoRA, part two*. https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two
- prvsiyan, *The Prompt Is Not the Author*. https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author
- Lucas Boesen, *LoRA x Adaptive Centering*. https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering
- GoPain, comment of 23 October in discussion 750390. https://www.kaggle.com/competitions/name-that-llm/discussion/750390
- Our #4 (grouping test passages by prompt) and log row 6 (#26, the OOF used here).
- `scripts/sinkhorn_sim.py`, the simulation behind the table (seed 23, about 3 minutes on a laptop).
