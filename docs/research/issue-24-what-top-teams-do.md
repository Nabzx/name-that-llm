# Issue #24: What separates 0.64 from 0.70

## Question

The best public notebook scores 0.639 on the public LB. The top team scores 0.703. What do the top teams do that the public notebooks don't?

## What we looked at

- The public LB, top 200, on 24 Oct at about 16:30 UK.
- About 60 discussion topics. We read in full the 35 that touch data, models, validation or rules, opening posts included.
- The 60 best public notebooks by score, and the text of 17 of them: the 12 from #2 plus five newer ones.
- Our notes for #18 and #19, and our experiment log.

Nothing was run. No top team has published its method, so every clue is indirect.

## Clues

### The leaderboard

- The top is thin: 0.703, then 0.688, then three teams at 0.673 to 0.674. 19 teams are at 0.653 or above, 44 above 0.639. 77 teams sit at exactly 0.639, the public notebook. Rank 200 is 0.636. (LB)
- None of the top 20 names on the LB matches an author of the 60 best public notebooks. The public authors sit lower. Nishant Dahal, whose two-stage LoRA is the core of every notebook above 0.62, is at 0.642. Kirill Tymchishin, author of the 0.639, is at 0.639. (LB, kernel list)
- Vishak K Bhat's 0.673 is dated 22 Oct, two days before the public 0.639 existed. (LB, 0.637 notebook)

So the top teams are not ahead because they found the public recipe first. They have something its authors don't.

### Data

- Pool size is the biggest lever that repeats across 134 public versions. Going from 35k to 84k passages gave +0.028 on the LB. (752357)
- On the 35k pool, four Qwen backbones from 7B to 14B all tied at 0.569. The same Qwen3-14B on the 84k pool reached 0.611. Stage one alone, before it sees any host passage, went from 0.493 to 0.572 on the host data. (Nishant part two)
- External data is worth about +0.11 to +0.12 on the LB, by one entrant's count. His best without it is 0.505. (749447)
- Coverage and style beat volume. Covering 13 of the 18 labels cost the other five 6 points of recall. Covering all 18 took a frozen-feature model from 0.447 to 0.508. (751871) External answers in a different style, even flattened to one paragraph, made a TF-IDF model worse, 0.326 to 0.279. (749447)
- Private data is already in use. The 0.639 fits its extra readouts partly on 4,404 privately generated passages: 0.639 against 0.637 for the same heads on train only. (0.637 notebook)
- Top teams are asking about it. Adisak Sukul (#7, 0.666) asked the hosts whether privately generated data under $5,000 counts as reasonable. (751011) Meni David Gottesman (#15, 0.659) asks how others validate members "trained on large external/generated pools". (752199) The hosts have not answered. (752084)
- Bigger pools than ours exist. nuridudak trained ModernBERT-large on 122k prompt-disjoint external passages. (0.626 notebook) Our open-licence pool is 84,470. (#19)
- The author of the 752357 lineage study reaches the same view: the gap past 0.639 is "mostly data, not architecture". (752357)

### Models

- A larger backbone on the same data did not help: 8B and 14B tied, and Gemma-4-12B was no better. (Nishant part two) New large LLM members added +0.005 to +0.006 at most, and none scored above 0.61 alone. (752357)
- The one full-text 27B in public (Qwen3.8-27B in 4 bit, all 64 layers, no truncation) gets only 10% weight, and only on the half of rows where the blend is least sure. (0.636 notebook)
- Top teams are still trying. Meni (#15) asks about running a 30B mixture-of-experts model in 4 bit on two T4s. (752198)
- Extra heads on existing features are cheap and repeat. An 8-layer and a 1-layer readout of the same LoRA disagree on 29% of passages and blend for about +0.02 CV. (751871) Gaussian group heads gave about +0.005 in each of five versions, and no public notebook has put them on the two 14B LoRAs. (752357)

### Test-time steps

- Hard Hungarian assignment loses on the LB, because the hidden test seems to have only one to three answers per prompt. (750213, 750706, 751802, 751871) Sinkhorn balancing helps a little: +0.011 for one LoRA. (750390)
- No public notebook pseudo-labels the test. The only label-free test steps are clustering and standardising features with test statistics. (prvsiyan) Entrants are asking whether pseudo-labelling inside the notebook is allowed (752084), and Meni asks about "steps that use same-prompt groups at test time". (752199)
- Pool-trained members read 0.03 to 0.04 higher in CV than on the LB, and near-copies of train passages don't explain it. (750390, 751941) That is a gap between pool text and host text, which is what adapting on the test would target.

### Labels on the hard pairs

- Requests to gemini-3.7-flash have been routed to 3.8 since 8 Oct, and 7 of 12 pool sources have no 3.7 rows. (751370, 751871) In our own SVC, 3.7 recall falls from 0.41 to 0.19 once the pool is in. (LOG, row 6)
- Every public member gets stuck on glm-5.3 against glm-5.3-flash and gemini-3.7-flash against 3.8. (Nishant part two) A Claude-only head trained with official-API Claude passages took one LoRA from 0.456 to 0.493 on fold 0. (750224)

### Noise

- The public LB's standard error is about 0.014. (750390) Picking the best of many submissions lifts the top scores above their true level. One entrant says part of their 0.583 before external data may have been LB probing. (749447)

## Ranked guesses

Most likely first. GPU hours are T4-pair hours, estimated from #5 and the run times in public notebooks. We have about 26 hours and 30 GPU hours left.

| # | Guess | Worth | Cost to us |
|---|---|---|---|
| 1 | **More in-style data, privately generated** through official APIs, covering every label, then members retrained on it | Most of the gap, about 0.03 to 0.05 | Out of reach. Our invariants allow only free public data, so no paid APIs, and 26 hours is too short anyway. The nearest step, one Qwen3-8B two-stage LoRA retrained on our 84k pool, costs about 4 to 6 GPU hours and adds little over Nishant's 14B, which saw the same pool. |
| 2 | **Adapting to the test**: pseudo-labels or self-training on the hidden test passages | 0.01 to 0.03, if it works | 2 to 3 GPU hours to cache LoRA features and check the heads on folds cut to one to three answers per prompt. The heads fit on CPU, and the final version runs inside the scoring run. |
| 3 | **Clean labels on the hard pairs**, such as real gemini-3.7-flash rows and official-API Claude rows | 0.005 to 0.015 | Under 1 GPU hour for heads on cached features. The SVC side is CPU only (#64). |
| 4 | **Some of the gap is noise** from picking the best of many public LB scores | 0.01 to 0.02 of the visible 0.064 | None. It means the real gap is probably 0.04 to 0.05. |
| 5 | **More heads on the strongest features**: Gaussian group heads and multi-layer readouts on the 14B LoRAs | About 0.005 each | 1 to 2 GPU hours to extract the 14B features once. Heads fit on CPU. |
| 6 | **Bigger or full-text backbones** (27B, 30B mixture of experts) | 0.005 or less on public data | 10 or more GPU hours per fit, and bitsandbytes is missing from Kaggle's image (#17). Out of reach. |

## Recommendation

Don't chase the data gap. We cannot match private paid data under our rules, and a 14B retrain does not fit. Aim for 0.64 to 0.65.

1. Build the published spine first (#27, #28). It is mostly inference and should land at 0.636 to 0.639.
2. Then try the two cheap levers no public notebook has: test-time pseudo-label heads (guess 2) and Gaussian heads on the 14B LoRA features (guess 5). About 2 to 3 GPU hours each. Pseudo-labels are code in the notebook, which our invariants allow.
3. Fix the gemini-3.7-flash rows (#64) on CPU, and carry the fix into any head we fit.
4. Judge every test-time step on folds cut to one to three answers per prompt, never on train's full 18 (750213).
5. Skip the 27B and 30B backbones.
6. Keep about 10 GPU hours for scoring runs. A full public stack takes about 2 to 4 hours per run, and a slow node can double that (751268).

## Sources

Leaderboard and notebook list:

- Public leaderboard, read with `kaggle competitions leaderboard name-that-llm --show`, 24 Oct 2026
- Public notebook list, `kaggle kernels list --competition name-that-llm --sort-by scoreDescending`, 24 Oct 2026

Discussion, all at `https://www.kaggle.com/competitions/name-that-llm/discussion/<id>`:

- 749447: Will the winner just be whoever can get the most extra training data?
- 750213: Test-time prompt grouping: simulation against the LB
- 750224: Extra Claude passages on new prompts
- 750390: Grouped CV that tracks the LB (GoPain's comment on pool-trained members)
- 750706: Every prompt is answered by every model exactly once
- 751011: Section 2.6 question on training data generated from the candidate models
- 751268: Forked a 0.600+ notebook and hit the 9-hour limit?
- 751370: Gemini-3.7 Flash requests routed to Gemini-3.8 Flash
- 751802: Does the test set keep one passage per model per prompt?
- 751871: Notes from our runs: what transferred to the LB, what didn't
- 751941: Does the public pool leak train prompts into grouped CV?
- 752084: Open rules questions in one place
- 752198: Throughput of 30B MoE models in 4 bit on 2xT4?
- 752199: How well does your CV track the public LB?
- 752357: What 134 public versions show, and what is left past 0.639

Notebooks, all at `https://www.kaggle.com/code/<ref>`:

- nishantdahal/two-stage-lora-part-two ("Nishant part two")
- kirilltym/0-639-teach-more-keep-the-signature
- kirilltym/0-637-remove-the-prompt-keep-the-signature ("0.637 notebook")
- lightningv08/0-636-lora-llama-qwen3-8-27b ("0.636 notebook")
- nuridudak/0-626-when-llm-decoders-meet-bert-encodersv2 ("0.626 notebook")
- prvsiyan/the-prompt-is-not-the-author ("prvsiyan")

Our own:

- `docs/research/issue-18-published-checkpoints.md`
- `docs/research/issue-19-public-pools.md`
- `docs/experiments/LOG.md`
