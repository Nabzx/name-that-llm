# Issue #38: What separates glm-5.3 from glm-5.3-flash

## Question

Is there anything in the text that tells glm-5.3 from glm-5.3-flash, and can we use it?

## What we looked at

- The 150 host train passages of each model. Both answer the same 150 prompts, one passage each.
- Per passage: words, sentences, words per sentence, and per 1,000 characters the rates of em dashes, curly quotes, semicolons, colons and commas. Mann-Whitney tests between the two sides.
- Words each side favours: log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008), prior from all 2,699 train passages, strength 1,000 tokens.
- First and last words of each passage.
- Out-of-fold predictions of our best SVC (`svc-pool-all-w025`, CV 0.484), on the frozen folds.

## Findings

**Style is almost identical.** Both write one paragraph, no markdown, straight quotes only, and an em dash in about 80% of passages.

| Per passage | glm-5.3 | glm-5.3-flash | p |
|---|---|---|---|
| Words, mean (median) | 110 (96) | 119 (112) | 0.20 |
| Words per sentence | 23.3 | 23.2 | 0.89 |
| Em dashes per 1k chars | 2.36 | 2.16 | 0.17 |
| Curly quotes per 1k chars | 0 | 0 | |
| Semicolons per 1k chars | 0.59 | 0.83 | 0.03 |
| Passages with a semicolon | 34% | 48% | 0.01 |
| Colons per 1k chars | 1.17 | 1.14 | 0.98 |
| Commas per 1k chars | 14.0 | 13.5 | 0.26 |

The semicolon is the only difference that clears p < 0.05, and with a dozen tests one such hit is expected by chance. On the same prompt, flash is longer only 54% of the time.

**No word marks either side.** The strongest log-odds scores are about 2.5, which is noise level across thousands of words.

| glm-5.3 favours | z | glm-5.3-flash favours | z |
|---|---|---|---|
| system | 2.4 | rather | 2.3 |
| so | 2.3 | we | 2.2 |
| stand | 2.2 | door | 2.2 |
| note | 2.1 | most | 2.2 |
| three | 2.0 | real | 2.1 |

Flash leans slightly towards "we", "rather", "whether" and semicolons, a little more essayistic. Too weak to use.

**Openings and closings match.** Both open with "The", "In", "Imagine", "Beneath", "Deep", "When". Every passage of both ends with a full stop. No title lines.

**The SVC barely splits them, and loses more to Claude than to each other.**

| True \ predicted | glm-5.3 | glm-5.3-flash | any Claude | other |
|---|---|---|---|---|
| glm-5.3 | 28 | 36 | 66 | 20 |
| glm-5.3-flash | 20 | 46 | 52 | 32 |

Recall is 0.19 for glm-5.3 and 0.31 for flash. Restricted to the two GLM columns, the pair is split at 0.54 balanced accuracy, close to a coin flip. Claude passages also flow the other way: 44 claude-fable-5 passages are read as GLM. A per-class offset on glm-5.3's score, tuned on four folds and applied to the fifth, lowers CV by 0.002.

## Recommendation

A clear no for the pair. Nothing in the surface text separates glm-5.3 from flash, and no offset or feature we can see would help. Treat the two as near-interchangeable and spend the effort on GLM against Claude, where 118 of the 300 GLM passages go. That is a larger and more tractable loss than the pair itself.
