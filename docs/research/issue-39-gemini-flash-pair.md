# Issue #39: What separates gemini-3.7-flash from gemini-3.8-flash

## Question

Is there anything in the text that tells gemini-3.7-flash from gemini-3.8-flash, and can we use it? Does the report of 3.7 requests being routed to 3.8 change the picture?

## What we looked at

- The 150 host train passages of each model. Both answer the same 150 prompts, one passage each.
- The same measures as #38: length, sentence length, punctuation rates per 1,000 characters, log-odds with an informative Dirichlet prior (strength 1,000 tokens, prior from all train passages), openings and closings.
- Out-of-fold predictions of our best SVC (`svc-pool-all-w025`, CV 0.484) on the frozen folds.
- Discussion 751370 on 3.7 requests routed to 3.8, and our LOG rows 6 and 8 (#26, #64).

## Findings

**3.7 writes longer passages, and nothing else stands out.** Neither uses em dashes. Both use curly quotes in under a third of passages.

| Per passage | 3.7-flash | 3.8-flash | p |
|---|---|---|---|
| Words, mean (median) | 117 (102) | 101 (84) | 0.01 |
| Words per sentence | 23.9 | 22.7 | 0.01 |
| Em dashes per 1k chars | 0.01 | 0 | 0.32 |
| Curly quotes per 1k chars | 0.50 | 0.54 | |
| Semicolons per 1k chars | 0.29 | 0.41 | 0.24 |
| Colons per 1k chars | 0.25 | 0.27 | 0.51 |
| Commas per 1k chars | 11.7 | 12.2 | 0.30 |

On the same prompt, 3.7 is the longer one 60% of the time (Wilcoxon p = 0.009). The spread is wide, though: standard deviation is about 55 words on each side, so length alone sorts few passages.

**Words: a faint register gap, no markers.** 3.7 leans on abstract nouns and "of" phrases, 3.8 on people and action. All scores are near noise level.

| 3.7-flash favours | z | 3.8-flash favours | z |
|---|---|---|---|
| of | 2.5 | who | 2.4 |
| when | 2.1 | so | 2.3 |
| evolution | 1.9 | stars | 2.2 |
| existence | 1.8 | blind | 2.1 |
| profound | 1.8 | demands | 2.1 |

"of" runs at 3.2 per 100 words in 3.7 and 2.8 in 3.8. 3.6-flash also sits at 3.2, so this may be the older flash register.

**Openings and closings match.** Both open with "Beneath", "The", "Deep", "When", "Across". "Deep within the" opens 7 of the 3.7 passages and 3 of the 3.8 ones. All but one passage of each end with a full stop.

**The SVC folds 3.7 into 3.8.**

| True \ predicted | 3.6-flash | 3.7-flash | 3.8-flash | other |
|---|---|---|---|---|
| 3.7-flash | 18 | 31 | 88 | 13 |
| 3.8-flash | 13 | 21 | 109 | 7 |

Recall is 0.21 for 3.7 and 0.73 for 3.8. Restricted to the pair, balanced accuracy is 0.54. The 3.7 passages read as 3.8 are longer on average (121 words) than the ones read correctly (107), so the SVC is not using length.

**The routing story.** Discussion 751370 checked the host's labels and found no sign of 3.8 text under the 3.7 label. Its host-trained models err both ways, and its LoRA reads 3.8 as 3.7 more often than the reverse. Our LOG agrees: host-only, 3.7 recall was 0.41. It fell to 0.19 once the pool came in (row 6), and 7 of 12 pool sources have no 3.7 rows. Dropping the pool's 3.7 and 3.8 rows made both worse (row 8, -0.030). The pull towards 3.8 comes from the pool, not the host text.

A per-class offset on 3.7's score, tuned on four folds and applied to the fifth, picks offsets near zero and lowers CV by 0.002. The bias is not a simple prior shift.

## Recommendation

No text-level lever. Length is real but too spread to act on, and the SVC's errors do not follow it. The one route left is the pool's labels: weight or filter Gemini 3.7 and 3.8 pool rows by source, keeping sources that carry real 3.7 rows. That is #64's ground, so carry this note there rather than opening a new ticket.
