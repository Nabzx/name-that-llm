# Issue #40: What separates the three gpt-5.6 models

## Question

Is there anything in the text that tells gpt-5.6-luna, gpt-5.6-sol and gpt-5.6-terra apart, and can we use it?

## What we looked at

- The 150 host train passages of each model. All three answer the same 150 prompts, one passage each.
- The same measures as #38: length, sentence length, punctuation rates per 1,000 characters, log-odds of each model against the other two with an informative Dirichlet prior (strength 1,000 tokens, prior from all train passages), openings and closings.
- Out-of-fold predictions of our best SVC (`svc-pool-all-w025`, CV 0.484) on the frozen folds.

## Findings

**Length is the same. Punctuation habits differ a little.** All three write one paragraph with no markdown.

| Per passage | luna | sol | terra |
|---|---|---|---|
| Words, mean (median) | 115 (105) | 119 (112) | 113 (101) |
| Words per sentence | 23.7 | 24.9 | **26.3** |
| Em dashes per 1k chars | **0.73** | 0.51 | 0.57 |
| Passages with an em dash | **37%** | 27% | 23% |
| Curly quotes per 1k chars | 1.43 | 1.77 | 1.32 |
| Semicolons per 1k chars | 0.68 | **0.99** | 0.80 |
| Passages with a semicolon | 43% | **62%** | 52% |
| Colons per 1k chars | 1.33 | 1.20 | 1.37 |
| Commas per 1k chars | 15.5 | 15.4 | 16.2 |
| Parentheses per 1k chars | **0.21** | 0.02 | 0.09 |

The differences that clear p < 0.01 in Mann-Whitney tests:

- Sol uses more semicolons than luna (p = 0.001).
- Terra writes longer sentences than luna (p = 0.003).
- Luna uses em dashes in more passages than terra (p = 0.008).
- Terra sometimes uses straight quotes, sol never (p = 0.004), but only in a few passages.

Luna also writes inline LaTeX (`\(T\)`) in 4 of 150 passages, terra in 1, sol in none. All of these are small shifts in shared habits, not markers.

**Words: weak, and mostly echoes of the punctuation.**

| luna favours | z | sol favours | z | terra favours | z |
|---|---|---|---|---|---|
| `\` (LaTeX) | 3.5 | no | 3.5 | distances | 2.5 |
| records | 2.8 | she | 2.7 | hunger | 2.4 |
| sometimes | 2.7 | ever | 2.6 | early | 2.4 |
| seems | 2.6 | again | 2.5 | tastes | 2.3 |
| `(` `)` | 2.5 | `;` | 2.5 | `,` | 1.9 |

Luna hedges ("sometimes", "seems") and explains in brackets. Sol is more narrative: "no", "ever", "again", "she", and quoted speech (`“ ”`, z 2.3). Terra's list is topic words and commas, fitting its longer sentences.

**Openings and closings match.** All three open with "The", "In", "At", "When", "Imagine". Every passage ends with a full stop, except two of sol's that end inside a closing quote.

**The SVC separates the trio better than the other hard pairs.**

| True \ predicted | luna | sol | terra | gpt-5.5 | gpt-6-astra | other |
|---|---|---|---|---|---|---|
| luna | 74 | 11 | 23 | 3 | 17 | 22 |
| sol | 26 | 51 | 32 | 8 | 8 | 25 |
| terra | 24 | 14 | 58 | 16 | 11 | 27 |

Recall is 0.49 for luna, 0.34 for sol and 0.39 for terra. Restricted to the three columns, accuracy is 0.57 against 0.33 for chance. Sol is the hardest: it scatters evenly to luna and terra. Roughly half of each model's errors go outside the trio, to gpt-5.5 (mostly terra), gpt-6-astra (mostly luna) and qwen3.8-max. A per-class offset on sol's score, tuned on four folds and applied to the fifth, lowers CV by 0.001.

## Recommendation

A small lever, cheap to test. The trio differ in rates the n-gram SVC sees only as raw counts: semicolons, em dashes, brackets, straight quotes and words per sentence. Add a dense block of about ten style rates, each per 1,000 characters plus mean words per sentence, to the stacker alongside the SVC's out-of-fold predictions, on the frozen folds, on CPU. Expect a small gain at most: the shifts are real but each one is a few points on overlapping distributions. If it adds nothing for the trio, drop it. The same block costs nothing to try on the GLM and Gemini pairs, though #38 and #39 found less there.
