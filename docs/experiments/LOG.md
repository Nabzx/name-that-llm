# Experiment log

One row per run. CV is balanced accuracy on the frozen `GroupKFold` folds over `prompt_id`. Δ is against the current baseline.

| # | Date | Issue | Change | CV | Δ | Public LB | Notes and next step |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-24 | #25 | Four-view n-gram LinearSVC (C 0.5, balanced class weights), host train only. Code `nthl-src` @ 343c099, notebook `nthl-svc-train` v1, CPU, 114 s | 0.3485 | baseline | not submitted | Folds range 0.31 to 0.39. The lab is right 0.57 of the time (balanced), and half the errors stay inside the right lab. Hardest: claude-fable-5 0.05, glm-5.3-flash 0.11. Worst swaps: gemini-3.7-flash and gemini-3.8-flash (86 passages between them), glm-5.3 read as claude-opus-5 (27). In line with public train-only SVCs (0.343 to 0.370, discussion 750390). Next: add the public pool (#26). |
