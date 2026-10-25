# AI sessions

Every AI session used for this entry, as required by the competition rules. Each one is exported before submission.

## How to export a session

1. Try the app first: the session menu's Export writes a zip to Downloads. For session 1 it failed twice on 2026-10-24.
2. Otherwise run our script with the transcript id:

   ```bash
   .venv/bin/python scripts/export_session.py <transcript-id> --name <nn-short-name>
   ```

   It finds `~/.claude/projects/*/<transcript-id>.jsonl` and writes `ai-sessions/<name>.zip` and `ai-sessions/<name>.md` in the repo, so the logs travel with the code. The zip holds the raw transcript untouched, the subagent transcripts if there are any, and a readable `transcript.md` where long tool results are cut. The raw file is the complete log.
3. The transcript id is the file name under `~/.claude/projects/`. The app id (`local_...`) is not the same thing.

Re-run the export after the last message of a session, and commit the result. An export is a snapshot: anything said after it is missing.

Before handing in, check each export for anything personal. Session 1 holds Nabil's email and memory notes on unrelated projects. Nothing secret: a scan on 2026-10-25 for Kaggle, GitHub, Hugging Face, OpenAI and AWS token patterns found none. Redacting would make the log incomplete, so decide with Nabil before cutting anything.

Every new session or subagent run gets a row here the moment it starts. Nabil used no AI chat outside Claude Code for this entry (confirmed 2026-10-25).

## Register

| # | Tool | Session | Started | Purpose | Exported |
|---|---|---|---|---|---|
| 1 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament", app id `local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459`, transcript `5d7d3493-861c-4044-bd86-4e023fd27b66` | 2026-10-24 02:03 BST | Track research, Kaggle playbook, task analysis, repo set-up, direction research, the Phase 0 to 5 tickets and the work on them | `ai-sessions/01-main-session.zip` (final export 2026-10-25 16:09 GMT with 8 subagent logs |
| 2 | Claude Code subagent of session 1, Claude Opus 5.5 | Research agent for #20 (blend validation). Transcript in session 1's `subagents` folder | 2026-10-24 16:28 BST | Read and wrote the #20 research note on its own branch | Exported with session 1 |
| 3 | Claude Code subagent of session 1, Claude Opus 5.5 | Research agent for #23 (Sinkhorn and centring). Transcript in session 1's `subagents` folder | 2026-10-24 16:28 BST | Read, simulated on our OOF predictions and wrote the #23 note | Exported with session 1 |
| 4 | Claude Code subagent of session 1, Claude Opus 5.5 | Research agent for #24 (what top teams do). Transcript in session 1's `subagents` folder | 2026-10-24 16:28 BST | Read the leaderboard and discussion and wrote the #24 note | Exported with session 1 |
| 5 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #32 and #33 (likelihood attribution and witness features). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 6 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #38 to #40 (hard pairs). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 7 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #48 (ablation table and report skeleton). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 8 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #28 (Zbigniew's checkpoint reader). Transcript in session 1's `subagents` folder | 2026-10-24 | Worked the ticket on its own branch and opened a PR | Exported with session 1 |
| 9 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament (fork)", app id `local_b253a26e-df68-47e4-8da8-023d80ce5091` | 2026-10-24 | An accidental fork of session 1. Interrupted after 4 turns, it changed nothing. Nabil deleted it, so its log cannot be exported | Not exportable (deleted) |
| 10 | Claude Code subagent of session 1, Claude Opus 5.5 | Agent for #28 (UltraBERT member). Transcript in session 1's `subagents` folder | 2026-10-24 | Wrote the UltraBERT reader on its own branch and opened a PR | Exported with session 1 |
| 11 | Claude Code desktop session started by session 1, Claude Opus 5.5 | "Wire our own LoRA into the submission", app id `local_23981830-b605-4c49-bdc3-9636adbe0c7f`, transcript `1788d878-ee76-4ed9-9017-f601dac928f4` | 2026-10-25 01:10 GMT | OURS_SHARE member and the nthl-check-ours notebook, PR #93 | `ai-sessions/11-own-lora-member.zip` |
| 12 | Claude Code desktop session started by session 1, Claude Opus 5.5 | "Fill the report draft with probes 1 to 5", app id `local_2149c9ef-d1bb-46ca-8579-8dcd5c6c2900`, transcript `95546f37-fd4a-4ca4-b719-e8966412838b` | 2026-10-25 01:10 GMT | Report and ablation table, PRs #94 and #102 | `ai-sessions/12-report-draft.zip` |
| 13 | Claude Code desktop session started by session 1, Claude Opus 5.5 | "Script and storyboard the 3-minute video", app id `local_79ec9917-58af-4c8a-b2e4-3861411ce814`, transcript `9014c337-56e2-446b-9370-78528936a8ee` | 2026-10-25 01:10 GMT | Video script and pipeline diagram, PRs #95 and #101 | `ai-sessions/13-video-script.zip` |
| 14 | Claude Code desktop session started by session 1, Claude Opus 5.5 | "Test the full-text Qwen3.8-27B classifier", app id `local_5382149d-7060-42cc-86a5-38af5011077e`, transcript `b11845df-3e07-44bc-860a-265c1192f2c8` | 2026-10-25 09:55 GMT | Our 27B loader and its held-out check, PR #100 | `ai-sessions/14-qwen27b-check.zip` |
| 15 | Claude Code desktop session started by session 1, Claude Opus 5.5 | "Add nuridudak's 122k-pool encoders as members", app id `local_fc1a59f7-32aa-4fd2-8a17-f0b9a424208c`, transcript `40d48bf4-5e80-4eef-9fc0-837396d4f3e5` | 2026-10-25 10:05 GMT | ModernBERT and desklib loader and check, PR #99 | `ai-sessions/15-encoders-122k.zip` |
| 16 | Claude Code desktop session started by session 1, Claude Opus 5.5 | "Hard-pair specialists for GLM and Gemini", app id `local_189a6d27-6360-4f5e-ae99-73795f412148`, transcript `66776953-8515-46ab-ba28-5071c1f5cd44` | 2026-10-25 10:05 GMT | Pair specialists and their CV check, PR #98 | `ai-sessions/16-hard-pairs.zip` |
