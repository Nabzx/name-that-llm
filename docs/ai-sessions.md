# AI sessions

Every AI session used for this entry, as required by the competition rules. Each one is exported before submission.

## How to export a session

1. Try the app first: the session menu's Export writes a zip to Downloads. For session 1 it failed twice on 2026-10-24.
2. Otherwise run our script with the transcript id:

   ```bash
   .venv/bin/python scripts/export_session.py <transcript-id> --name <nn-short-name>
   ```

   It finds `~/.claude/projects/*/<transcript-id>.jsonl` and writes `materials/ai-sessions/<name>.zip` (gitignored). The zip holds the raw transcript untouched, the subagent transcripts if there are any, and a readable `transcript.md` where long tool results are cut. The raw file is the complete log.
3. The transcript id is the file name under `~/.claude/projects/`. The app id (`local_...`) is not the same thing.

Before handing in, check each export for anything personal. Session 1 holds Nabil's email and memory notes on unrelated projects. Nothing secret: token strings in it are masked or placeholders. Redacting would make the log incomplete, so decide with Nabil before cutting anything.

Every new session or subagent run gets a row here the moment it starts.

## Register

| # | Tool | Session | Started | Purpose | Exported |
|---|---|---|---|---|---|
| 1 | Claude Code desktop, Claude Opus 5.5 | "AI Soc Kaggle Tournament", app id `local_1aa315e8-bae2-43a1-9ca3-370ff2a0a459`, transcript `5d7d3493-861c-4044-bd86-4e023fd27b66` | 2026-10-24 02:03 BST | Track research, Kaggle playbook, task analysis, repo set-up, direction research, the Phase 0 to 5 tickets and the work on them | Trial export 2026-10-24, final export at the end |
