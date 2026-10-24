# CLAUDE.md

Conventions for Claude Code when working in this repository.

## Quick orientation

Read `AGENTS.md` for the full workflow: competition invariants, phases, the wayfinder protocol, labels, research and ADR conventions. This file covers the Claude Code-specific angle.

## House style

- British English throughout (organise, colour, behaviour, licence as a noun).
- No em dashes. Use a comma, a colon, or a full stop instead.
- Short, direct sentences. No padding, no hedging.
- No AI-generated filler: words like "comprehensive", "robust", "seamless", or "leverage" only earn their place if they're doing real work.
- Don't add abstractions, error handling, or options for scenarios that can't happen.

## Commits

- Conventional Commits: `type(scope): summary`, e.g. `feat(cv): add frozen group folds (#3)`.
- One logical change per commit. Several small commits per issue, not one large one.
- Reference the issue with `refs #n` in the body, not `closes #n`. The maintainer closes issues after review.
- No co-author trailers on commits in this repository.

## Agents and sessions

- Every session and subagent you start is added to `docs/ai-sessions.md`. Logs are a submission requirement.
- Parallel experiment agents each work in their own worktree on one `wayfinder:task` ticket, read the frozen folds, and never edit `src/cv` or the metric.
- No model is fitted locally. Experiments run as Kaggle notebooks pushed with the Kaggle CLI.

## Skills to reach for

- `simplify` and `code-review` before calling a change finished.
- `commit-commands` for the commit, push, and PR flow.
- `grilling` for `wayfinder:grilling` tickets.
- `claude-md-management` to keep this file and `AGENTS.md` current.

## Documents

Reference materials (briefs, research exports) go in `/materials/`, which is gitignored. Never commit raw documents to the repository.
