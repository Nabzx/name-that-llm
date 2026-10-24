# AGENTS.md

Structure and conventions for anyone, human or agent, working in this repository.

This repository is a solo entry to the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm), run as the Imperial Research Arm qualifier (Sat 24 Oct to Sun 25 Oct 2026, 18:00 UK).

## Competition invariants

These hold for every change. Breaking one can disqualify the entry.

- **Train on Kaggle only.** Every model fit runs in a Kaggle notebook under the account `nabilshah`. The local machine is for writing code, orchestrating runs and descriptive EDA.
- **No forking public notebooks.** Start from the official starter or from scratch. Public work may be read; cite it in `CITATIONS.md`.
- **Free and public only.** External data and pretrained models must be free and publicly available. Each one is listed in `CITATIONS.md`. No paid APIs.
- **Every AI session is logged.** Each Claude Code session, subagent run, workflow or other AI chat gets a row in `docs/ai-sessions.md` and is exported before submission. Missing logs mean disqualification.
- **Folds are frozen.** Cross-validation is `GroupKFold` on `prompt_id`, fixed in phase 0. Every experiment uses the same folds file. Changing it needs an ADR.
- **Every experiment is logged.** It saves out-of-fold and test predictions and adds a row to `docs/experiments/LOG.md`: the change, local CV against the baseline, and the next step.
- **Seeds are fixed** in every notebook and script.
- **The repository stays private** until after the deadline. No code or ideas are shared with other participants.

## Phases

Work is organised into phases. Each phase maps to one milestone (`vX.Y.Z`) and one `phase-N` label. A phase ends when its milestone closes. `ROADMAP.md` holds the live phase table.

## Tracks

If the work splits into parallel workstreams, each gets a `track-*` label (for example `track-a`, `track-b`). Not in use until that happens.

## The wayfinder protocol

Each phase opens with one `wayfinder:map` issue: the phase's single source of truth for what's known, decided, and next.

A map issue body has three sections, kept up to date as children close:

- **Fog**: open unknowns.
- **Decisions so far**: what's been settled, and why.
- **Frontier**: the next unblocked tickets.

Child tickets hang off the map issue with a `Part of: #n` line, and carry one of:

- `wayfinder:research`: investigate and report back, no implementation.
- `wayfinder:grilling`: stress-test a plan or decision before committing to it.
- `wayfinder:prototype`: throwaway build to answer a design question.
- `wayfinder:task`: a concrete, scoped piece of work. For this repository, usually one experiment.

## Labels

- Phase: `phase-0`, `phase-1`, and so on.
- Track: `track-a`, `track-b`, added once more than one workstream exists.
- Wayfinder: `wayfinder:map`, `wayfinder:research`, `wayfinder:grilling`, `wayfinder:prototype`, `wayfinder:task`.
- Severity: `p1`, `p2`, `p3`.
- Workflow: `needs-decision`, `needs-research`, `needs-repro`, `ready-for-agent`.
- Defaults: `bug`, `documentation`, `enhancement`.

## Research

Output of a `wayfinder:research` ticket lands in `docs/research/issue-<n>-<slug>.md`. Every file ends with a recommendation. A recommendation that's adopted gets promoted to a numbered ADR in `docs/adr/`.

## Architecture decisions

`docs/adr/` holds numbered ADRs, one decision per file, starting from `docs/adr/0000-template.md`. The index lives in `docs/adr/README.md` and must stay in sync with the files present.

## Layout

- `notebooks/`: one folder per Kaggle notebook, each with its `kernel-metadata.json`.
- `src/`: shared code (folds, metric, features) used by the notebooks.
- `docs/experiments/LOG.md`: the experiment log, one row per run.
- `docs/ai-sessions.md`: every AI session used, for export at the end.
- `CITATIONS.md`: every public notebook, dataset and pretrained model used.
- `data/`, `outputs/`: local only, gitignored.

## Pull requests

- Title: Conventional Commits, with scope where it helps, for example `feat(features): add char n-gram tf-idf (#12)`.
- Body: *What this does*, *Why*, a checklist, and a test plan. For an experiment, include the CV result and the log row.
- Reference the issue with `refs #n`, not `closes #n`. The maintainer closes issues on merge.
- Small commits, Conventional Commits, no co-author trailers.

## House style

- British English throughout.
- No em dashes.
- Short and direct. No filler, no hedging.

## Documents

Reference materials go in `/materials/`, which is gitignored and never committed.
