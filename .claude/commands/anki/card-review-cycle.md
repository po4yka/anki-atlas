---
description: "Autonomous card review cycle: scan, improve, delete, generate cards using autoresearch loop pattern"
argument-hint: "[Deck: <name>] [Topic: <path>] [Loop-Kind: audit|generation|both] [Iterations: N] [Scanners: fsrs,duplicates,llm-review] [Apply: auto|dry-run]"
---

EXECUTE IMMEDIATELY -- do not deliberate, do not ask clarifying questions before reading the protocol.

## Argument Parsing (do this FIRST, before reading any files)

Extract these from $ARGUMENTS -- the user may provide extensive context alongside config. Ignore prose and extract ONLY structured fields:

- `Deck:` -- Anki deck name filter (optional)
- `Topic:` -- Taxonomy path filter (optional)
- `Loop-Kind:` -- `audit` (default), `generation`, or `both`
- `Iterations:` or `--iterations` -- integer N for bounded mode (CRITICAL: if set, run exactly N iterations then stop)
- `Scanners:` -- comma-separated extras: `fsrs`, `duplicates`, `llm-review` (audit scanner always runs)
- `Apply:` -- `auto` (default) or `dry-run`

If `Iterations: N` or `--iterations N` is found, set `max_iterations = N`. Track `current_iteration` starting at 0. After iteration N, print final summary and STOP.

## Execution

1. Read the skill protocol: `.claude/skills/card-review-cycle/SKILL.md`
2. If Deck, Loop-Kind, and Iterations are all extracted -- proceed directly to Phase 0
3. If any critical field is missing -- use `AskUserQuestion` with batched questions as defined in SKILL.md "Interactive Setup" section
4. Execute the autonomous loop: Scan -> Next -> Fix -> Resolve -> Verify -> Decide -> Log -> Repeat
5. If bounded: after each iteration, check `current_iteration < max_iterations`. If not, STOP and print summary.

IMPORTANT: Start executing immediately. Stream all output live -- never run in background. Never stop early unless queue empty or max_iterations reached.
