---
name: card-review-cycle
description: Autonomous card review cycle using autoresearch loop pattern. Scans Anki decks, improves/deletes/generates cards with mechanical metrics, git-as-memory, and automatic rollback. Supports bounded iterations, usage-limit detection, and checkpoint/resume.
version: 2.0.0
---

# Card Review Cycle -- Autonomous Improvement Loop

Inspired by [autoresearch](https://github.com/karpathy/autoresearch). Applies constraint-driven
autonomous iteration to Anki card quality: Scan -> Fix -> Verify -> Keep/Discard -> Repeat.

**Core idea:** You are an autonomous agent. Fix ONE card per iteration. Verify mechanically.
Keep improvements, auto-revert regressions. Never stop until interrupted or iterations exhausted.

## MANDATORY: Interactive Setup Gate

**CRITICAL -- READ THIS FIRST BEFORE ANY ACTION:**

1. Check if the user provided ALL config inline (Deck, Loop-Kind, Iterations, etc.)
2. If ANY field is missing, use `AskUserQuestion` to collect it BEFORE proceeding
3. Follow the Interactive Setup section below exactly when context is missing

## Inline Config Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `Deck:` | No | all decks | Anki deck name filter |
| `Topic:` | No | all topics | Taxonomy path filter |
| `Loop-Kind:` | No | `audit` | `audit`, `generation`, or `both` |
| `Iterations:` | No | unbounded | Integer N for bounded mode |
| `Scanners:` | No | audit only | Comma-separated: `fsrs`, `duplicates`, `llm-review` |
| `Apply:` | No | `auto` | `auto` (apply fixes) or `dry-run` (report only) |

## Interactive Setup (when invoked without full config)

If the user did not provide Deck or Iterations inline, ask via batched `AskUserQuestion` calls.

**Batch 1 -- Scope & Mode (3 questions in one call):**

| # | Header | Question | Options |
|---|--------|----------|---------|
| 1 | `Deck` | "Which deck(s) to review? (leave blank for all)" | List from `mcp__anki__deckActions` (getDeckNames) |
| 2 | `Loop-Kind` | "What to focus on?" | "Audit existing cards (fix quality)", "Generate missing cards (fill gaps)", "Both (audit + generation)" |
| 3 | `Iterations` | "How many cards to process?" | "10 (quick session)", "25 (standard)", "50 (deep)", "Unlimited (run until interrupted)" |

**Batch 2 -- Extras & Launch (3 questions in one call):**

| # | Header | Question | Options |
|---|--------|----------|---------|
| 4 | `Scanners` | "Enable extra scanners? (audit always runs)" | "None (fast)", "FSRS retention signals", "Duplicate detection", "LLM review (slow, costs tokens)", "All" |
| 5 | `Apply` | "Apply fixes or dry-run?" | "Auto (fix cards + sync to Anki)", "Dry-run (report issues only)" |
| 6 | `Launch` | "Ready to start?" | "Launch", "Edit config", "Cancel" |

**YOU MUST NOT start any loop or execution without completing interactive setup when config is missing.**

## Usage Limit Detection

The loop monitors Claude Code subscription usage via the OMC HUD cache to avoid
wasting iterations when the model is about to be throttled or downgraded.

### How It Works

The OMC HUD polls `api.anthropic.com/api/oauth/usage` every 90 seconds and caches
the result. The skill reads this cache file -- a lightweight disk read, no API call.

**Cache location (find dynamically):**
```bash
USAGE_CACHE=$(find ~/.claude/plugins -name ".usage-cache.json" -path "*/oh-my-claudecode/*" 2>/dev/null | head -1)
```

**Cache structure (relevant fields):**
```json
{
  "timestamp": 1743782400000,
  "data": {
    "fiveHourPercent": 45,
    "weeklyPercent": 12,
    "fiveHourResetsAt": "2026-04-04T17:30:00Z",
    "weeklyResetsAt": "2026-04-08T10:00:00Z"
  },
  "error": false,
  "source": "anthropic"
}
```

### Limit Check Procedure

Run this check at the START of every Phase 1 (Review), BEFORE picking the next item:

```bash
# 1. Find the cache file
USAGE_CACHE=$(find ~/.claude/plugins -name ".usage-cache.json" -path "*/oh-my-claudecode/*" 2>/dev/null | head -1)

# 2. If not found, skip limit checks (HUD not installed)
if [ -z "$USAGE_CACHE" ]; then
  echo "WARN: Usage monitoring unavailable -- HUD cache not found"
  # Continue without limit checks
fi

# 3. Read and parse
cat "$USAGE_CACHE" | jq '{
  five_hour: .data.fiveHourPercent,
  weekly: .data.weeklyPercent,
  five_hour_resets: .data.fiveHourResetsAt,
  weekly_resets: .data.weeklyResetsAt,
  cache_age_s: ((now - (.timestamp / 1000)) | floor),
  has_error: .error
}'
```

### Decision Table

| Condition | Action |
|-----------|--------|
| Cache file not found | Log warning, continue without checks |
| `cache_age_s > 900` (stale >15 min) | Log warning, continue with caution |
| `error == true` | Log warning, skip check |
| `five_hour < 80` AND `weekly < 80` | Continue normally |
| `five_hour >= 80` OR `weekly >= 80` (but < 95) | Log warning: "Usage at N% -- approaching limit", continue |
| `five_hour >= 95` OR `weekly >= 95` | **GRACEFUL STOP**: write checkpoint, print summary, stop |

### Graceful Stop Procedure

When usage >= 95% on either window:

1. **Do NOT start the next iteration** -- stop at Phase 1 before Phase 2
2. Write checkpoint file (see Checkpoint & Resume section below)
3. Print stop message:
   ```
   === Usage Limit Reached ===
   Five-hour window: <N>% (resets at <time>)
   Weekly window: <N>% (resets at <time>)
   
   Checkpoint saved to .cardloop/review-checkpoint.json
   Resume later with: /card-review-cycle --resume
   ```
4. Print the standard progress summary (baseline -> current, keeps/discards)
5. STOP the loop

**CRITICAL:** Never stop between Phase 3 (Fix) and Phase 8 (Decide). Always complete
the current iteration before checking limits. The check happens at the START of Phase 1.

## Checkpoint & Resume

### Checkpoint File

**Location:** `.cardloop/review-checkpoint.json` (gitignored)

Written on graceful stop (usage limit or any future pause reason). Contains all state
needed to resume without re-asking the user.

```json
{
  "version": 1,
  "paused_at": "2026-04-04T15:30:00Z",
  "reason": "five_hour_limit_95_percent",
  "resume_after": "2026-04-04T17:30:00Z",
  "iteration": 23,
  "max_iterations": 50,
  "baseline_metric": 45.0,
  "current_metric": 67.2,
  "last_item_id": "abc123def456",
  "last_item_status": "keep",
  "resume_count": 0,
  "config": {
    "deck": "Kotlin",
    "topic": null,
    "loop_kind": "audit",
    "scanners": ["fsrs"],
    "apply": "auto"
  },
  "stats": {
    "keeps": 15,
    "discards": 6,
    "crashes": 1,
    "skipped": 1
  }
}
```

### Resume Logic (runs at the very start of Phase 0)

When invoked with `--resume` OR when a checkpoint file exists:

```
1. Read .cardloop/review-checkpoint.json
2. IF file does not exist:
     Print "No checkpoint found. Starting fresh."
     Proceed to normal Phase 0

3. IF checkpoint.paused_at is older than 7 days:
     Print "Stale checkpoint (>7 days old) -- discarding"
     Delete checkpoint file
     Proceed to normal Phase 0

4. IF checkpoint.paused_at is older than 24 hours:
     ASK user: "Found checkpoint from <N>h ago. Resume or start fresh?"
     IF user says fresh: delete checkpoint, proceed to normal Phase 0

5. Check current usage limits:
     Read OMC HUD cache
     IF five_hour >= 95 OR weekly >= 95:
       Print "Still limited (five-hour: <N>%, weekly: <N>%)"
       Print "Resets at: <time>"
       STOP (do not delete checkpoint -- user will retry later)

6. IF checkpoint.resume_count >= 3 AND time_since_pause < 5 minutes:
     Print "Repeated immediate re-hits detected (3+ resumes within 5 min)"
     Print "Wait for the limit window to reset before resuming."
     STOP (do not delete checkpoint)

7. Restore state from checkpoint:
     current_iteration = checkpoint.iteration
     max_iterations = checkpoint.max_iterations
     config = checkpoint.config
     stats = checkpoint.stats
     Increment resume_count in checkpoint (write back before deleting)

8. Print:
     "Resuming card review cycle from iteration <N>"
     "Paused: <reason> at <time>"
     "Config: Deck=<deck>, Iterations=<max>"
     "Progress: <keeps> keeps, <discards> discards"

9. Delete checkpoint file (state is now in memory)
10. Skip to Phase 1 (do NOT re-run Phase 0 baseline/scan)
```

### What Already Persists (no extra work)

| State | Location | Survives restart? |
|-------|----------|-------------------|
| Work item queue | `.cardloop/state.db` (SQLite) | Yes |
| Item statuses | `.cardloop/state.db` | Yes |
| Score history | `.cardloop/progression.jsonl` | Yes |
| Iteration log | `card-review-results.tsv` | Yes |
| Git experiment history | Git commits | Yes |

The checkpoint only adds: iteration counter, original config, cumulative stats, and
pause reason -- everything else already persists automatically.

## Phase 0: Baseline (do once)

### 0.0 Resume Check (runs FIRST, before anything else)

Before starting a fresh session, check for an existing checkpoint:

```bash
# Check for checkpoint
if [ -f .cardloop/review-checkpoint.json ]; then
  echo "Checkpoint found"
  cat .cardloop/review-checkpoint.json | jq '{iteration, reason, paused_at, resume_after}'
fi
```

Follow the resume logic described in "Checkpoint & Resume" section above.
If resuming: skip to Phase 1 with restored state.
If no checkpoint or user chose fresh start: continue to 0.1.

### 0.1 Precondition Checks

```bash
# 1. Verify git repo is clean
git status --porcelain
# If dirty: warn user and ask to stash or commit first

# 2. Ensure cardloop directory exists
mkdir -p .cardloop

# 3. Verify CLI is available
cargo run --bin anki-atlas -- --help 2>&1 | head -1
# If fails: tell user to build first (cargo build --bin anki-atlas)
```

If any check fails, stop and inform the user. Do not enter the loop with broken preconditions.

### 0.2 Initial Scan

Build the scan command from config:

```bash
anki-atlas cardloop scan --registry .cardloop/cards.db \
  [--anki-collection <path>]   # if Scanners includes fsrs
  [--detect-duplicates]         # if Scanners includes duplicates
  [--llm-review]                # if Scanners includes llm-review
```

### 0.3 Read Baseline Metric

```bash
anki-atlas cardloop status --json
```

Parse the JSON output (ScoreSummary):
- `overall` (0.0-1.0) -- lenient score
- `open_count` -- items needing work
- `fixed_count` -- items resolved
- `strict_score` -- strict score (wontfix counts against)

Compute composite metric:

```
total = open_count + fixed_count
card_health = overall * 70 + (1 - open_count / max(total, 1)) * 30
```

This gives a 0-100 score where higher is better.

### 0.4 Create Results Log

Create `card-review-results.tsv` (gitignored):

```bash
echo "# metric_direction: higher_is_better" > card-review-results.tsv
echo -e "iteration\tcommit\tmetric\tdelta\tguard\tstatus\tslug\tissue_kind\tdescription" >> card-review-results.tsv
```

Add to `.gitignore` if not already present:

```bash
grep -q 'card-review-results.tsv' .gitignore || echo 'card-review-results.tsv' >> .gitignore
```

Record baseline as iteration #0:

```
0	<commit-hash>	<card_health>	0.0	-	baseline	-	-	initial state: <open_count> open, score <overall*100>%
```

### 0.5 Print Baseline Summary

```
=== Card Review Cycle: Baseline ===
Score:      <card_health>/100 (overall: <overall*100>%)
Open:       <open_count> items
Fixed:      <fixed_count> items
By tier:    T1:<n> T2:<n> T3:<n> T4:<n>
Mode:       <bounded N | unbounded>
Scanners:   <list>
Apply:      <auto | dry-run>
```

## The Loop

```
LOOP (FOREVER or N times):
  Phase 1: Review       -- situational awareness
  Phase 2: Next         -- get highest-priority work item
  Phase 3: Fix          -- one atomic change
  Phase 4: Commit       -- git commit before verification
  Phase 5: Resolve      -- close item with verification gate
  Phase 6: Verify       -- re-read composite metric
  Phase 7: Guard        -- validate changed card
  Phase 8: Decide       -- keep / discard / crash
  Phase 9: Log          -- record result
  Phase 10: Repeat      -- go to Phase 1
```

### Phase 1: Review (situational awareness)

**You MUST complete ALL steps before picking the next item.**

1. **LIMIT GATE (first!):** Run the usage limit check procedure from "Usage Limit Detection" section
   - If result is **GRACEFUL STOP** (>= 95%): write checkpoint, print summary, STOP
   - If result is **WARNING** (>= 80%): log the warning, continue
   - If result is **OK** or **UNKNOWN**: continue normally
2. Read last 10 entries from `card-review-results.tsv`
3. Run `git log --oneline -20` to see recent card improvements
4. Run `anki-atlas cardloop status` for current score dashboard
5. If bounded: check `current_iteration < max_iterations`
6. Identify patterns: what issue kinds succeed? What slug patterns fail?

**Why read git history every time?** Git IS the memory. The log shows which fixes were
kept vs reverted. Use this to inform the next iteration -- avoid repeating failed approaches.

**Why check limits first?** The limit gate runs BEFORE any new work begins. This ensures
we never start a fix we can't finish. The current iteration always completes fully before
the next limit check.

### Phase 2: Next (get work item)

```bash
anki-atlas cardloop next -n 1 [--loop-kind <audit|generation>]
```

- If no open items remain: rescan once (`anki-atlas cardloop scan --registry .cardloop/cards.db`)
- If still no items after rescan: print completion summary and STOP
- Note the `id`, `slug`, `issue_kind`, `tier`, `summary`, `detail`

**Priority order** (automatic, handled by CLI):
- Tier 1 (AutoFix) before Tier 2 (QuickFix) before Tier 3 (Rework) before Tier 4 (Delete)
- Within same tier: oldest first

### Phase 3: Fix (one atomic change)

Apply fix based on `issue_kind`. Use the strategy table below.
Make ONE focused change per iteration -- never fix multiple unrelated issues at once.

**If Apply is `dry-run`:** Skip this phase. Log the issue and move to next item.

#### Fix Strategies by IssueKind

**MissingTags** (T1 AutoFix):
```bash
anki-atlas tag-audit <file> --fix
```
Or via MCP: `mcp__anki__tagActions` with `replaceTags` action.

**LowQuality** (T2 QuickFix or T3 Rework):
1. Read the card file and the `detail` field from the work item
2. Identify the weak dimension (clarity, atomicity, testability, memorability, accuracy, relevance)
3. Rewrite the question/answer following card principles:
   - Atomic: one fact per card
   - Active recall: "what/how/why", not yes/no
   - Concise: answer under 100 words
   - Bilingual: EN + RU (Cyrillic only, never transliteration)
4. Update file + sync to Anki via `mcp__anki__updateNoteFields`

**ValidationError** (T1-T2):
1. Read the validation error from `detail`
2. Fix the specific format/content issue
3. Re-validate: `anki-atlas validate <file> --quality`

**DeadSkill** (T3 Rework or T4 Delete):
- Syntax recall -> Transform to: "When/why would you use this?" or "What goes wrong if..."
- Boilerplate -> Transform to: "What problem does this pattern solve?" or "Compare with alternatives"
- Pure lookup -> Tag with `skill::dead` and suspend via MCP
- If transformation not viable -> Delete via `mcp__anki__deleteNotes`

**Duplicate** (similarity > 0.95, T4 Delete):
- Compare both cards
- Delete the weaker one via `mcp__anki__deleteNotes`
- Keep the one with better quality/more reviews

**SemanticOverlap** (similarity 0.85-0.95, T2 QuickFix):
- Add distinguishing `[Context: ...]` prefix to question
- Or merge into a single better card
- Update via `mcp__anki__updateNoteFields`

**SplitCandidate** (T3 Rework):
- Split into N atomic cards (one fact each)
- Create new cards via `mcp__anki__addNote`
- Update original via `mcp__anki__updateNoteFields`

**StaleContent** (T2 QuickFix):
- Read the source Obsidian note (from `source_path` in work item)
- Update card content to match current note
- Sync via `mcp__anki__updateNoteFields`

**UncoveredTopic** (T3 Rework):
- Read the source note from `source_path`
- Generate EN + RU card pair following generate-cards skill conventions
- Add via `mcp__anki__addNote` for each card
- Or use: `anki-atlas generate <source-note.md>`

**MissingLanguage** (T2 QuickFix):
- Find the existing card (EN or RU)
- Create the missing bilingual pair
- RU must be Cyrillic only (never transliteration)
- Add via `mcp__anki__addNote`

### Phase 4: Commit (before verification)

```bash
# Stage ONLY modified files (never git add -A)
git add <file1> <file2> ...

# Check if there's actually something to commit
git diff --cached --quiet
# Exit code 0 = no changes -> log as "no-op", skip to next iteration

# Commit with descriptive message
git commit -m "experiment(cards): <slug> - <one-sentence description>"
```

**Nothing to commit:** If `git diff --cached --quiet` shows no changes, this is NOT a crash.
Log as `status=no-op` and proceed to next iteration.

**Hook failure:** If a pre-commit hook blocks the commit:
1. Read the hook's error output
2. If fixable (lint, formatting): fix and retry
3. If not fixable within 2 attempts: log as `hook-blocked`, revert changes, move on
4. NEVER use `--no-verify`

### Phase 5: Resolve (close item with verification gate)

```bash
anki-atlas cardloop resolve <id> \
  --status fixed \
  --attest "<what was changed and why>" \
  --registry .cardloop/cards.db
```

The `--registry` flag triggers automatic re-scan of the card:
- If issue no longer exists: item closes as `fixed`
- If issue persists: item is **auto-reopened** -- treat this as failed verification

For skip/wontfix:
```bash
anki-atlas cardloop resolve <id> --status skipped --attest "<reason>"
anki-atlas cardloop resolve <id> --status wontfix --attest "<reason>"
```

### Phase 6: Verify (re-read metric)

```bash
anki-atlas cardloop status --json
```

Compute new `card_health` using same formula as baseline.
Calculate `delta = new_card_health - previous_card_health`.

### Phase 7: Guard (validate changed card)

```bash
anki-atlas validate <changed-file> --quality
```

Check for new validation errors introduced by the fix.
Guard is a soft check -- new warnings are acceptable, new errors are not.

### Phase 8: Decide (no ambiguity)

```
IF metric_improved_or_unchanged AND resolve_succeeded AND guard_passed:
    STATUS = "keep"
    # Commit stays. Git history preserves this success.

ELIF resolve_auto_reopened (verification gate failed):
    STATUS = "discard"
    git revert HEAD --no-edit
    # If revert conflicts: git revert --abort && git reset --hard HEAD~1

ELIF metric_regressed OR guard_introduced_errors:
    STATUS = "discard"
    git revert HEAD --no-edit

ELIF crashed:
    # Attempt fix (max 3 tries)
    IF fixable:
        Fix -> re-commit -> re-resolve -> re-verify
    ELSE:
        STATUS = "crash"
        git revert HEAD --no-edit
```

**Simplicity override:** If metric barely changed but card is clearly simpler/better,
treat as "keep". If metric improved but card is now confusing, treat as "discard".

**Rollback preference:** Always prefer `git revert` over `git reset --hard`.
Revert preserves the experiment in history so future iterations can learn from it.
Only use `git reset --hard HEAD~1` if revert produces merge conflicts.

### Phase 9: Log (record result)

Append to `card-review-results.tsv`:

```
<iteration>	<commit|->	<metric>	<delta>	<pass|fail|->	<status>	<slug>	<issue_kind>	<description>
```

**Valid statuses:** `keep`, `discard`, `crash`, `no-op`, `hook-blocked`, `skipped`, `dry-run`

**Every 5 iterations**, print a progress summary:

```
=== Card Review Progress (iteration N) ===
Baseline: <baseline>% -> Current: <current>% (+<delta>%)
Open: <open_count> remaining
Keeps: X | Discards: Y | Crashes: Z | Skipped: W
Last 5: keep, discard, keep, keep, no-op
```

### Phase 10: Repeat

**Unbounded mode (default):**
Go to Phase 1. NEVER STOP. NEVER ASK "should I continue?"

**Bounded mode (Iterations: N):**
```
IF current_iteration < max_iterations:
    Go to Phase 1
ELIF open_count == 0:
    Print: "Queue empty at iteration N! All items resolved."
    Print final summary. STOP.
ELSE:
    Print final summary. STOP.
```

**Usage limit stop (triggered at Phase 1):**
When the limit gate fires at Phase 1, the loop writes a checkpoint and stops.
The stop happens BEFORE the current iteration begins (not mid-iteration).
See "Usage Limit Detection" and "Checkpoint & Resume" sections above.

**Final summary format:**
```
=== Card Review Cycle Complete (N/N iterations) ===
Baseline: <baseline>/100 -> Final: <current>/100 (<delta>)
Open: <start_open> -> <end_open> (<diff>)
Keeps: X | Discards: Y | Crashes: Z | Skipped: W
Best iteration: #N -- <description>
Top issue kinds fixed: MissingTags (X), LowQuality (Y), ...
```

**If stopped by usage limit, append:**
```
Stopped: usage limit (<window> at <N>%)
Checkpoint: .cardloop/review-checkpoint.json
Resume after: <resets_at time>
Command: /card-review-cycle --resume
```

## Git as Memory

Every iteration MUST read `git log --oneline -20` to:

1. **Avoid repeating failures:** If a slug was already discarded, try a different fix approach
2. **Exploit successes:** If tag fixes keep working, prioritize T1 items
3. **Detect patterns:** If 3+ consecutive discards on LowQuality rewrites, switch to T1/T4 items

**When stuck (5+ consecutive discards):**
1. Re-read the full cardloop status
2. Rescan: `anki-atlas cardloop scan --registry .cardloop/cards.db`
3. Try a different tier (if stuck on T3, switch to T1)
4. Try a different loop-kind (if stuck on audit, try generation)
5. Try working a different cluster (`--cluster <id>`)

## Critical Rules

1. **Loop until done** -- Unbounded: loop forever. Bounded: loop N times then summarize.
2. **Read before write** -- Always read the card file and work item detail before fixing
3. **One fix per iteration** -- Atomic changes. If it breaks, you know exactly why.
4. **Mechanical verification only** -- Use `cardloop status --json` scores. No subjective "looks good".
5. **Automatic rollback** -- Failed fixes revert instantly via `git revert`.
6. **Simplicity wins** -- Equal results + simpler card = KEEP. Tiny improvement + confusing card = DISCARD.
7. **Git is memory** -- Every experiment committed with `experiment(cards):` prefix. Read `git log` before each iteration.
8. **When stuck, rescan** -- Re-read files, rescan queue, try different tier or cluster.
9. **Bilingual always** -- Every card needs EN + RU pair. RU in Cyrillic only.
10. **Respect the verification gate** -- Always pass `--registry` to `resolve`. If it auto-reopens, the fix didn't work.

## Communication Rules

- **DO NOT** ask "should I keep going?" -- in unbounded mode, YES. ALWAYS.
- **DO NOT** summarize after each iteration -- just log and continue
- **DO** print a brief 1-line status every 5 iterations
- **DO** alert if you discover something surprising (e.g., >50% dead-skill cards)
- **DO** print a final summary when bounded loop completes or queue empties

## Adapting to Different Focus Areas

| Focus | Deck | Loop-Kind | Scanners | Expected Outcome |
|-------|------|-----------|----------|-----------------|
| Quick cleanup | any | audit | (none) | Fix T1/T2 tag and format issues |
| Deep quality | specific | audit | fsrs | Fix retention-based quality issues |
| Fill gaps | any | generation | (none) | Generate cards for uncovered topics |
| Full audit | specific | both | fsrs, duplicates | Comprehensive deck improvement |
| Dedup pass | any | audit | duplicates | Remove duplicate and overlapping cards |

## Example Invocations

```
# Quick 10-card audit session
/card-review-cycle
Deck: Kotlin
Iterations: 10

# Deep quality pass with FSRS signals
/card-review-cycle
Deck: Android
Loop-Kind: both
Scanners: fsrs, duplicates
Iterations: 50

# Unlimited gap-filling (stops automatically at usage limit)
/card-review-cycle
Loop-Kind: generation

# Dry-run to see what needs fixing
/card-review-cycle
Apply: dry-run
Iterations: 20

# Full audit of everything
/card-review-cycle
Scanners: fsrs, duplicates, llm-review

# Resume after usage limit pause
/card-review-cycle --resume
```
