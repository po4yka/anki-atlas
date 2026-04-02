---
name: cardloop
description: Persistent card quality loop with scan/next/resolve cycle. Use for systematic card improvement with queue-driven workflow, score tracking, and audit trail.
allowed-tools: Bash, Read, Edit, Write
---

# Cardloop -- Agent Reference

This skill is injected into ralph agents running the cardloop preset.
All content is self-contained (ralph cannot follow relative links).

## Job Statement

Maximize the card quality score by working through a persistent queue of issues.
The main cycle: **scan -> status -> next -> [fix card] -> resolve -> repeat**.

The queue survives agent restarts. Progress is tracked with an audit trail and
overall/strict scores so improvement is measurable.

## CLI Invocation

Use `cargo run --bin anki-atlas --` (or `anki-atlas` if installed).

## Commands Reference

### Scan (populate the work queue)

```bash
# Basic audit + generation issues
anki-atlas cardloop scan --registry .cardloop/cards.db

# Also pull FSRS retention signals from Anki collection
anki-atlas cardloop scan --registry .cardloop/cards.db --anki-collection ~/path/to/collection.anki2

# Also run LLM batch review (slower, costs tokens)
anki-atlas cardloop scan --registry .cardloop/cards.db --llm-review

# Also detect semantic duplicates (requires Qdrant + PostgreSQL)
anki-atlas cardloop scan --registry .cardloop/cards.db --detect-duplicates
anki-atlas cardloop scan --registry .cardloop/cards.db --detect-duplicates --dup-threshold 0.85

# Full scan with all signals
anki-atlas cardloop scan \
  --registry .cardloop/cards.db \
  --anki-collection ~/path/to/collection.anki2 \
  --llm-review \
  --detect-duplicates
```

### Status (score dashboard)

```bash
# Human-readable score dashboard
anki-atlas cardloop status

# Machine-readable JSON (parse open_count, overall_score, strict_score)
anki-atlas cardloop status --json
```

### Next (get work item)

```bash
# Get the highest-priority open item
anki-atlas cardloop next

# Get multiple items (batch preview)
anki-atlas cardloop next -n 5

# Filter by loop kind
anki-atlas cardloop next --loop-kind audit
anki-atlas cardloop next --loop-kind generation

# Filter by duplicate cluster
anki-atlas cardloop next --cluster <cluster-id>
```

### Resolve (close a work item)

```bash
# Mark fixed with attestation (re-scans the card when --registry provided)
anki-atlas cardloop resolve <id> --attest "Rewrote question to test reasoning not recall" --registry .cardloop/cards.db

# Mark fixed without re-scan verification
anki-atlas cardloop resolve <id> --attest "Fixed tags"

# Skip (will not reappear unless rescan detects it again)
anki-atlas cardloop resolve <id> --status skipped --attest "Not actionable in current context"

# Won't fix (counts against strict_score, excluded from overall_score)
anki-atlas cardloop resolve <id> --status wontfix --attest "Intentional design choice"
```

### Log (audit trail)

```bash
# Show recent resolutions
anki-atlas cardloop log

# Show last N entries
anki-atlas cardloop log -n 20
```

## Workflow Phases

### Phase 1: Scan & Assess

1. Run scan to populate/refresh the queue:
   ```bash
   anki-atlas cardloop scan --registry .cardloop/cards.db
   ```
2. Check the score dashboard:
   ```bash
   anki-atlas cardloop status
   ```
3. Review `open_count`, `overall_score`, and `strict_score` to understand scope.
4. If `open_count == 0`, the queue is clear -- publish completion.

### Phase 2: Work the Queue

1. Get the next item:
   ```bash
   anki-atlas cardloop next -n 1
   ```
2. Read the work item: note the `id`, `slug`, `summary`, `issue_kind`, and `tier`.
3. Fix the card (see Fix Strategies below).
4. Resolve with attestation:
   ```bash
   anki-atlas cardloop resolve <id> \
     --status fixed \
     --attest "<what was changed and why>" \
     --registry .cardloop/cards.db
   ```
   The `--registry` flag triggers automatic re-scan of the card. If the issue
   persists, the item is auto-reopened rather than closed.
5. Git commit:
   ```bash
   git add -A && git commit -m "improve(cards): <slug> - <action>"
   ```

### Phase 3: Verify & Iterate

1. Check score improvement:
   ```bash
   anki-atlas cardloop status
   ```
2. If `open_count > 0`, loop back to Phase 2.
3. Periodically rescan to catch cascading effects (e.g., a tag fix may resolve
   multiple dependent issues):
   ```bash
   anki-atlas cardloop scan --registry .cardloop/cards.db
   ```

## Fix Strategies by Issue Kind

### MissingTags

```bash
# Audit and auto-fix tag conventions
anki-atlas tag-audit <file> --fix

# Or use MCP to update tags directly
mcp__anki__tagActions  # replaceTags action
```

### LowQuality

Edit the card file directly, or use MCP:
```
mcp__anki__updateNoteFields with id and updated fields
```

Quality checklist:
- Atomic: one fact per card
- Active recall: "what/how/why", not yes/no
- Concise: answer under 100 words
- Bilingual: EN + RU (Cyrillic only, never transliteration)

### DeadSkill

Transform or suspend:
- Syntax recall -> "When/why would you use this?" or "What goes wrong if..."
- Boilerplate -> "What problem does this pattern solve?"
- Pure lookup -> tag `skill::dead` and suspend via MCP

### Duplicate / SemanticOverlap

- Near-identical (> 0.95): delete the weaker card
- Similar (0.85-0.95): add distinguishing `[Context]` prefix or merge
- Related (0.70-0.85): add cross-reference note, no deletion needed

### UncoveredTopic

Generate cards for the gap:
```bash
anki-atlas generate path/to/source-note.md
```

## Tier System

| Tier | Label | Effort | Examples |
|------|-------|--------|---------|
| T1 | AutoFix | < 2 min | Tag convention fix, format cleanup |
| T2 | QuickFix | 2-10 min | Content tweak, question reformulation |
| T3 | Rework | 10-30 min | Card split, full rewrite, merging duplicates |
| T4 | Delete | < 5 min | Dead skill removal, duplicate deletion |

Work T1 and T2 before T3 and T4 for fastest score improvement per unit of time.

## Score Tracking

| Metric | Definition |
|--------|-----------|
| `overall_score` | Lenient: excludes `wontfix` items from denominator |
| `strict_score` | Strict: `wontfix` items count against the score |
| `open_count` | Items currently in the queue |
| `resolved_count` | Items closed (fixed + skipped + wontfix) |

Target: drive `overall_score` to 1.0 and `open_count` to 0.

## Verification Gate

When `--registry` is passed to `resolve`, the system re-scans the resolved card:
- If the issue no longer exists: item closes as `fixed`.
- If the issue persists: item is auto-reopened, preventing false resolutions.

Always pass `--registry` when resolving `fixed` items to benefit from this gate.

## Tips for Agents

1. **Work in clusters**: use `--cluster <id>` to batch all duplicates in one
   cluster together, then resolve them atomically.
2. **Easiest tier first**: T1 items cost almost nothing and boost score quickly.
3. **Meaningful attestation**: write what changed and why, not just "fixed".
   The audit trail is the primary record of what was done.
4. **Re-scan after bulk changes**: a tag taxonomy fix can cascade and close
   many items at once -- rescan before picking the next item.
5. **Respect `wontfix`**: use it sparingly; it permanently impacts `strict_score`.
6. **Validate after edits**: run `anki-atlas validate <file> --quality` after
   content changes before resolving.
