---
description: Audit and fix tag convention violations across the collection
argument-hint: "[--deck DeckName] [--fix] [--dry-run]"
allowed-tools: [Read, mcp__anki__getTags, mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__tagActions]
---

# Tag Audit

## Task

Scan all tags in the Anki collection (or a specific deck), identify convention violations, and optionally fix them.

## Arguments

- `--deck DeckName`: Limit audit to a specific deck
- `--fix`: Apply fixes automatically
- `--dry-run`: Show what would change without modifying anything (default behavior)

## Process

### 1. Fetch All Tags

```
mcp__anki__getTags
```

Record total tag count for before/after comparison.

### 2. Classify Tags

Place each tag into one of these categories:

| Category | Rule | Examples |
|----------|------|---------|
| **Conforming** | Matches `prefix::kebab-case` or valid unprefixed kebab-case | `android::compose`, `state-management` |
| **Non-conforming** | Violates format rules | `cs_algorithms`, `my_tag`, `android/compose` |
| **Irreducible** | Special formats preserved as-is | `slug:bias-*`, `bias_*` |
| **Code identifier** | API/class names in original casing | `ArrayList`, `WorkManager`, `SQLDelight` |

### 3. Check Convention Rules

For each non-irreducible, non-code-identifier tag, check:

| Rule | Violation Pattern | Fix |
|------|-------------------|-----|
| Underscore in prefix | `cs_algorithms` | `cs::algorithms` |
| Underscore in words | `my_custom_tag` | `my-custom-tag` |
| Slash as separator | `android/compose` | `android::compose` |
| Uppercase prefix | `Android::Compose` | `android::compose` |
| Deep nesting (>2 levels) | `a::b::c::d` | Flatten to `a::b-c-d` |
| Missing domain prefix | Known concept without prefix | Suggest prefixed form |
| Status/process tags | `new`, `verified`, `batch-1` | Flag for removal |

### 4. Count Affected Cards

For each violation, count affected cards:

```
mcp__anki__findNotes with query "tag:violating-tag"
```

If `--deck` is specified, add `deck:DeckName` to the query.

### 5. Generate Report

```
## Tag Audit Report

### Summary
- Total tags scanned: [count]
- Conforming: [count]
- Non-conforming: [count]
- Irreducible (skipped): [count]
- Code identifiers (skipped): [count]

### Violations Found

| Tag | Issue | Suggested Fix | Cards Affected |
|-----|-------|---------------|----------------|
| cs_algorithms | Underscore prefix separator | cs::algorithms | 15 |
| cs_distributed_systems | Underscore prefix separator | cs::distributed-systems | 8 |
| my_tag | Underscore word separator | my-tag | 3 |
| ... | ... | ... | ... |

### Actions
- [--fix]: Will apply all suggested fixes
- [--dry-run]: No changes made (report only)
```

### 6. Apply Fixes (if --fix or user confirms)

For each violation:

```
mcp__anki__tagActions with replaceTags:
  - Find all notes with old tag
  - Add new tag to those notes
  - Remove old tag from those notes
```

After all replacements:
```
mcp__anki__tagActions with clearUnusedTags
```

### 7. Summary

```
Tag Audit Complete
──────────────────
Before: [count] tags
After: [count] tags
Fixed: [count] violations
Remaining: [count] (irreducible/code identifiers)

Status: [Applied / Dry run only]
```

## Error Handling

| Error | Action |
|-------|--------|
| No tags found | Check Anki connection |
| Tag replacement fails | Report which tag failed, continue with others |
| Ambiguous fix | Present options to user instead of auto-fixing |

## Examples

```
# Dry-run audit of entire collection
/anki/tag-audit --dry-run

# Audit specific deck
/anki/tag-audit --deck "Programming::Android"

# Audit and auto-fix
/anki/tag-audit --fix

# Default (dry-run)
/anki/tag-audit
```
