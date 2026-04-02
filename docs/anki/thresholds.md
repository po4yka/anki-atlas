# Similarity Thresholds Reference

## Semantic Duplicate Detection

Use these standardized thresholds across all skills:

| Score | Classification | Action |
|-------|----------------|--------|
| > 0.95 | Exact duplicate | Skip creation |
| 0.85-0.95 | Near duplicate | Add prefix `[context]` to distinguish |
| 0.70-0.85 | Related content | Consider comparison card instead |
| < 0.70 | Distinct | Safe to create |

## Search Commands

```bash
# Check for semantic duplicates before creating card
cargo run --bin anki-atlas -- search "question text" --semantic -n 5

# Find all duplicates in collection
cargo run --bin anki-atlas -- duplicates --threshold 0.95

# Filter duplicates by deck or tag
cargo run --bin anki-atlas -- duplicates --threshold 0.92 --deck "Kotlin" --verbose

# Limit number of clusters returned
cargo run --bin anki-atlas -- duplicates --threshold 0.92 --max 50
```

## Interference Prevention

When similarity score is 0.85-0.95:
1. Add distinguishing prefix: `[StateFlow]`, `[SharedFlow]`
2. Create a comparison card: "X vs Y"
3. Skip if truly duplicate content

## Content Hash

Two hash variants are used for change detection:

| Variant | Length | Input | Used For |
|---------|--------|-------|----------|
| Card-level | 6 hex chars | `"{apf_html}\|{note_type}\|{sorted_tags}"` | ContentHash field, HASH_MISMATCH detection |
| Slug-level | 12 hex chars | `"{front}\|{back}"` | Content-addressed dedup in SlugService |

Both use SHA-256, truncated to the specified length.

## Hash-Based Detection

| Status | Detection | Action |
|--------|-----------|--------|
| SYNCED | slug + hash match | None |
| LOCAL_ONLY | slug in local, not Anki | Sync |
| HASH_MISMATCH | same slug, different hash | Update |
| ORPHAN | source_path deleted | Delete |
