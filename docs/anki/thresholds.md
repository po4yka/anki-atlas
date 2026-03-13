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
cargo run --bin anki-atlas -- search "question text" --semantic --top 5

# Find all duplicates in collection
cargo run --bin anki-atlas -- duplicates --threshold 0.95
```

## Interference Prevention

When similarity score is 0.85-0.95:
1. Add distinguishing prefix: `[StateFlow]`, `[SharedFlow]`
2. Create a comparison card: "X vs Y"
3. Skip if truly duplicate content

## Hash-Based Detection

| Status | Detection | Action |
|--------|-----------|--------|
| SYNCED | slug + hash match | None |
| LOCAL_ONLY | slug in local, not Anki | Sync |
| HASH_MISMATCH | same slug, different hash | Update |
| ORPHAN | source_path deleted | Delete |
