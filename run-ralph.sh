#!/bin/bash
set -euo pipefail

# Anki Atlas Rust Rewrite - Ralph Loop Runner
# Executes specs sequentially, one ralph run per crate.
#
# Usage:
#   ./run-ralph.sh              # run all specs from the beginning
#   ./run-ralph.sh 06-indexer   # resume from a specific spec

SPECS=(
  # Phase 1: Foundation (no inter-crate deps)
  01-common
  02-taxonomy

  # Phase 2: Data layer
  03-database
  04-anki-reader

  # Phase 3: Core services
  05-anki-sync
  06-indexer
  07-search

  # Phase 4: Domain crates
  08-analytics
  09-card
  10-validation
  11-llm
  12-obsidian
  13-rag
  14-generator
  15-jobs

  # Phase 5: Binary crates
  16-cli
  17-api
  18-mcp
  19-worker
)

# Optional: resume from a specific spec
START_FROM="${1:-}"
SKIP=false
if [[ -n "$START_FROM" ]]; then
  SKIP=true
fi

TOTAL=${#SPECS[@]}
COMPLETED=0

echo "=== Anki Atlas Rust Rewrite ==="
echo "=== $TOTAL crates to build via TDD ralph loops ==="
echo ""

for spec in "${SPECS[@]}"; do
  if [[ "$SKIP" == "true" ]]; then
    if [[ "$spec" == "$START_FROM" ]]; then
      SKIP=false
    else
      echo "Skipping: $spec (resuming from $START_FROM)"
      ((COMPLETED++))
      continue
    fi
  fi

  echo ""
  echo "================================================================"
  echo "=== [$((COMPLETED + 1))/$TOTAL] Starting: $spec"
  echo "================================================================"
  echo ""

  # Point ralph to the current spec
  echo "${spec}.md" > specs/CURRENT_SPEC.txt
  git add specs/CURRENT_SPEC.txt
  git commit -m "ralph: start $spec" --allow-empty

  # Run the ralph loop for this crate
  ralph run --config presets/tdd-rewrite.yml

  ((COMPLETED++))

  echo ""
  echo "=== Completed: $spec ($COMPLETED/$TOTAL) ==="
  echo ""
done

echo ""
echo "================================================================"
echo "=== ALL $TOTAL CRATES COMPLETE ==="
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. cargo test --workspace"
echo "  2. cargo clippy --workspace -- -D warnings"
echo "  3. cargo build --release"
echo "  4. Remove Python source (packages/, apps/, tests/)"
echo "  5. Update CI/CD pipeline"
