#!/bin/bash
set -euo pipefail

# Anki Atlas Rust Rewrite - Ralph Loop Runner
# Executes specs sequentially, one ralph run per crate.
#
# Usage:
#   ./run-ralph.sh              # auto-detect remaining work and run
#   ./run-ralph.sh 19-worker    # force start from a specific spec
#   ./run-ralph.sh --status     # show completion status only

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

# Check if a crate is already implemented (has >0 tests or non-stub source)
is_crate_done() {
  local spec="$1"
  local crate_name

  case "$spec" in
    01-common)      crate_name="common" ;;
    02-taxonomy)    crate_name="taxonomy" ;;
    03-database)    crate_name="database" ;;
    04-anki-reader) crate_name="anki-reader" ;;
    05-anki-sync)   crate_name="anki-sync" ;;
    06-indexer)     crate_name="indexer" ;;
    07-search)      crate_name="search" ;;
    08-analytics)   crate_name="analytics" ;;
    09-card)        crate_name="card" ;;
    10-validation)  crate_name="validation" ;;
    11-llm)         crate_name="llm" ;;
    12-obsidian)    crate_name="obsidian" ;;
    13-rag)         crate_name="rag" ;;
    14-generator)   crate_name="generator" ;;
    15-jobs)        crate_name="jobs" ;;
    16-cli)         crate_name="anki-atlas" ;;
    17-api)         crate_name="anki-atlas-api" ;;
    18-mcp)         crate_name="anki-atlas-mcp" ;;
    19-worker)      crate_name="anki-atlas-worker" ;;
    *)              return 1 ;;
  esac

  # Check if crate has tests that pass (at least 1)
  local test_output
  test_output=$(cargo test -p "$crate_name" 2>&1 || true)
  local passed
  passed=$(echo "$test_output" | grep -oP '\d+ passed' | head -1 | grep -oP '\d+' || echo "0")

  # A crate is "done" if it has at least 1 passing test
  [[ "$passed" -gt 0 ]]
}

# Status mode
if [[ "${1:-}" == "--status" ]]; then
  echo "=== Anki Atlas Rust Rewrite Status ==="
  echo ""
  for spec in "${SPECS[@]}"; do
    if is_crate_done "$spec"; then
      echo "  [x] $spec"
    else
      echo "  [ ] $spec"
    fi
  done
  echo ""
  exit 0
fi

# Determine start point
START_FROM="${1:-}"

TOTAL=${#SPECS[@]}
COMPLETED=0
SKIPPED=0

echo "=== Anki Atlas Rust Rewrite ==="
echo ""

# Auto-detect completed crates if no explicit start
if [[ -z "$START_FROM" ]]; then
  echo "Auto-detecting completed crates..."
  echo ""
fi

for spec in "${SPECS[@]}"; do
  # If explicit start given, skip until we reach it
  if [[ -n "$START_FROM" ]]; then
    if [[ "$spec" != "$START_FROM" && "$SKIPPED" -eq 0 ]] || \
       [[ "$SKIPPED" -gt 0 && "$spec" != "$START_FROM" && "$COMPLETED" -eq 0 ]]; then
      # Simple skip: skip everything before START_FROM
      if [[ "$spec" != "$START_FROM" ]]; then
        echo "Skipping: $spec (resuming from $START_FROM)"
        ((SKIPPED++))
        continue
      fi
    fi
    START_FROM=""  # Found it, clear the flag
  elif is_crate_done "$spec"; then
    echo "  Already done: $spec"
    ((COMPLETED++))
    continue
  fi

  REMAINING=$((TOTAL - COMPLETED))
  echo ""
  echo "================================================================"
  echo "=== [$((COMPLETED + 1))/$TOTAL] Starting: $spec ($REMAINING remaining)"
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
echo "=== ALL CRATES COMPLETE ==="
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. cargo test --workspace"
echo "  2. cargo clippy --workspace -- -D warnings"
echo "  3. cargo build --release"
echo "  4. Remove Python source (packages/, apps/, tests/)"
echo "  5. Update CI/CD pipeline"
