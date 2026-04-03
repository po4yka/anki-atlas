#!/usr/bin/env bash
# audit_crate.sh -- Run convention-compliance checks against a single crate.
# Usage: ./scripts/audit_crate.sh <crate-name>
# Exits 0 even when issues are found (audit, not gate).

set -euo pipefail

CRATE="${1:?Usage: audit_crate.sh <crate-name>}"

# Resolve crate path (crates/ or bins/) and cargo package name
if [ -d "crates/$CRATE" ]; then
    CRATE_DIR="crates/$CRATE"
elif [ -d "bins/$CRATE" ]; then
    CRATE_DIR="bins/$CRATE"
else
    # Try matching by cargo package name (e.g., anki-atlas-cli -> bins/cli)
    CRATE_DIR=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
        | jq -r --arg name "$CRATE" '.packages[] | select(.name == $name) | .manifest_path' \
        | sed 's|/Cargo.toml||' \
        | head -1)
    # Make path relative if absolute
    CRATE_DIR="${CRATE_DIR#"$(pwd)/"}"
    if [ -z "$CRATE_DIR" ] || [ ! -d "$CRATE_DIR" ]; then
        echo "ERROR: Cannot find crate '$CRATE' in crates/ or bins/"
        echo "Hint: use directory name (e.g., 'cli') or cargo package name (e.g., 'anki-atlas-cli')"
        exit 1
    fi
fi

# Resolve cargo package name from Cargo.toml ([package] section)
CARGO_PKG=$(awk '/^\[package\]/{found=1} found && /^name/{gsub(/name = "|"/, ""); print; exit}' "$CRATE_DIR/Cargo.toml")

echo "=== Audit: $CRATE ==="
echo "Path: $CRATE_DIR"
echo "Package: $CARGO_PKG"
echo ""

ISSUES=0

# --- Check 1: Clippy ---
echo "--- Clippy ---"
CLIPPY_OUTPUT=$(cargo clippy -p "$CARGO_PKG" -- -D warnings 2>&1) && CLIPPY_OK=true || CLIPPY_OK=false
# -D warnings promotes warnings to errors, so count both patterns
CLIPPY_COUNT=$(echo "$CLIPPY_OUTPUT" | grep -cE "(warning|error)\[" || true)
if $CLIPPY_OK; then
    echo "PASS (0 warnings)"
else
    echo "FAIL ($CLIPPY_COUNT issues)"
    echo "$CLIPPY_OUTPUT" | grep -E "(warning|error)\[" | head -10
    ISSUES=$((ISSUES + CLIPPY_COUNT))
fi
echo ""

# --- Check 2: Format ---
echo "--- Format ---"
FMT_OUTPUT=$(cargo fmt -p "$CARGO_PKG" -- --check 2>&1) && FMT_OK=true || FMT_OK=false
FMT_DIFF=$(echo "$FMT_OUTPUT" | grep -c "^Diff in" || true)
if $FMT_OK; then
    echo "PASS (0 violations)"
else
    echo "FAIL ($FMT_DIFF files need formatting)"
    echo "$FMT_OUTPUT" | grep "^Diff in" | head -10
    ISSUES=$((ISSUES + FMT_DIFF))
fi
echo ""

# --- Check 3: unwrap/expect in lib code (not tests) ---
echo "--- unwrap/expect in lib code ---"
# Exclude test files and inline #[cfg(test)] blocks via awk.
# The awk script tracks whether we are inside a test module and skips those lines.
UNWRAP_COUNT=0
if [ -d "$CRATE_DIR/src" ]; then
    # Use awk to strip inline test blocks, then grep for unwrap/expect
    UNWRAP_HITS=$(find "$CRATE_DIR/src" -name '*.rs' \
        -not -name '*_test.rs' \
        -not -name 'tests.rs' \
        -not -path '*/tests/*' \
        -print0 \
        | xargs -0 awk '
        /^[[:space:]]*#\[cfg\(test\)\]/ { in_test=1; depth=0; next }
        in_test && /{/ { depth++; next }
        in_test && /}/ { depth--; if (depth <= 0) { in_test=0 }; next }
        in_test { next }
        /\.unwrap\(\)|\.expect\(/ { print FILENAME ":" NR ": " $0 }
    ' 2>/dev/null || true)
    # Count separately to avoid empty-string issues
    if [ -n "$UNWRAP_HITS" ]; then
        UNWRAP_COUNT=$(echo "$UNWRAP_HITS" | wc -l | tr -d ' ')
    fi
    if [ "$UNWRAP_COUNT" -gt 0 ]; then
        echo "WARN ($UNWRAP_COUNT occurrences)"
        echo "$UNWRAP_HITS" | head -10
    else
        echo "PASS (0 occurrences)"
    fi
else
    echo "SKIP (no src/ directory)"
fi
echo ""

# --- Check 4: pub async fn missing #[instrument] ---
echo "--- Missing #[instrument] on pub async fn ---"
MISSING_INSTRUMENT=0
if [ -d "$CRATE_DIR/src" ]; then
    # Accumulate attribute blocks to handle multi-line attribute stacks like:
    #   #[instrument]
    #   #[allow(clippy::too_many_arguments)]
    #   pub async fn my_fn(...)
    INSTRUMENT_HITS=$(find "$CRATE_DIR/src" -name '*.rs' -not -path '*tests*' -print0 \
        | xargs -0 awk '
        /^[[:space:]]*#\[/ { attr_block = attr_block " " $0; next }
        /pub async fn/ {
            if (attr_block !~ /#\[instrument/) {
                print FILENAME ":" NR ": " $0
                count++
            }
            attr_block = ""
        }
        { attr_block = "" }
    ' 2>/dev/null || true)
    if [ -n "$INSTRUMENT_HITS" ]; then
        MISSING_INSTRUMENT=$(echo "$INSTRUMENT_HITS" | wc -l | tr -d ' ')
    fi
    if [ "$MISSING_INSTRUMENT" -gt 0 ]; then
        echo "WARN ($MISSING_INSTRUMENT functions)"
        echo "$INSTRUMENT_HITS" | head -10
    else
        echo "PASS (0 functions)"
    fi
else
    echo "SKIP (no src/ directory)"
fi
echo ""

# --- Check 5: Wildcard re-exports (pub use *) ---
echo "--- Wildcard re-exports ---"
WILDCARD_COUNT=0
if [ -d "$CRATE_DIR/src" ]; then
    WILDCARD_HITS=$(grep -rn 'pub use .*\*' "$CRATE_DIR/src/" --include='*.rs' || true)
    if [ -n "$WILDCARD_HITS" ]; then
        WILDCARD_COUNT=$(echo "$WILDCARD_HITS" | wc -l | tr -d ' ')
    fi
    if [ "$WILDCARD_COUNT" -gt 0 ]; then
        echo "WARN ($WILDCARD_COUNT occurrences)"
        echo "$WILDCARD_HITS" | head -5
    else
        echo "PASS (0 occurrences)"
    fi
else
    echo "SKIP (no src/ directory)"
fi
echo ""

# --- Summary ---
TOTAL=$((ISSUES + UNWRAP_COUNT + MISSING_INSTRUMENT + WILDCARD_COUNT))
echo "=== Summary ==="
echo "Crate: $CRATE"
echo "Total issues: $TOTAL"
if ! $CLIPPY_OK || ! $FMT_OK; then
    echo "Status: FAIL"
elif [ "$TOTAL" -gt 0 ]; then
    echo "Status: WARN"
else
    echo "Status: PASS"
fi
