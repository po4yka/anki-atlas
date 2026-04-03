---
name: project-audit
description: "Run project-wide or per-crate convention audits. Checks clippy, fmt, unwrap/expect, missing #[instrument], wildcard re-exports. Triggers on: audit, project audit, convention check, code health, crate audit"
allowed-tools: Read, Glob, Grep, Bash
---

# Project Audit

Run convention-compliance audits across the anki-atlas workspace.

## When to Use

- Checking crate health before a release or merge
- Periodic convention drift detection
- After major refactors to verify compliance
- When the user says "audit", "code health", or "convention check"

## When NOT to Use

- Running CI checks -> use `/act-ci`
- Fixing specific CI failures -> use `/ci:fix-ci`
- Writing tests -> use `/tdd`

## Single Crate Audit

Run the audit script against one crate:

```bash
./scripts/audit_crate.sh <crate-name>
```

Example:

```bash
./scripts/audit_crate.sh common
./scripts/audit_crate.sh search
./scripts/audit_crate.sh anki-atlas  # CLI binary
```

## Full Workspace Audit

Use the Ralph preset to audit all crates systematically:

```bash
ralph run -c presets/project-audit.yml
```

This runs a Scanner -> Auditor -> Reporter loop that:
1. Picks the next un-audited crate
2. Runs `scripts/audit_crate.sh` against it
3. Writes a report to `.audit/<crate>.md`
4. Repeats until all crates are covered

## Audit Checks

| Check | Severity | Convention |
|-------|----------|------------|
| Clippy warnings | FAIL | `cargo clippy -- -D warnings` must be clean |
| Format violations | FAIL | `cargo fmt --check` must pass |
| `unwrap()`/`expect()` in lib code | WARN | No unwrap/expect in library crates |
| Missing `#[instrument]` on pub async fn | WARN | All public async functions should be traced |
| Wildcard re-exports (`pub use *`) | WARN | Explicit re-exports only |

## Severity Levels

- **PASS**: Zero issues across all checks
- **WARN**: Only convention drift issues (unwrap, instrument, wildcards)
- **FAIL**: Clippy or format failures (hard gates)

## Output

Reports are written to `.audit/<crate-name>.md` with per-check results,
occurrence counts, and prioritized recommendations.

## Related

- [act-ci](../act-ci/SKILL.md) - Run CI checks locally
- [tdd](../tdd/SKILL.md) - TDD workflow for fixes
- `presets/project-audit.yml` - Ralph automation preset
