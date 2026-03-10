# Historical Note: CLI App Plan

This document described an earlier Python-era plan for extending `apps/cli/`.

It is no longer the current implementation reference.

Use these documents instead:

- [CLI Spec](/Users/po4yka/GitRep/anki-atlas/specs/16-cli.md)
- [Architecture](/Users/po4yka/GitRep/anki-atlas/docs/ARCHITECTURE.md)
- [README](/Users/po4yka/GitRep/anki-atlas/README.md)

Current reality:

- the CLI lives in `bins/cli/`
- it uses clap, not Typer
- it executes sync/index directly through `crates/surface-runtime`
- preview workflows are explicit about unsupported persistence paths
