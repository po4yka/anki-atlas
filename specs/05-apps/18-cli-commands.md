# Historical Note: CLI App Plan

This document described an earlier Python-era plan for extending `apps/cli/`.

It is no longer the current implementation reference.

Use these documents instead:

- [CLI Spec](specs/16-cli.md)
- [Architecture](docs/ARCHITECTURE.md)
- [README](README.md)

Current reality:

- the CLI lives in `bins/cli/`
- it uses clap, not Typer
- it executes sync/index directly through the `surface-runtime + surface-contracts` boundary
- preview workflows are explicit about unsupported persistence paths
